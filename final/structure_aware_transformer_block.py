from contextlib import nullcontext

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import make_viewless_tensor
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

try:
	from megatron.core.extensions.transformer_engine import (
		TEDelayedScaling,
		TENorm,
		get_cpu_offload_context,
		te_checkpoint,
	)

	HAVE_TE = True
	LayerNormImpl = TENorm
except ImportError:
	HAVE_TE = False
	get_cpu_offload_context = None

	try:
		import apex  # pylint: disable=unused-import

		LayerNormImpl = FusedLayerNorm

	except ImportError:
		from megatron.core.transformer.torch_norm import WrappedTorchNorm

		LayerNormImpl = WrappedTorchNorm


class StructureAwareTransformerBlock(TransformerBlock):

	def forward(
			self,
			hidden_states: Tensor,
			attention_mask: Tensor,
			attention_bias: Tensor,
			code_token_rel_pos_ids: Tensor,
			ll_sims: Tensor,
			text_token_rel_pos_ids: Tensor = None,
			context: Tensor = None,
			context_mask: Tensor = None,
			rotary_pos_emb: Tensor = None,
			rotary_pos_cos: Tensor = None,
			rotary_pos_sin: Tensor = None,
			inference_params: InferenceParams = None,
			packed_seq_params: PackedSeqParams = None,
			sequence_len_offset: Tensor = None,
    ):
		if not self.pre_process:
			# See set_input_tensor()
			hidden_states = self.input_tensor

		# Update the inference parameters with the current batch size in case it is variable
		if inference_params and not self.training:
			inference_params.current_batch_size = hidden_states.size(1)

		# Viewless tensor.
		# - We only need to create a viewless tensor in the case of micro batch
		#   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
		#   above creates a view tensor, and '.contiguous()' is a pass-through.
		#   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
		#   the need to make it viewless.
		#
		#   However, we don't explicitly check mbs == 1 here because
		#   make_viewless_tensor() has negligible overhead when its input
		#   is already viewless.
		#
		# - For the 'else' case above, calling make_viewless_tensor() here is
		#   likely redundant, since p2p_communication.py (likely originator)
		#   already creates viewless tensors. That said, make_viewless_tensor()
		#   is called here to be future-proof and corner-case-proof.
		hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

		if self.config.sequence_parallel:
			rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
		else:
			rng_context = nullcontext()

		if self.config.fp8:
			import transformer_engine  # To keep out TE dependency when not training in fp8

			if self.config.fp8 == "e4m3":
				fp8_format = transformer_engine.common.recipe.Format.E4M3
			elif self.config.fp8 == "hybrid":
				fp8_format = transformer_engine.common.recipe.Format.HYBRID
			else:
				raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

			fp8_recipe = TEDelayedScaling(
				config=self.config,
				fp8_format=fp8_format,
				override_linear_precision=(False, False, not self.config.fp8_wgrad),
			)
			fp8_group = None
			if parallel_state.model_parallel_is_initialized():
				fp8_group = parallel_state.get_amax_reduction_group(
					with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
				)
			fp8_context = transformer_engine.pytorch.fp8_autocast(
				enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
			)
		else:
			fp8_context = nullcontext()

		with rng_context, fp8_context:
			# Forward pass.
			if self.config.recompute_granularity == 'full' and self.training:
				hidden_states = self._checkpointed_forward(
					hidden_states=hidden_states,
					attention_mask=attention_mask,
					context=context,
					context_mask=context_mask,
					rotary_pos_emb=rotary_pos_emb,
					attention_bias=attention_bias,
					packed_seq_params=packed_seq_params,
				)
			else:
				for l_no, layer in enumerate(self.layers):
					with self.offload_context:
						layer.use_cudagraph = True
						if (len(self.cuda_graphs) == 0) or (not self.training):
							hidden_states, context = layer(
								hidden_states=hidden_states,
								attention_mask=attention_mask,
								context=context,
								context_mask=context_mask,
								rotary_pos_emb=rotary_pos_emb,
								rotary_pos_cos=rotary_pos_cos,
								rotary_pos_sin=rotary_pos_sin,
								attention_bias=attention_bias,
								inference_params=inference_params,
								packed_seq_params=packed_seq_params,
								sequence_len_offset=sequence_len_offset,
								code_token_rel_pos_ids=code_token_rel_pos_ids,
								text_token_rel_pos_ids=text_token_rel_pos_ids,
								ll_sims=ll_sims,
							)
						else:
							# CUDA graph replay for layer `l_no` and microbatch
							# `self.current_microbatch`. TransformerEngine versions>=1.10
							# allow keyword arguments with CUDA graph. However, CUDA graph
							# acccepts only Tensor inputs and Tensor outputs. Hence,
							# `inference_params` and `packed_seq_params` are excluded from
							# input list while output is limited to `hidden_states`.
							cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
							assert not any(
								[inference_params, packed_seq_params]
							), "CUDA graph accepts only Tensor inputs."
							optional_inputs = self.get_cuda_graph_optional_args(
								attention_mask,
								context,
								context_mask,
								rotary_pos_emb,
								attention_bias,
								inference_params,
								packed_seq_params,
							)
							hidden_states = self.cuda_graphs[l_no][cg_index](
								hidden_states, **optional_inputs
							)

					if (
							torch.is_grad_enabled()
							and self.config.cpu_offloading
							and self.group_prefetch_offload_commit_async is not None
					):
						hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

		# Final layer norm.
		if self.final_layernorm is not None:
			hidden_states = self.final_layernorm(hidden_states)
			# TENorm produces a "viewed" tensor. This will result in schedule.py's
			# deallocate_output_tensor() throwing an error, so a viewless tensor is
			# created to prevent this.
			hidden_states = make_viewless_tensor(
				inp=hidden_states, requires_grad=True, keep_graph=True
			)

		return hidden_states
