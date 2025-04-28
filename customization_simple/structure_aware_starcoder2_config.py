from dataclasses import dataclass, field
import torch

from nemo.collections import llm
from nemo.utils.import_utils import safe_import
from megatron.core.transformer.spec_utils import ModuleSpec
from nemo.utils import logging
from nemo.lightning import get_vocab_size

from structure_aware_mcore_gpt_model import StructureAwareMCoreGPTModel

_, HAVE_TE = safe_import("transformer_engine")


@dataclass
class StructureAwareStarcoder2Config(llm.Starcoder2Config3B):

	def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreGPTModel":
		if self.enable_cuda_graph:
			assert HAVE_TE, "Transformer Engine is required for cudagraphs."
			assert getattr(self, 'use_te_rng_tracker', False), (
				"Transformer engine's RNG tracker is required for cudagraphs, it can be "
				"enabled with use_te_rng_tracker=True'."
			)

		vp_size = self.virtual_pipeline_model_parallel_size
		is_pipeline_asymmetric = getattr(self, 'account_for_embedding_in_pipeline_split', False) or getattr(
			self, 'account_for_loss_in_pipeline_split', False
		)
		if vp_size and not is_pipeline_asymmetric:
			p_size = self.pipeline_model_parallel_size
			assert (
						   self.num_layers // p_size
				   ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

		from megatron.core import parallel_state

		transformer_layer_spec = self.transformer_layer_spec
		if not isinstance(transformer_layer_spec, ModuleSpec):
			transformer_layer_spec = transformer_layer_spec(self)

		if hasattr(self, 'vocab_size'):
			vocab_size = self.vocab_size
			if tokenizer is not None:
				logging.info(
					f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
					f" {vocab_size - tokenizer.vocab_size}."
				)
		else:
			vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

		model = StructureAwareMCoreGPTModel(
			self,
			transformer_layer_spec=transformer_layer_spec,
			vocab_size=vocab_size,
			max_sequence_length=self.seq_length,
			fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
			parallel_output=self.parallel_output,
			share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
			position_embedding_type=self.position_embedding_type,
			rotary_percent=self.rotary_percent,
			rotary_base=self.rotary_base,
			seq_len_interpolation_factor=self.seq_len_interpolation_factor,
			pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
			post_process=post_process or parallel_state.is_pipeline_last_stage(),
			scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
		)

		# If using full TE layer, need to set TP, CP group since the module call
		# is not routed through megatron core, which normally handles passing the
		# TP, CP group to the TE modules.
		# Deep iterate but skip self to avoid infinite recursion.
		if HAVE_TE and self.use_transformer_engine_full_layer_spec:
			# Copied from:
			# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
			if parallel_state.get_tensor_model_parallel_world_size() > 1:
				for index, child in enumerate(model.modules()):
					if index == 0:
						continue
					if hasattr(child, "set_tensor_parallel_group"):
						tp_group = parallel_state.get_tensor_model_parallel_group()
						child.set_tensor_parallel_group(tp_group)

			if parallel_state.get_context_parallel_world_size() > 1:
				cp_stream = torch.cuda.Stream()
				for module in self.get_model_module_list():
					for index, child in enumerate(module.modules()):
						if index == 0:
							continue
						if hasattr(child, "set_context_parallel_group"):
							child.set_context_parallel_group(
								parallel_state.get_context_parallel_group(),
								parallel_state.get_context_parallel_global_ranks(),
								cp_stream,
							)

		return model
