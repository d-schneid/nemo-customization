import json
from typing import Callable, Dict, Union
from dataclasses import dataclass, field

from structure_aware_mcore_gpt_model import StructureAwareMCoreGPTModel
from structure_aware_layer_spec import structure_aware_layer_spec

import torch
from megatron.core.transformer.spec_utils import ModuleSpec

from nemo.collections.llm import Starcoder2Config3B, GPTConfig
from nemo.utils.import_utils import safe_import
from nemo.utils import logging
from nemo.lightning import get_vocab_size

_, HAVE_TE = safe_import("transformer_engine")


def get_batch_on_this_context_parallel_rank(batch) -> Dict[str, torch.Tensor]:
	from megatron.core import parallel_state

	if (cp_size := parallel_state.get_context_parallel_world_size()) > 1:
		num_valid_tokens_in_ub = None
		if 'loss_mask' in batch and batch['loss_mask'] is not None:
			num_valid_tokens_in_ub = batch['loss_mask'].sum()

		cp_rank = parallel_state.get_context_parallel_rank()
		for key, val in batch.items():
			if val is not None:
				seq_dim = 1 if key != 'attention_mask' else 2
				_val = val.view(
					*val.shape[0:seq_dim],
					2 * cp_size,
					val.shape[seq_dim] // (2 * cp_size),
					*val.shape[(seq_dim + 1) :],
				)
				index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).to(
					_val.device, non_blocking=True
				)
				_val = _val.index_select(seq_dim, index)
				_val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
				batch[key] = _val
		batch['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub
	return batch


def structure_aware_gpt_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
	from megatron.core import parallel_state

	batch = next(dataloader_iter)

	_batch: dict
	if isinstance(batch, tuple) and len(batch) == 3:
		_batch = batch[0]
	else:
		_batch = batch

	required_device_keys = set()
	required_host_keys = set()

	required_device_keys.add("attention_mask")
	if 'cu_seqlens' in _batch:
		required_device_keys.add('cu_seqlens')
		required_host_keys.add('cu_seqlens_argmin')
		required_host_keys.add('max_seqlen')

	if parallel_state.is_pipeline_first_stage():
		required_device_keys.update(("code_token_ids", "code_token_pos_ids", "code_token_rel_pos_ids",
									 "ll_sims", "lr_paths_types", "lr_paths_len", "dfg_node_mask"))
	if parallel_state.is_pipeline_last_stage():
		required_device_keys.update(("labels", "loss_mask"))

	_batch_required_keys = {}
	for key, val in _batch.items():
		if key in required_device_keys:
			_batch_required_keys[key] = val.cuda(non_blocking=True)
		elif key in required_host_keys:
			_batch_required_keys[key] = val.cpu()
		else:
			_batch_required_keys[key] = None

	# slice batch along sequence dimension for context parallelism
	output = get_batch_on_this_context_parallel_rank(_batch_required_keys)

	return output


def get_packed_seq_params(batch):
	from megatron.core.packed_seq_params import PackedSeqParams

	cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
	# remove -1 "paddings" added in collate_fn
	if (cu_seqlens_argmin := batch.get('cu_seqlens_argmin', None)) is not None:
		# pre-compute cu_seqlens_argmin in dataset class for perf
		cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
	else:
		cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

	# pre-compute max_seqlens in dataset class for perf
	max_seqlen = batch['max_seqlen'].squeeze() if 'max_seqlen' in batch else None

	# these args are passed eventually into TEDotProductAttention.forward()
	return PackedSeqParams(
		cu_seqlens_q=cu_seqlens,
		cu_seqlens_kv=cu_seqlens,
		max_seqlen_q=max_seqlen,
		max_seqlen_kv=max_seqlen,
		qkv_format='thd',
	)


def structure_aware_gpt_forward_step(model, batch) -> torch.Tensor:
	forward_args = {
		"code_token_ids": batch["code_token_ids"],
		"code_token_pos_ids": batch["code_token_pos_ids"],
		"code_token_rel_pos_ids": batch["code_token_rel_pos_ids"],
		"ll_sims": batch["ll_sims"],
		"lr_paths_types": batch["lr_paths_types"],
		"lr_paths_len": batch["lr_paths_len"],
		"dfg_node_mask": batch["dfg_node_mask"],
		"labels": batch["labels"],
	}

	if 'attention_mask' not in batch:
		assert (
			HAVE_TE
		), "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
			This requires Transformer Engine's implementation of fused or flash attention."
	else:
		forward_args["attention_mask"] = batch['attention_mask']

	if 'cu_seqlens' in batch:
		forward_args['packed_seq_params'] = get_packed_seq_params(batch)

	return model(**forward_args)


@dataclass
class StructureAwareStarcoder2Config(Starcoder2Config3B):

	transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = structure_aware_layer_spec
	forward_step_fn: Callable = structure_aware_gpt_forward_step
	data_step_fn: Callable = structure_aware_gpt_data_step
	num_ast_node_types: int = field(init=False)
	max_ast_depth: int = field(init=False)
	max_code_token_rel_pos: int = field(init=False)

	def __post_init__(self):
		super().__post_init__()

		with open('../../data/pretraining/metadata.json', 'r') as f_metadata:
			metadata = json.load(f_metadata)

		self.num_ast_node_types = metadata['num_ast_node_types']
		self.max_ast_depth = metadata['max_ast_depth']
		self.max_code_token_rel_pos = metadata['max_code_token_rel_pos']

	def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "StructureAwareMCoreGPTModel":
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
