from collections import OrderedDict
from typing import Optional, Literal

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec


class StructureAwareMCoreGPTModel(MCoreGPTModel):

	def __init__(
			self,
			config: TransformerConfig,
			transformer_layer_spec: ModuleSpec,
			vocab_size: int,
			max_sequence_length: int,
			pre_process: bool = True,
			post_process: bool = True,
			fp16_lm_cross_entropy: bool = False,
			parallel_output: bool = True,
			share_embeddings_and_output_weights: bool = False,
			position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
			rotary_percent: float = 1.0,
			rotary_base: int = 10000,
			rope_scaling: bool = False,
			rope_scaling_factor: float = 8.0,
			scatter_embedding_sequence_parallel: bool = True,
			seq_len_interpolation_factor: Optional[float] = None,
	) -> None:
		super().__init__(
			config=config,
			transformer_layer_spec=transformer_layer_spec,
			vocab_size=vocab_size,
			max_sequence_length=max_sequence_length,
			pre_process=pre_process,
			post_process=post_process,
			fp16_lm_cross_entropy=fp16_lm_cross_entropy,
			parallel_output=parallel_output,
			share_embeddings_and_output_weights=share_embeddings_and_output_weights,
			position_embedding_type=position_embedding_type,
			rotary_percent=rotary_percent,
			rotary_base=rotary_base,
			rope_scaling=rope_scaling,
			rope_scaling_factor=rope_scaling_factor,
			scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
			seq_len_interpolation_factor=seq_len_interpolation_factor
		)
		if self.pre_process:
			self.dfg_node_embedding = LanguageModelEmbedding(
				config=self.config,
				vocab_size=3, # unique embedding for all DFG nodes + padding + <SEP> token
				max_sequence_length=self.max_sequence_length,
				position_embedding_type='none',
				scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
			)

			self.ast_node_type_embedding = LanguageModelEmbedding(
				config=self.config,
				vocab_size=self.config.num_ast_node_types + 1, # padding
				max_sequence_length=self.max_sequence_length,
				position_embedding_type='none',
				scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
			)

			self.ast_node_depth_embedding = LanguageModelEmbedding(
				config=self.config,
				vocab_size=self.config.max_ast_depth,
				max_sequence_length=self.max_sequence_length,
				position_embedding_type='none',
				scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
			)

	def forward(
			self,
			code_token_ids: Tensor,
			code_token_pos_ids: Tensor,
			ll_sims: Tensor,
			lr_paths_types: Tensor,
			lr_paths_len: Tensor,
			dfg_node_mask: Tensor,
			attention_mask: Tensor,
			decoder_input: Tensor = None,
			labels: Tensor = None,
			inference_params: InferenceParams = None,
			packed_seq_params: PackedSeqParams = None,
			extra_block_kwargs: dict = None,
			runtime_gather_output: Optional[bool] = None,
	) -> Tensor:
		# If decoder_input is provided (not None), then input_ids and position_ids are ignored.
		# Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

		# Decoder embedding.
		if decoder_input is not None:
			pass
		elif self.pre_process:
			code_token_embedding = self.embedding(input_ids=code_token_ids, position_ids=code_token_pos_ids)

			ast_node_type_embedding = self.ast_node_type_embedding(input_ids=lr_paths_types, position_ids=None)
			len_longest_lr_path = lr_paths_types.shape[-1]
			node_heights = torch.tensor([[i] for i in range(len_longest_lr_path)], device=lr_paths_types.device)
			ast_node_depth_embedding = self.ast_node_depth_embedding(input_ids=node_heights, position_ids=None).unsqueeze(0)
			leaf_embedding_mult = ast_node_type_embedding * ast_node_depth_embedding

			range_tensor = torch.arange(len_longest_lr_path).view(1,1,len_longest_lr_path).to(lr_paths_types.device)
			path_mask = range_tensor < lr_paths_len.T.unsqueeze(-1)
			path_mask = path_mask.unsqueeze(-1)
			leaf_embedding_mask = leaf_embedding_mult * path_mask
			final_leaf_embedding = leaf_embedding_mask.sum(dim=2)

			dfg_node_embedding = self.dfg_node_embedding(input_ids=dfg_node_mask, position_ids=None)

			decoder_input = torch.cat((code_token_embedding, final_leaf_embedding, dfg_node_embedding), dim=0)
		else:
			# intermediate stage of pipeline
			# decoder will get hidden_states from encoder.input_tensor
			decoder_input = None

		# Rotary positional embeddings (embedding is None for PP intermediate devices)
		rotary_pos_emb = None
		rotary_pos_cos = None
		rotary_pos_sin = None
		if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
			if not self.training and self.config.flash_decode and inference_params:
				# Flash decoding uses precomputed cos and sin for RoPE
				rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
					inference_params.max_sequence_length,
					self.rotary_pos_emb.get_cos_sin(inference_params.max_sequence_length),
				)
			else:
				rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
					inference_params, self.decoder, decoder_input, self.config, packed_seq_params
				)
				rotary_pos_emb = self.rotary_pos_emb(
					rotary_seq_len,
					packed_seq=packed_seq_params is not None
							   and packed_seq_params.qkv_format == 'thd',
				)
		if (
				(self.config.enable_cuda_graph or self.config.flash_decode)
				and rotary_pos_cos is not None
				and inference_params
		):
			sequence_len_offset = torch.tensor(
				[inference_params.sequence_len_offset] * inference_params.current_batch_size,
				dtype=torch.int32,
				device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
			)
		else:
			sequence_len_offset = None

		# Run decoder.
		hidden_states = self.decoder(
			hidden_states=decoder_input,
			attention_mask=attention_mask,
			inference_params=inference_params,
			rotary_pos_emb=rotary_pos_emb,
			rotary_pos_cos=rotary_pos_cos,
			rotary_pos_sin=rotary_pos_sin,
			packed_seq_params=packed_seq_params,
			sequence_len_offset=sequence_len_offset,
			**(extra_block_kwargs or {}),
		)

		if not self.post_process:
			return hidden_states

		# logits and loss
		output_weight = None
		if self.share_embeddings_and_output_weights:
			output_weight = self.shared_embedding_or_output_weight()
		logits, _ = self.output_layer(
			hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
		)

		if has_config_logger_enabled(self.config):
			payload = OrderedDict(
				{
					'input_ids': code_token_ids,
					'position_ids': code_token_pos_ids,
					'attention_mask': attention_mask,
					'decoder_input': decoder_input,
					'logits': logits,
				}
			)
			log_config_to_disk(self.config, payload, prefix='input_and_logits')

		if labels is None:
			# [s b h] => [b s h]
			return logits.transpose(0, 1).contiguous()

		loss = self.compute_language_model_loss(labels, logits)

		return loss
