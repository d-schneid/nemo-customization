import copy

import torch

from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding


class StructureAwareSelfAttention(SelfAttention):

	def __init__(
			self,
			config: TransformerConfig,
			submodules: SelfAttentionSubmodules,
			layer_number: int,
			attn_mask_type=AttnMaskType.padding,
			cp_comm_type: str = None,
	):
		super().__init__(
			config=config,
			submodules=submodules,
			layer_number=layer_number,
			attn_mask_type=attn_mask_type,
			cp_comm_type=cp_comm_type
		)

		config_copy = copy.copy(config)
		config_copy.hidden_size = 1 # scalar embedding

		vocab_size_code_text_rel_pos = config.max_code_token_rel_pos + 1 # padding
		self.code_text_token_rel_pos_embedding = LanguageModelEmbedding(
			config=config_copy,
			vocab_size=vocab_size_code_text_rel_pos if vocab_size_code_text_rel_pos % 2 == 0 else vocab_size_code_text_rel_pos + 1,  # even
			max_sequence_length=-1,
			position_embedding_type='none',
			scatter_to_sequence_parallel=True,
		)

		self.ll_sims_weight_bias = LanguageModelEmbedding(
			config=config_copy,
			vocab_size=2,  # parameter for weight and bias
			max_sequence_length=-1,
			position_embedding_type='none',
			scatter_to_sequence_parallel=True,
		)

	def forward(
			self,
			hidden_states,
			attention_mask,
			code_token_rel_pos_ids,
			ll_sims,
			attention_bias,
			text_token_rel_pos_ids=None,
			key_value_states=None,
			inference_params=None,
			rotary_pos_emb=None,
			rotary_pos_cos=None,
			rotary_pos_sin=None,
			packed_seq_params=None,
			sequence_len_offset=None,
	):
		code_token_rel_pos_embedding= self.code_text_token_rel_pos_embedding(input_ids=code_token_rel_pos_ids, position_ids=None)
		code_token_rel_pos_embedding = code_token_rel_pos_embedding.permute(1, 3, 0, 2)

		ll_sims_weight_param = self.ll_sims_weight_bias(input_ids=torch.tensor([0], device=ll_sims.device), position_ids=None)
		ll_sims_bias_param = self.ll_sims_weight_bias(input_ids=torch.tensor([1], device=ll_sims.device), position_ids=None)
		weighted_ll_sims = ll_sims_weight_param * ll_sims.unsqueeze(1) + ll_sims_bias_param

		batch, head, height_code_token, width_code_token = code_token_rel_pos_embedding.shape

		if text_token_rel_pos_ids is not None:
			text_token_rel_pos_embedding = self.code_text_token_rel_pos_embedding(input_ids=text_token_rel_pos_ids, position_ids=None)
			text_token_rel_pos_embedding = text_token_rel_pos_embedding.permute(1, 3, 0, 2)
			batch, head, height_text_token, width_text_token = text_token_rel_pos_embedding.shape
			target_text_tokens = attention_bias[:, :, -height_text_token:, -width_text_token:]
			mask_text_tokens = target_text_tokens > -1
			updated_text_tokens = torch.where(
				mask_text_tokens,
				target_text_tokens + text_token_rel_pos_embedding,
				target_text_tokens
			)
			attention_bias[:, :, -height_text_token:, -width_text_token:] = updated_text_tokens

			target_code_tokens = attention_bias[:, :, -(height_code_token + height_text_token):-height_text_token, -(width_code_token + width_text_token):-width_text_token]
		else:
			target_code_tokens = attention_bias[:, :, -height_code_token:, -width_code_token:]

		mask_code_tokens = target_code_tokens > -1 # only update tokens that shall attend to each other
		updated_code_tokens = torch.where(
			mask_code_tokens,
			target_code_tokens + code_token_rel_pos_embedding,
			target_code_tokens
		)

		if text_token_rel_pos_ids is not None:
			attention_bias[:, :, -(height_code_token + height_text_token):-height_text_token, -(width_code_token + width_text_token):-width_text_token] = updated_code_tokens
		else:
			attention_bias[:, :, -height_code_token:, -width_code_token:] = updated_code_tokens

		batch, head, height_ll_sims, width_ll_sims = weighted_ll_sims.shape
		target_ll_sims = attention_bias[:, :, :height_ll_sims, :width_ll_sims]
		mask_ll_sims = target_ll_sims > -1 # only update tokens that shall attend to each other
		updated_ll_sims = torch.where(
			mask_ll_sims,
			target_ll_sims + weighted_ll_sims,
			target_ll_sims
		)
		attention_bias[:, :, :height_ll_sims, :width_ll_sims] = updated_ll_sims

		output, bias =  super().forward(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			key_value_states=key_value_states,
			inference_params=inference_params,
			rotary_pos_emb=rotary_pos_emb,
			rotary_pos_cos=rotary_pos_cos,
			rotary_pos_sin=rotary_pos_sin,
			attention_bias=attention_bias,
			packed_seq_params=packed_seq_params,
			sequence_len_offset=sequence_len_offset,
		)

		return output, bias
