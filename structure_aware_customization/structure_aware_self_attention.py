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

		self.code_token_rel_pos_embedding = LanguageModelEmbedding(
			config=config_copy,
			vocab_size=config.max_code_token_rel_pos + 1,  # padding
			max_sequence_length=-1,
			position_embedding_type='none',
			scatter_to_sequence_parallel=True,
		)

		self.ll_sims_weight_param = LanguageModelEmbedding(
			config=config_copy,
			vocab_size=1,  # one parameter
			max_sequence_length=-1,
			position_embedding_type='none',
			scatter_to_sequence_parallel=True,
		)

		self.ll_sims_bias_param = LanguageModelEmbedding(
			config=config_copy,
			vocab_size=1,  # one parameter
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
			key_value_states=None,
			inference_params=None,
			rotary_pos_emb=None,
			rotary_pos_cos=None,
			rotary_pos_sin=None,
			attention_bias=None,
			packed_seq_params=None,
			sequence_len_offset=None,
	):
		code_token_rel_pos_embedding= self.code_token_rel_pos_embedding(input_ids=code_token_rel_pos_ids, position_ids=None)
		code_token_rel_pos_embedding = code_token_rel_pos_embedding.squeeze(-1)
		code_token_rel_pos_embedding = code_token_rel_pos_embedding.permute(1, 0, 2)

		ll_sims_weight_param = self.ll_sims_weight_param(input_ids=torch.tensor([0], device=ll_sims.device), position_ids=None)
		ll_sims_bias_param = self.ll_sims_bias_param(input_ids=torch.tensor([0], device=ll_sims.device), position_ids=None)
		weighted_ll_sims = ll_sims_weight_param * ll_sims + ll_sims_bias_param

		batch, _, height, width = attention_mask.shape
		_, height_code_token, width_code_token = code_token_rel_pos_embedding.shape
		_, height_ll_sims, width_ll_sims = weighted_ll_sims.shape
		attention_bias = torch.zeros(batch, height, width, dtype=ll_sims.dtype, device=ll_sims.device)
		attention_bias[:, :height_code_token, :width_code_token] = code_token_rel_pos_embedding
		attention_bias[:, height_code_token:height_code_token + height_ll_sims, width_code_token:width_code_token + width_ll_sims] = weighted_ll_sims
		attention_bias = attention_bias.unsqueeze(1)

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
