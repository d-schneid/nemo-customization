from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor


class StructureAwareTransformerLayer(TransformerLayer):

	def forward(
			self,
			hidden_states,
			code_token_rel_pos_ids,
			ll_sims,
			attention_mask=None,
			context=None,
			context_mask=None,
			rotary_pos_emb=None,
			rotary_pos_cos=None,
			rotary_pos_sin=None,
			attention_bias=None,
			inference_params=None,
			packed_seq_params=None,
			sequence_len_offset=None,
	):
		# Residual connection.
		residual = hidden_states

		# Optional Input Layer norm
		input_layernorm_output = self.input_layernorm(hidden_states)

		# Self attention.
		attention_output_with_bias = self.self_attention(
			input_layernorm_output,
			attention_mask=attention_mask,
			inference_params=inference_params,
			rotary_pos_emb=rotary_pos_emb,
			rotary_pos_cos=rotary_pos_cos,
			rotary_pos_sin=rotary_pos_sin,
			attention_bias=attention_bias,
			packed_seq_params=packed_seq_params,
			sequence_len_offset=sequence_len_offset,
			code_token_rel_pos_ids=code_token_rel_pos_ids,
			ll_sims=ll_sims,
		)

		# TODO: could we move `bias_dropout_add_exec_handler` itself
		# inside the module provided in the `bias_dropout_add_spec` module?
		with self.bias_dropout_add_exec_handler():
			hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
				attention_output_with_bias, residual, self.hidden_dropout
			)

		# Residual connection.
		residual = hidden_states

		# Optional Layer norm after self-attention
		pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

		# Cross attention.
		attention_output_with_bias = self.cross_attention(
			pre_cross_attn_layernorm_output,
			attention_mask=context_mask,
			key_value_states=context,
			inference_params=inference_params,
		)

		if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
			context = attention_output_with_bias["context"]

		# TODO: could we move `bias_dropout_add_exec_handler` itself
		# inside the module provided in the `bias_dropout_add_spec` module?
		with self.bias_dropout_add_exec_handler():
			hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
				attention_output_with_bias, residual, self.hidden_dropout
			)

		# Residual connection.
		residual = hidden_states

		# Optional Layer norm post the cross-attention.
		pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

		# MLP.
		mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

		# TODO: could we move `bias_dropout_add_exec_handler` itself
		# inside the module provided in the `bias_dropout_add_spec` module?
		with self.bias_dropout_add_exec_handler():
			hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
				mlp_output_with_bias, residual, self.hidden_dropout
			)

		# Jit compiled function creates 'view' tensor. This tensor
		# potentially gets saved in the MPU checkpoint function context,
		# which rejects view tensors. While making a viewless tensor here
		# won't result in memory savings (like the data loader, or
		# p2p_communication), it serves to document the origin of this
		# 'view' tensor.
		output = make_viewless_tensor(
			inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
		)

		# CUDA graph requires returned values to be Tensors
		if self.config.external_cuda_graph and self.training:
			return output
		return output, context
