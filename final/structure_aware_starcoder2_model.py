from typing import Annotated, Callable, Optional, TYPE_CHECKING
from pathlib import Path

import torch
from torch import nn

from structure_aware_starcoder2_config import StructureAwareStarcoder2Config

from nemo.collections.llm import Starcoder2Model, Starcoder2Config
from nemo.lightning import OptimizerModule, teardown, io
from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.utils import dtype_from_hf

if TYPE_CHECKING:
	from transformers import Starcoder2ForCausalLM

	from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
	from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


class StructureAwareStarcoder2Model(Starcoder2Model):

	def __init__(
			self,
			config: Annotated[Optional[Starcoder2Config], Config[Starcoder2Config]] = None,
			optim: Optional[OptimizerModule] = None,
			tokenizer: Optional["TokenizerSpec"] = None,
			model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
	):
		super().__init__(config=config, optim=optim, tokenizer=tokenizer, model_transform=model_transform)

	def forward(
			self,
			code_token_ids: torch.Tensor,
			code_token_rel_pos_ids: torch.Tensor,
			ll_sims: torch.Tensor,
			lr_paths_types: torch.Tensor,
			lr_paths_len: torch.Tensor,
			dfg_node_mask: torch.Tensor,
			attention_bias: torch.Tensor,
			text_token_ids: Optional[torch.Tensor] = None,
			text_token_rel_pos_ids: Optional[torch.Tensor] = None,
			attention_mask: Optional[torch.Tensor] = None,
			labels: Optional[torch.Tensor] = None,
			decoder_input: Optional[torch.Tensor] = None,
			inference_params=None,
			packed_seq_params=None,
	) -> torch.Tensor:
		extra_kwargs = {'packed_seq_params': packed_seq_params} if packed_seq_params is not None else {}
		output_tensor = self.module(
			code_token_ids=code_token_ids,
			code_token_rel_pos_ids=code_token_rel_pos_ids,
			ll_sims=ll_sims,
			lr_paths_types=lr_paths_types,
			lr_paths_len=lr_paths_len,
			dfg_node_mask=dfg_node_mask,
			attention_bias=attention_bias,
			text_token_ids=text_token_ids,
			text_token_rel_pos_ids=text_token_rel_pos_ids,
			attention_mask=attention_mask,
			decoder_input=decoder_input,
			labels=labels,
			inference_params=inference_params,
			**extra_kwargs,
		)

		return output_tensor


@io.model_importer(StructureAwareStarcoder2Model, "hf")
class HFStructureAwareStarcoder2Importer(io.ModelConnector["Starcoder2ForCausalLM", StructureAwareStarcoder2Model]):

	def init(self) -> StructureAwareStarcoder2Model:
		return StructureAwareStarcoder2Model(self.config, tokenizer=self.tokenizer)

	def apply(self, output_path: Path) -> Path:
		from transformers import Starcoder2ForCausalLM

		source = Starcoder2ForCausalLM.from_pretrained(str(self), torch_dtype='auto')
		target = self.init()
		trainer = self.nemo_setup(target)
		self.convert_state(source, target)
		self.nemo_save(output_path, trainer)

		print(f"Converted StructureAwareStarcoder2 model to Nemo, model saved to {output_path}")

		teardown(trainer, target)
		del trainer, target

		return output_path

	def convert_state(self, source, target):
		mapping = {
			"model.embed_tokens.weight": "embedding.word_embeddings.weight",
			"model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
			"model.layers.*.self_attn.o_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
			"model.layers.*.mlp.c_fc.weight": "decoder.layers.*.mlp.linear_fc1.weight",
			"model.layers.*.mlp.c_fc.bias": "decoder.layers.*.mlp.linear_fc1.bias",
			"model.layers.*.mlp.c_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
			"model.layers.*.mlp.c_proj.bias": "decoder.layers.*.mlp.linear_fc2.bias",
			"model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
			"model.layers.*.input_layernorm.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
			"model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
			"model.layers.*.post_attention_layernorm.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
			"model.norm.weight": "decoder.final_layernorm.weight",
			"model.norm.bias": "decoder.final_layernorm.bias",
			"lm_head.weight": "output_layer.weight",
		}

		return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv_bias, _import_qkv_weight])

	@property
	def tokenizer(self) -> "AutoTokenizer":
		from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

		return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

	@property
	def config(self) -> StructureAwareStarcoder2Config:
		from transformers import Starcoder2Config as HFStarcoder2Config

		source = HFStarcoder2Config.from_pretrained(str(self))

		def make_vocab_size_divisible_by(vocab_size):
			base = 128
			while vocab_size % base != 0:
				base //= 2
			return base

		output = StructureAwareStarcoder2Config(
			num_layers=source.num_hidden_layers,
			hidden_size=source.hidden_size,
			ffn_hidden_size=source.intermediate_size,
			num_attention_heads=source.num_attention_heads,
			init_method_std=source.initializer_range,
			seq_length=source.max_position_embeddings,
			layernorm_epsilon=source.norm_epsilon,
			num_query_groups=source.num_key_value_heads,
			rotary_base=source.rope_theta,
			make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
			share_embeddings_and_output_weights=False,
			fp16=(dtype_from_hf(source) == torch.float16),
			bf16=(dtype_from_hf(source) == torch.bfloat16),
			params_dtype=dtype_from_hf(source),
		)

		return output


@io.state_transform(
	source_key=(
		"model.layers.*.self_attn.q_proj.weight",
		"model.layers.*.self_attn.k_proj.weight",
		"model.layers.*.self_attn.v_proj.weight",
	),
	target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_weight(ctx: io.TransformCTX, q, k, v):
	megatron_config = ctx.target.config

	head_num = megatron_config.num_attention_heads
	num_query_groups = megatron_config.num_query_groups
	heads_per_group = head_num // num_query_groups
	hidden_size = megatron_config.hidden_size
	head_size = megatron_config.kv_channels

	old_tensor_shape = q.size()
	new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
	new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

	q = q.view(*new_q_tensor_shape)
	k = k.view(*new_kv_tensor_shape)
	v = v.view(*new_kv_tensor_shape)

	qkv_weights_l = []
	for i in range(num_query_groups):
		qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
		qkv_weights_l.append(k[i : i + 1, :, :])
		qkv_weights_l.append(v[i : i + 1, :, :])

	qkv_weights = torch.cat(qkv_weights_l)
	assert qkv_weights.ndim == 3, qkv_weights.shape
	assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
	assert qkv_weights.shape[1] == head_size, qkv_weights.shape
	assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

	qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

	return qkv_weights


@io.state_transform(
	source_key=(
		"model.layers.*.self_attn.q_proj.bias",
		"model.layers.*.self_attn.k_proj.bias",
		"model.layers.*.self_attn.v_proj.bias",
	),
	target_key="decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, qb, kb, vb):
	megatron_config = ctx.target.config

	head_num = megatron_config.num_attention_heads
	num_query_groups = megatron_config.num_query_groups
	heads_per_group = head_num // num_query_groups
	head_size = megatron_config.kv_channels

	new_q_bias_tensor_shape = (head_num, head_size)
	new_kv_bias_tensor_shape = (num_query_groups, head_size)

	qb = qb.view(*new_q_bias_tensor_shape)
	kb = kb.view(*new_kv_bias_tensor_shape)
	vb = vb.view(*new_kv_bias_tensor_shape)

	qkv_bias_l = []
	for i in range(num_query_groups):
		qkv_bias_l.append(qb[i * heads_per_group : (i + 1) * heads_per_group, :])
		qkv_bias_l.append(kb[i : i + 1, :])
		qkv_bias_l.append(vb[i : i + 1, :])

	qkv_bias = torch.cat(qkv_bias_l)
	qkv_bias = qkv_bias.reshape([head_size * (head_num + 2 * num_query_groups)])

	return qkv_bias
