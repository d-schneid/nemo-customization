import warnings
from typing import Optional

from structure_aware_self_attention import StructureAwareSelfAttention
from structure_aware_transformer_layer import StructureAwareTransformerLayer

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.utils import is_te_min_version
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from nemo.collections.llm.gpt.model import transformer_engine_full_layer_spec, local_layer_spec, GPTConfig

try:
	from megatron.core.extensions.transformer_engine import (
		TEColumnParallelLinear,
		TEDotProductAttention,
		TELayerNormColumnParallelLinear,
		TENorm,
		TERowParallelLinear,
	)

	HAVE_TE = True
except ImportError:
	HAVE_TE = False

try:
	import apex  # pylint: disable=unused-import

	from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

	HAVE_APEX = True
	LNImpl = FusedLayerNorm
except ImportError:
	from megatron.core.transformer.torch_norm import WrappedTorchNorm

	warnings.warn('Apex is not installed. Falling back to Torch Norm')
	LNImpl = WrappedTorchNorm


def get_gpt_layer_with_transformer_engine_spec(
	num_experts: Optional[int] = None,
	moe_grouped_gemm: Optional[bool] = False,
	qk_layernorm: Optional[bool] = False,
	multi_latent_attention: Optional[bool] = False,
	fp8: Optional[str] = None,  # pylint: disable=unused-arguments
	moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:

	if fp8 is not None:
		warnings.warn(
			'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
			' and will be removed soon. Please update your code accordingly.'
		)

	mlp = get_mlp_module_spec(
		use_te=True,
		num_experts=num_experts,
		moe_grouped_gemm=moe_grouped_gemm,
		moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
	)

	# TENorm significantly harms convergence when used
	# for QKLayerNorm if TE Version < 1.9;
	# we instead use the Apex implementation.
	qk_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm

	return ModuleSpec(
		module=StructureAwareTransformerLayer,
		submodules=TransformerLayerSubmodules(
			self_attention=ModuleSpec(
				module=StructureAwareSelfAttention,
				params={"attn_mask_type": AttnMaskType.arbitrary},
				submodules=SelfAttentionSubmodules(
					linear_qkv=TELayerNormColumnParallelLinear,
					core_attention=TEDotProductAttention,
					linear_proj=TERowParallelLinear,
					q_layernorm=qk_norm if qk_layernorm else IdentityOp,
					k_layernorm=qk_norm if qk_layernorm else IdentityOp,
				),
			),
			self_attn_bda=get_bias_dropout_add,
			pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
			mlp=mlp,
			mlp_bda=get_bias_dropout_add,
		),
	)


def transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
	return get_gpt_layer_with_transformer_engine_spec(
		num_experts=config.num_moe_experts,
		moe_grouped_gemm=config.moe_grouped_gemm,
		qk_layernorm=config.qk_layernorm,
		fp8=bool(config.num_moe_experts and (config.fp8 is not None)),
	)


def structure_aware_layer_spec(config: "GPTConfig") -> ModuleSpec:
	if HAVE_TE:
		if config.use_transformer_engine_full_layer_spec:
			return transformer_engine_full_layer_spec(config)
		else:
			return transformer_engine_layer_spec(config)
	else:
		return local_layer_spec(config)
