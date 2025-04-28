from typing import Optional, Literal

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
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
		super().__init__(config, transformer_layer_spec, vocab_size, max_sequence_length,
			pre_process, post_process, fp16_lm_cross_entropy, parallel_output,
			share_embeddings_and_output_weights, position_embedding_type,
			rotary_percent, rotary_base, rope_scaling, rope_scaling_factor,
			scatter_embedding_sequence_parallel, seq_len_interpolation_factor)
