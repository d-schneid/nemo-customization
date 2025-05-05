from typing import Annotated, Callable, Optional, TYPE_CHECKING

import torch
from torch import nn

from nemo.collections.llm import Starcoder2Model, Starcoder2Config
from nemo.lightning import OptimizerModule
from nemo.collections.llm.utils import Config

if TYPE_CHECKING:
	from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


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
			code_token_pos_ids: torch.Tensor,
			ll_sims: torch.Tensor,
			lr_paths_types: torch.Tensor,
			lr_paths_len: torch.Tensor,
			dfg_node_mask: torch.Tensor,
			attention_mask: Optional[torch.Tensor] = None,
			labels: Optional[torch.Tensor] = None,
			decoder_input: Optional[torch.Tensor] = None,
			inference_params=None,
			packed_seq_params=None,
	) -> torch.Tensor:
		extra_kwargs = {'packed_seq_params': packed_seq_params} if packed_seq_params is not None else {}
		output_tensor = self.module(
			code_token_ids=code_token_ids,
			code_token_pos_ids=code_token_pos_ids,
			ll_sims=ll_sims,
			lr_paths_types=lr_paths_types,
			lr_paths_len=lr_paths_len,
			dfg_node_mask=dfg_node_mask,
			attention_mask=attention_mask,
			decoder_input=decoder_input,
			labels=labels,
			inference_params=inference_params,
			**extra_kwargs,
		)

		return output_tensor
