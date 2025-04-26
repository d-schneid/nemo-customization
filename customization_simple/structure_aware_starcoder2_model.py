from nemo.collections import llm
from nemo.collections.llm import Starcoder2Model, Qwen2Model
from torch import nn
from typing import Annotated, Callable, Optional
from nemo.lightning import OptimizerModule
from nemo.collections.llm.utils import Config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class StructureAwareStarcoder2Model(Qwen2Model):

	def __init__(
			self,
			config: Annotated[Optional[llm.Qwen2Config], Config[llm.Qwen2Config]] = None,
			optim: Optional[OptimizerModule] = None,
			tokenizer: Optional["TokenizerSpec"] = None,
			model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
	):
		super().__init__(
			config=config,
			optim=optim,
			tokenizer=tokenizer,
			model_transform=model_transform,
		)
