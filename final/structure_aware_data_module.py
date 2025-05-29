from typing import Optional, List, TYPE_CHECKING

from structure_aware_dataset import StructureAwareDataset

from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from nemo.collections.llm.gpt.data.mock import MockDataModule

if TYPE_CHECKING:
	from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class StructureAwareDataModule(MockDataModule):

	def __init__(
			self,
			train_dataset: StructureAwareDataset,
			validation_dataset: StructureAwareDataset,
			test_dataset: StructureAwareDataset,
			seq_length: int = 2048,
			tokenizer: Optional["TokenizerSpec"] = None,
			micro_batch_size: int = 2,
			global_batch_size: int = 4,
			rampup_batch_size: Optional[List[int]] = None,
			num_train_samples: int = 10_000,
			num_val_samples: int = 10_000,
			num_test_samples: int = 10_000,
			num_workers: int = 1,
			pin_memory: bool = True,
			persistent_workers: bool = False,
			create_attention_mask: bool = False,
			vocab_file: Optional[str] = None,
			merges_file: Optional[str] = None,
	):
		super().__init__(
			seq_length=seq_length,
			tokenizer=tokenizer,
			micro_batch_size=micro_batch_size,
			global_batch_size=global_batch_size,
			rampup_batch_size=rampup_batch_size,
			num_train_samples=num_train_samples,
			num_val_samples=num_val_samples,
			num_test_samples=num_test_samples,
			num_workers=num_workers,
			pin_memory=pin_memory,
			persistent_workers=persistent_workers,
			create_attention_mask=create_attention_mask,
			vocab_file=vocab_file,
			merges_file=merges_file,
		)
		self.train_dataset = train_dataset
		self.validation_dataset = validation_dataset
		self.test_dataset = test_dataset

	def setup(self, stage: str = "") -> None:
		self._train_ds = self.train_dataset
		self._validation_ds = self.validation_dataset
		self._test_ds = self.test_dataset

	def train_dataloader(self) -> TRAIN_DATALOADERS:
		if not hasattr(self, "_train_ds"):
			self.setup()
		return self._create_dataloader(self._train_ds)

	def val_dataloader(self) -> EVAL_DATALOADERS:
		if not hasattr(self, "_validation_ds"):
			self.setup()
		return self._create_dataloader(self._validation_ds)

	def test_dataloader(self) -> EVAL_DATALOADERS:
		if not hasattr(self, "_test_ds"):
			self.setup()
		return self._create_dataloader(self._test_ds)

	def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
		return DataLoader(
			dataset,
			num_workers=self.num_workers,
			pin_memory=self.pin_memory,
			persistent_workers=self.persistent_workers,
			collate_fn=dataset.collate_fn,
			**kwargs,
		)
