from data_handler import DataHandler, PAD_TOK_ID_DFG
import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Optional, List, TYPE_CHECKING
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from nemo.utils.import_utils import safe_import
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

_, HAVE_TE = safe_import("transformer_engine")

if TYPE_CHECKING:
	from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

with open('../data/pretraining/metadata.json', 'r') as f_metadata:
	metadata = json.load(f_metadata)

PAD_TOK_ID_AST = metadata['num_ast_node_types']


class StructureAwareDataModule(pl.LightningDataModule):

	def __init__(
			self,
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
			merges_file: Optional[str] = None
	):
		super().__init__()
		self.seq_length = seq_length
		self.micro_batch_size = micro_batch_size
		self.global_batch_size = global_batch_size
		self.num_train_samples = num_train_samples
		self.num_val_samples = num_val_samples
		self.num_test_samples = num_test_samples
		self.num_workers = num_workers
		self.pin_memory = pin_memory
		self.persistent_workers = persistent_workers
		self.create_attention_mask = create_attention_mask or not HAVE_TE

		if tokenizer is None:
			from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

			self.tokenizer = get_nmt_tokenizer(
				"megatron", "GPT2BPETokenizer", vocab_file=vocab_file, merges_file=merges_file
			)
		else:
			self.tokenizer = tokenizer

		self.data_sampler = MegatronDataSampler(
			seq_len=self.seq_length,
			micro_batch_size=self.micro_batch_size,
			global_batch_size=self.global_batch_size,
			rampup_batch_size=rampup_batch_size,
		)

	def setup(self, stage: str = "") -> None:
		self._train_ds = StructureAwareDataset('../data/pretraining')
		self._validation_ds = StructureAwareDataset('../data/pretraining')
		self._test_ds = StructureAwareDataset('../data/pretraining')

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


class StructureAwareDataset(Dataset):

	def __init__(self, data_dir) -> None:
		super().__init__()
		self.data_handler = DataHandler(save_dir=data_dir)
		self.padding_value = self.data_handler.tokenizer.eos_token_id
		self.data = self.data_handler.get_concat_stored_data()

		self.data['tokens'] = (self.data['code_tokens'].apply(lambda x: list(map(int, x.split(',')))).
									apply(lambda x: torch.tensor(x)))

		self.data['position_ids'] = (self.data['code_tokens_pos_ids'].
											apply(lambda x: list(map(int, x.split(',')))).
											apply(lambda x: torch.tensor(x)))

	def __getitem__(self, idx):
		tokens = self.data.iloc[idx]['tokens']

		batch = {
			'tokens': tokens[:-1],
			'position_ids': self.data.iloc[idx]['position_ids'],
			'labels': tokens[1:],
			'loss_mask': torch.ones(len(tokens[1:]))
		}

		return batch

	def __len__(self) -> int:
		return len(self.data)

	def _collate_fn(self, batch):
		"""
		A default implementation of a collation function.
		Users should override this method to define custom data loaders.
		"""
		return data.dataloader.default_collate(batch)

	def collate_fn(self, batch):
		"""Method that user pass as functor to DataLoader.

		The method optionally performs neural type checking and add types to the outputs.

		Please note, subclasses of Dataset should not implement `input_types`.

		# Usage:
		dataloader = torch.utils.data.DataLoader(
				....,
				collate_fn=dataset.collate_fn,
				....
		)

		Returns
		-------
			Collated batch, with or without types.
		"""
		# Initialize a dictionary to store the batch data
		batch_dict = {}
		for key in batch[0].keys():
			batch_dict[key] = [sample[key] for sample in batch]
			if key not in ['tokens', 'position_ids', 'labels', 'loss_mask']:
				batch_dict[key] = pad_2d_tensors(batch_dict[key], padding_value=self.padding_value)

			padding_value = self.padding_value if key != 'dfg_node_mask' else PAD_TOK_ID_DFG
			batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=padding_value)

		return batch_dict


def pad_labels_loss_mask(batch_dict):
	labels = batch_dict['labels']
	loss_mask = batch_dict['loss_mask']
	dfg_node_mask = batch_dict['dfg_node_mask']
	lr_paths_len = batch_dict['lr_paths_len']

	pad_len = dfg_node_mask[0].size(0) + lr_paths_len[0].size(0)

	padded_labels = []
	padded_loss_mask = []
	for label, mask in zip(labels, loss_mask):
		padded_label = F.pad(label, (0, pad_len), value=0)
		padded_mask = F.pad(mask, (0, pad_len), value=0)
		padded_labels.append(padded_label)
		padded_loss_mask.append(padded_mask)

	batch_dict['labels'] = torch.stack(padded_labels)
	batch_dict['loss_mask'] = torch.stack(padded_loss_mask)

	return batch_dict


def pad_inner_lists(list_of_lists, padding_value, padding_side='right'):
	tensors = [torch.tensor(x) for x in list_of_lists]

	return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side=padding_side) if tensors else [torch.tensor(-1)]


def pad_2d_tensors(tensor_list, padding_value, padding_side='right'):
	max_rows = max(tensor.size(0) for tensor in tensor_list)
	max_cols = max(tensor.size(1) for tensor in tensor_list)

	padded_tensors = []
	for tensor in tensor_list:
		rows_to_pad = max_rows - tensor.size(0)
		cols_to_pad = max_cols - tensor.size(1)

		if padding_side == 'right':
			padded_tensor = torch.nn.functional.pad(tensor, (0, cols_to_pad, 0, rows_to_pad), mode='constant', value=padding_value)
		else:
			padded_tensor = torch.nn.functional.pad(tensor, (cols_to_pad, 0, 0, rows_to_pad), mode='constant', value=padding_value)

		padded_tensors.append(padded_tensor)

	return padded_tensors
