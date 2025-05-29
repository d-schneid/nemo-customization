import os
import ast
import json
from abc import ABC, abstractmethod

from data_handler import DataHandler, PAD_TOK_ID_DFG

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class StructureAwareDataset(ABC, Dataset):

	def __init__(self, save_dir='../../data/pretraining', task='code_completion', split='train') -> None:
		super().__init__()
		self.data_handler = DataHandler(save_dir=os.path.join(save_dir, task))
		self.padding_value = self.data_handler.tokenizer.eos_token_id
		self.data = self.data_handler.get_concat_stored_data(split=split)
		with open(os.path.join(save_dir, task, 'metadata.json'), 'r') as f_metadata:
			metadata = json.load(f_metadata)
		self.pad_tok_id_ast = metadata['num_ast_node_types']

		self.data['code_tokens'] = (self.data['code_tokens'].apply(lambda x: list(map(int, x.split(',')))).
									apply(lambda x: torch.tensor(x)))

		self.data['code_tokens_rel_pos_ids'] = (self.data['code_tokens_rel_pos_ids'].apply(ast.literal_eval).
												apply(lambda x: torch.tensor(x)))

		self.data['ll_sims'] = (self.data['ll_sims'].
								apply(lambda x: [list(map(float, sublist.split(','))) for sublist in x.split(';')]).
								apply(pad_inner_lists, padding_value=self.padding_value, padding_side='left'))

		self.data['lr_paths_types'] = (self.data['lr_paths_types'].apply(lambda x: ast.literal_eval(x)).
									   apply(pad_inner_lists, padding_value=self.pad_tok_id_ast))

		self.data['lr_paths_len'] = (self.data['lr_paths_len'].apply(lambda x: list(map(int, x.split(',')))).
									 apply(lambda x: torch.tensor(x)))

		self.data['dfg_node_mask'] = (self.data['dfg_node_mask'].apply(lambda x: list(map(int, x.split(',')))).
									  apply(lambda x: torch.tensor(x)))

		self.data['attn_code_tokens'] = self.data['attn_code_tokens'].apply(ast.literal_eval).apply(lambda x: torch.tensor(x))

		self.data['attn_ast_leaves'] = self.data['attn_ast_leaves'].apply(ast.literal_eval).apply(lambda x: torch.tensor(x))

		self.data['attn_dfg_edges'] = self.data['attn_dfg_edges'].apply(ast.literal_eval).apply(lambda x: torch.tensor(x))

		self.data['attn_code_ast'] = self.data['attn_code_ast'].apply(ast.literal_eval).apply(lambda x: torch.tensor(x))

		self.data['attn_code_dfg'] = self.data['attn_code_dfg'].apply(ast.literal_eval).apply(lambda x: torch.tensor(x))

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx):
		batch = {
			'code_token_ids': self.data.iloc[idx]['code_tokens'],
			'code_token_rel_pos_ids': self.data.iloc[idx]['code_tokens_rel_pos_ids'],
			'll_sims': self.data.iloc[idx]['ll_sims'],
			'lr_paths_types': self.data.iloc[idx]['lr_paths_types'],
			'lr_paths_len': self.data.iloc[idx]['lr_paths_len'],
			'dfg_node_mask': self.data.iloc[idx]['dfg_node_mask'],
			'attn_code_tokens': self.data.iloc[idx]['attn_code_tokens'],
			'attn_ast_leaves': self.data.iloc[idx]['attn_ast_leaves'],
			'attn_dfg_edges': self.data.iloc[idx]['attn_dfg_edges'],
			'attn_code_ast': self.data.iloc[idx]['attn_code_ast'],
			'attn_code_dfg': self.data.iloc[idx]['attn_code_dfg'],
		}

		return batch

	@abstractmethod
	def get_key_not_in(self):
		pass

	@abstractmethod
	def get_attn_keys(self):
		pass

	@abstractmethod
	def get_labels_loss_pad_len(self, batch_dict):
		pass

	@abstractmethod
	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		pass

	def collate_fn(self, batch):
		# Initialize a dictionary to store the batch data
		batch_dict = {}
		for key in batch[0].keys():
			batch_dict[key] = [sample[key] for sample in batch]
			if key not in self.get_key_not_in():
				if key == 'lr_paths_types':
					batch_dict[key] = pad_2d_tensors(batch_dict[key], padding_value=self.pad_tok_id_ast)
				elif key in self.get_attn_keys():
					batch_dict[key] = pad_2d_tensors(batch_dict[key], padding_value=-1e9)
				else:
					batch_dict[key] = pad_2d_tensors(batch_dict[key], padding_value=self.padding_value)

			padding_value = self.padding_value
			if key == 'dfg_node_mask':
				padding_value = PAD_TOK_ID_DFG
			if key in self.get_attn_keys():
				padding_value = -1e9

			if key in ['labels', 'loss_mask']:
				batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=padding_value, padding_side='left')
			else:
				batch_dict[key] = pad_sequence(batch_dict[key], batch_first=True, padding_value=padding_value)

		pad_len = self.get_labels_loss_pad_len(batch_dict)
		batch_dict = pad_labels_loss_mask(batch_dict, pad_len)

		# individual padded attention masks
		attn_code_tokens = batch_dict['attn_code_tokens']
		attn_ast_leaves = batch_dict['attn_ast_leaves']
		attn_dfg_edges = batch_dict['attn_dfg_edges']
		attn_code_ast = batch_dict['attn_code_ast']
		attn_code_dfg = batch_dict['attn_code_dfg']

		# Compute transpose
		attn_code_ast_T = attn_code_ast.transpose(1, 2)
		attn_code_dfg_T = attn_code_dfg.transpose(1, 2)

		# Compute null matrix for attention between AST leaves and DFG edges
		attn_ast_dfg = torch.full((attn_ast_leaves.size(0), attn_ast_leaves.size(1), attn_dfg_edges.size(2)), fill_value=-1e9)
		attn_ast_dfg_T = attn_ast_dfg.transpose(1, 2)

		# Build block matrices column-wise
		first_col_matrix = torch.cat((attn_ast_leaves, attn_ast_dfg_T, attn_code_ast), dim=1)
		second_col_matrix = torch.cat((attn_ast_dfg, attn_dfg_edges, attn_code_dfg), dim=1)
		third_col_matrix = torch.cat((attn_code_ast_T, attn_code_dfg_T, attn_code_tokens), dim=1)

		attn_bias = self.build_attn_bias(batch_dict, first_col_matrix, second_col_matrix, third_col_matrix)

		batch_dict['attention_bias'] = attn_bias.unsqueeze(1).bfloat16() # broadcast across all attention heads

		keys_to_remove = self.get_attn_keys()
		for key in keys_to_remove:
			del batch_dict[key]

		return batch_dict


def pad_labels_loss_mask(batch_dict, pad_len):
	labels = batch_dict['labels']
	loss_mask = batch_dict['loss_mask']

	padded_labels = []
	padded_loss_mask = []

	for label, mask in zip(labels, loss_mask):
		padded_label = F.pad(label, (pad_len, 0), value=0)
		padded_mask = F.pad(mask, (pad_len, 0), value=0)
		padded_labels.append(padded_label)
		padded_loss_mask.append(padded_mask)

	batch_dict['labels'] = torch.stack(padded_labels)
	batch_dict['loss_mask'] = torch.stack(padded_loss_mask)

	return batch_dict


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


def pad_inner_lists(list_of_lists, padding_value, padding_side='right'):
	tensors = [torch.tensor(x) for x in list_of_lists]

	return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side=padding_side) if tensors else [torch.tensor(-1)]
