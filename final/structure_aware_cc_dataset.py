from structure_aware_dataset import StructureAwareDataset

import torch


class StructureAwareCCDataset(StructureAwareDataset):

	def __init__(self, save_dir='../../data/pretraining', split='train') -> None:
		super().__init__(save_dir=save_dir, task='code_completion', split=split)

	def __getitem__(self, idx):
		batch = super().__getitem__(idx)

		code_tokens = self.data.iloc[idx]['code_tokens']
		labels = torch.cat([torch.tensor([self.padding_value]), code_tokens[1:]])
		loss_mask = torch.cat([torch.tensor([0]), torch.ones(len(code_tokens[:-1]))])

		batch['labels'] = labels
		batch['loss_mask'] = loss_mask

		return batch

	def get_key_not_in(self):
		return ['code_token_ids', 'dfg_node_mask', 'lr_paths_len', 'labels', 'loss_mask']

	def get_attn_keys(self):
		return ['attn_code_tokens', 'attn_ast_leaves', 'attn_dfg_edges', 'attn_code_ast', 'attn_code_dfg']

	def get_labels_loss_pad_len(self, batch_dict):
		return batch_dict['dfg_node_mask'][0].size(0) + batch_dict['lr_paths_len'][0].size(0)

	def build_attn_bias(self, batch_dict, first_col_matrix, second_col_matrix, third_col_matrix):
		# Build block matrices column-wise
		attn_bias = torch.cat((first_col_matrix, second_col_matrix, third_col_matrix), dim=2)

		return attn_bias
