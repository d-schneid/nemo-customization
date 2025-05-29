from attn_mask import AttnMask

import numpy as np


class CodeCompletionAttnMask(AttnMask):

	def __init__(self):
		super().__init__(save_dir_suffix='code_completion')

	def get_cols(self):
		return [
			'attn_code_tokens',
			'attn_ast_leaves',
			'attn_dfg_edges',
			'attn_code_ast',
			'attn_code_dfg',
		]

	def build_attention_matrix(self, row, attn_col, num_targets, attn_col_offset):
		code_tokens = row['code_tokens'].split(',')
		num_code_tokens = len(code_tokens)

		attention_matrix = [[-1e9] * num_targets for _ in range(num_code_tokens)]

		for j, code_token_idxs in enumerate(row[attn_col]):
			for i in code_token_idxs:
				attention_matrix[i][j + attn_col_offset] = 0 # adjust for padding

		return attention_matrix

	def generate_adj_matrix(self, edges, num_nodes):
		adj_matrix = [[-1e9] * num_nodes for _ in range(num_nodes)]

		for to_node, from_nodes in edges:
			for from_node in from_nodes:
				if from_node <= to_node:
					adj_matrix[to_node][from_node] = 0

		return adj_matrix

	def masked_attention(self, row):
		length = len(row.split(','))
		mask = np.triu(np.ones((length, length), dtype=float) * -1e9, k=1)
		mask = mask + np.tril(np.zeros((length, length), dtype=float))

		return mask.tolist()

	def compute_attention_masks(self, data):
		data['attn_code_tokens'] = data['code_tokens'].apply(self.masked_attention)
		data['attn_ast_leaves'] = data['lr_paths_len'].apply(self.masked_attention)
		data['attn_dfg_edges'] = data.apply(
			lambda row: self.generate_adj_matrix(
			row['dfg_edges'], len(row['dfg_node_mask'].split(','))),axis=1
		)

		data['attn_code_ast'] = data.apply(
			lambda row: self.build_attention_matrix(
				row=row,
				attn_col='ast_leaf_code_token_idxs',
				num_targets=len(row['lr_paths_len'].split(',')),
				attn_col_offset=1  # adjust for padding of AST leaves
			),
			axis=1
		)

		data['attn_code_dfg'] = data.apply(
			lambda row: self.build_attention_matrix(
				row=row,
				attn_col='dfg_node_code_token_idxs',
				num_targets=len(row['dfg_node_mask'].split(',')),
				attn_col_offset=1  # adjust for padding of DFG nodes
			),
			axis=1
		)

		return data
