from attn_mask import AttnMask

import numpy as np


class CodeTextAttnMask(AttnMask):

	def __init__(self):
		super().__init__(save_dir_suffix='code_text')

	def get_cols(self):
		return [
			'text_tokens',
			'text_tokens_rel_pos_ids',
			'attn_text_tokens',
			'attn_code_tokens',
			'attn_ast_leaves',
			'attn_dfg_edges',
			'attn_code_ast',
			'attn_code_dfg',
			'attn_code_text',
			'attn_ast_text',
			'attn_dfg_text',
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
				adj_matrix[to_node][from_node] = 0

		return adj_matrix

	def masked_attention(self, row):
		length = len(row.split(','))
		mask = np.triu(np.ones((length, length), dtype=float) * -1e9, k=1)
		mask = mask + np.tril(np.zeros((length, length), dtype=float))

		return mask.tolist()

	def full_attention(self, row, row_name):
		col_len = len(row['text_tokens'].split(','))
		row_len = len(row[row_name].split(','))

		return [[-1e9 for _ in range(col_len)] for _ in range(row_len)]

	def compute_attention_masks(self, data):
		data['attn_text_tokens'] = data['text_tokens'].apply(self.masked_attention)
		data['attn_code_tokens'] = data['code_tokens'].apply(
			lambda row: [[0] * len(row.split(',')) for _ in range(len(row.split(',')))]
		)
		data['attn_ast_leaves'] = data['lr_paths_len'].apply(
			lambda row: [[0] * len(row.split(',')) for _ in range(len(row.split(',')))]
		)
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

		data['attn_code_text'] = data.apply(
			lambda row: self.full_attention(
				row=row,
				row_name='code_tokens'
			),
			axis=1
		)

		data['attn_ast_text'] = data.apply(
			lambda row: self.full_attention(
				row=row,
				row_name='lr_paths_len'
			),
			axis=1
		)

		data['attn_dfg_text'] = data.apply(
			lambda row: self.full_attention(
				row=row,
				row_name='dfg_node_mask'
			),
			axis=1
		)

		return data
