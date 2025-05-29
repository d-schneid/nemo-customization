from abc import ABC, abstractmethod


class AttnMask(ABC):

	def __init__(self, save_dir_suffix):
		self.save_dir_suffix = save_dir_suffix

	@abstractmethod
	def compute_attention_masks(self, data):
		pass

	@abstractmethod
	def get_cols(self):
		pass
