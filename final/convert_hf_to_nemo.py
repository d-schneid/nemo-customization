from pathlib import Path

from structure_aware_customization.model.structure_aware_starcoder2_model import StructureAwareStarcoder2Model
from structure_aware_customization.model.structure_aware_starcoder2_config import StructureAwareStarcoder2Config

from nemo.collections import llm


if __name__ == "__main__":
	#model = llm.Starcoder2Model(llm.Starcoder2Config3B())
	model = StructureAwareStarcoder2Model(StructureAwareStarcoder2Config())
	llm.import_ckpt(model=model, source='hf://local_hf_dir', output_path=Path('output_dir'), overwrite=True)
