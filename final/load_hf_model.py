from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
	model_name = "bigcode/starcoder2-3b"
	model = AutoModelForCausalLM.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	local_dir = "local_dir"
	model.save_pretrained(local_dir)
	tokenizer.save_pretrained(local_dir)
