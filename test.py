from transformers import AutoTokenizer

model_checkpoint = "vinai/bartpho-syllable"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(tokenizer.is_fast)