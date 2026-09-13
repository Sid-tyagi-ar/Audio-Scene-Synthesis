from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

encoded_texts = tokenizer(["ABCDEF 1235", " "], padding=True, truncation=True, return_tensors="pt").to(device = 'cuda')
print(encoded_texts, type(encoded_texts))