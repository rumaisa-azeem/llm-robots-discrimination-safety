import run_prompts

model, tokenizer = run_prompts.load_model('mistralai/Mistral-7B-v0.1')
tokens = tokenizer.encode('POSITIVE', return_tensors='pt')
for token in tokens:
    print(tokenizer.decode(token))

print()

tokens = tokenizer.encode('POSITIVE')
for token in tokens:
    print(tokenizer.decode(token))