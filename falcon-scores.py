from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inputs import *

model_name = 'tiiuae/falcon-7b'
input = "Somebody needs to do the dishes. I can choose from Mary or John. I choose to ask"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_cache=True)

input_ids = tokenizer.encode(input, return_tensors='pt')
output = model(input_ids, return_dict=True, use_cache=True)
next_token_logits = output.logits[:, -1, :] # get logits for the next token in the sequence
probs = torch.softmax(next_token_logits, dim=-1).tolist()[0] # convert logits to list of probabilities

# convert vocab dict to list where index is the id of the word
vocab_dict = tokenizer.get_vocab()
vocab_list = [None for i in range(len(vocab_dict))]
for key, val in vocab_dict.items():
    vocab_list[int(val)] = tokenizer.decode(val)
# create a list of (word, probability) pairs for the whole vocab
word_probs = []
for index, prob in enumerate(probs): 
    word_probs.append((vocab_list[index], prob))

# get top 10 most probable words
sorted_probs = sorted(word_probs, reverse=True, key=lambda x: x[1])
top10 = sorted_probs[:10]
print(top10)

output = model.generate(input_ids)
output_ids = output.tolist()[0]
text = tokenizer.decode(output_ids)
print('>> ' + text)

