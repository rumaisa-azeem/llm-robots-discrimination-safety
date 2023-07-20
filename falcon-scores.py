from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import random
from datetime import datetime
from typing import TextIO
from inputs import *
from itertools import combinations
from tqdm import tqdm
from vocab import vocab_list

model_name = 'tiiuae/falcon-7b'
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_cache=True)

input = "The first letter in the alphabet is"
input_ids = tokenizer.encode(input, return_tensors='pt')
output = model(input_ids, return_dict=True, use_cache=True)
next_token_logits = output.logits[:, -1, :]
probs = torch.softmax(next_token_logits, dim=-1)
probs = probs.tolist()[0] # probabilities for each word in the vocabulary being part of the output sequence
vocab = tokenizer.get_vocab()
word_probs = []
for index, val in enumerate(probs): # create a list of each word and its probability as a tuple
    word_probs.append((vocab_list[index], val))

sorted_probs = sorted(word_probs, reverse=True, key=lambda x: x[1])
print(sorted_probs[:10])


output = model.generate(input_ids)
output_ids = output.tolist()[0]
text = tokenizer.decode(output_ids)
print('>> ' + text)

