from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import random
from datetime import datetime
from typing import TextIO
from inputs import *
from itertools import combinations
from tqdm import tqdm

model_name = 'tiiuae/falcon-7b'
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

input = "Once upon a time"
input_ids = tokenizer.encode(input, return_tensors='pt')
output = model(input_ids, return_dict=True)
next_token_logits = output.logits[:, -1, :]
probs = torch.softmax(next_token_logits, dim=-1)
probs = probs.tolist()[0]
vocab = tokenizer.get_vocab()
word_probs = [(vocab[idx], prob) for idx, prob in enumerate(probs)]

# Sort the word_probabilities based on the probabilities in descending order
sorted_word_probs = sorted(word_probs, key=lambda x: x[1], reverse=True)

# Print the top 10 words and their probabilities
print("Top 10 Words and their Probabilities:")
for word, prob in sorted_word_probs[:10]:
    print(f"{word}: {prob:.4f}") 




