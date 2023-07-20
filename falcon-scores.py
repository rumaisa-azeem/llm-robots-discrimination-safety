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
output = model(input_ids, output_hidden_states=True, return_dict=True) 
print(output.logits)
print(output.logits.__class__)
sm = torch.softmax(output.logits, dim=-1)
print(sm)
