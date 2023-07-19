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
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

input = "Once upon a time"
input_ids = tokenizer.encode(input, return_tensors='pt')
output = model.generate(input_ids, max_length = 20, output_scores=True)
print(output.__class__)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output.__class__)
attr = dir(decoded_output)
for i in output:
    print(i)


#sequence_probabilities = output.scores[0].tolist()
#print(f"Generated Sequence {i+1}: {decoded_output}")
#print(f"Probabilities: {sequence_probabilities}\n")
