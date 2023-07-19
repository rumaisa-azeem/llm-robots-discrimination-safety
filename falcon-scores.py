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
tokenizer = AutoTokenizer.from_pretrained(model_name)

input = "Once upon a time"
input_ids = tokenizer.encode(input, return_tensors='pt')
output = model.generate(input_ids, max_length = 20, output_scores=True, output_attentions=True, output_hidden_states=True, return_dict=True) 

for i, generated_seq in enumerate(output.sequences):
    decoded_output = tokenizer.decode(generated_seq, skip_special_tokens=True)
    sequence_probabilities = output.scores[i].tolist()
    print(f"Generated Sequence {i+1}: {decoded_output}")
    print(f"Probabilities: {sequence_probabilities}\n")


