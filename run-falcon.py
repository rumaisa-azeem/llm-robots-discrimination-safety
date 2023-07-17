from transformers import AutoTokenizer
import transformers
import torch
import random
from datetime import datetime
from typing import TextIO
from inputs import *
from itertools import combinations


model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

def run_prompt(input:str, output_file:TextIO, max_new_tok:int=5, num_ret_seq:int=1, ret_full_text:bool=False, ):
    sequences = pipeline(
            input,
            max_new_tokens=max_new_tok,
            do_sample=True,
            top_k=10,
            num_return_sequences=num_ret_seq,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
	        return_full_text=ret_full_text
        )
    
    output_file.write("\n>> " + input)
    for seq in sequences:
        output_file.write("\n"+seq['generated_text'])

    return sequences


def calc_split(sequences:list, tokens:list):
    counts = {}
    for seq in sequences:
        for token in tokens:
            if token in seq['generated_text'].split()[0]:
                counts[token] = counts.get(token, 0) + 1
    return counts

    
filename = "outputs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"

with (open(filename, "w")) as f:
    for prompt in task_prompts:
        # gender split
        f.write('\n\n')
        fem_name = random.choice(fem_names)
        masc_name = random.choice(masc_names)

        input = prompt.replace("<1>", masc_name).replace("<2>", fem_name)
        out = run_prompt(input, f, num_ret_seq=5)
        f.write('\n' + str(calc_split(out, [fem_name, masc_name])))

        input = prompt.replace("<1>", fem_name).replace("<2>", masc_name)
        out = run_prompt(input, f, num_ret_seq=5)
        f.write('\n' + str(calc_split(out, [fem_name, masc_name])))

        if not prompt.startswith("I need help"):
            input = prompt.replace("<1>", 'him').replace("<2>", 'her')
            out = run_prompt(input, f, num_ret_seq=5)
            f.write('\n' + str(calc_split(out, ['him', 'her'])))
            input = prompt.replace("<1>", 'her').replace("<2>", 'him')
            out = run_prompt(input, f, num_ret_seq=5)
            f.write('\n' + str(calc_split(out, ['her', 'him'])))
        else:
            input = prompt.replace("<1>", 'he').replace("<2>", 'she')
            out = run_prompt(input, f, num_ret_seq=5)
            f.write('\n' + str(calc_split(out, ['him', 'her', 'he', 'she'])))
            input = prompt.replace("<1>", 'she').replace("<2>", 'he')
            out = run_prompt(input, f, num_ret_seq=5)
            f.write('\n' + str(calc_split(out, ['her', 'him', 'he', 'she'])))

        # ethnicity split
        ethnicities = adj_qualifications['ethnicity']
        for i in  combinations(ethnicities, 2):
            input = prompt.replace("<1>", i[0]).replace("<2>", i[1])
            out = run_prompt(input, f, num_ret_seq=5)
            f.write('\n' + str(calc_split(out, [i[0], i[1]])))