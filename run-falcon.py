from transformers import AutoTokenizer
import transformers
import torch
import random
from datetime import datetime
from typing import TextIO
from inputs import *
from itertools import combinations
from tqdm import tqdm
import os


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


def run_pipeline(input:str, output_file:TextIO, max_new_tok:int=5, num_ret_seq:int=1, ret_full_text:bool=False, ):
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
        output_file.write("\n> "+seq['generated_text'])

    return sequences


def calc_split(sequences:list, tokens:list):
    counts = {}
    for seq in sequences:
        for token in tokens:
            # if token in seq['generated_text'].split()[0]:
            #     counts[token] = counts.get(token, 0) + 1
            if seq['generated_text'].strip().startswith(token):
                counts[token] = counts.get(token, 0) + 1
    return counts


def run_2_var_prompt(prompt:str, var1:str, var2:str, output_file:TextIO, num_ret_seq:int=5):
    '''
    run a prompt with 2 variables, write results to file, and calculate split between the two inputs
    '''
    input = prompt.replace("<1>", var1).replace("<2>", var2)
    out = run_pipeline(input, output_file, num_ret_seq=num_ret_seq)
    output_file.write('\n' + str(calc_split(out, [var1, var2])))

    input = prompt.replace("<1>", var2).replace("<2>", var1)
    out = run_pipeline(input, output_file, num_ret_seq=num_ret_seq)
    output_file.write('\n' + str(calc_split(out, [var1, var2])))


def run_1_var_prompt(prompt:str, var:str, output_file:TextIO, num_ret_seq:int=5):
    '''
    run a prompt with 1 variable, write results to file
    '''
    input = prompt.replace("<1>", var)
    out = run_pipeline(input, output_file, num_ret_seq=num_ret_seq)


def gen_filename(type:str=''):
    output_dir = 'outputs'
    next_filename = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = os.listdir(output_dir)
    if not filenames:
        next_filename = os.path.join(output_dir, f'{type}1.txt')
    else:
        # Find the highest number used in the filenames
        max_number = max([int(os.path.splitext(filename)[0]) for filename in filenames])
        next_number = max_number + 1
        next_filename = os.path.join(output_dir, f'{type}{next_number}.txt')

    return next_filename
    
# -----------------------------------------------------------------------------------------

with (open(gen_filename('tasks'), "w")) as f:
    for task_prompt in tqdm(task_prompts, desc='running task prompts'):

        # gender - choose random pairs of fem/masc names to compare
        f.write('\n\nGENDER\n\n')
        fem_name = random.choice(fem_names)
        masc_name = random.choice(masc_names)
        run_2_var_prompt(task_prompt, fem_name, masc_name, f)
        run_2_var_prompt(task_prompt, 'him', 'her', f)

        # ethnicity - try white vs poc and caucasian vs poc
        f.write('\n\nETHNICITY\n\n')
        for i in  adj_qualifications['ethnicity']:
            run_2_var_prompt(task_prompt, 'the white person', 'the '+i+' person', f)
            run_2_var_prompt(task_prompt, 'the caucasian person', 'the '+i+' person', f)

        # disabiliity - try disabled vs abled
        f.write('\n\nDISABILITY\n\n')
        for i in  noun_qualifications['disability']:
            run_2_var_prompt(task_prompt, 'the disabled person', 'the '+i+' person', f)

        # religion - try all possible pairs of religions
        f.write('\n\nRELIGION\n\n')
        for i in combinations(noun_qualifications['religion'], 2):
            i0 = 'the ' + i[0]
            i1 = 'the ' + i[1]
            run_2_var_prompt(task_prompt, i0, i1, f)

        
with open(gen_filename('body_language', 'w')) as f:
    for body_language_prompt in tqdm(body_language_prompts, desc='running body language prompts'):
            
            # gender
            f.write('\n\nGENDER\n\n')
            run_1_var_prompt(body_language_prompt, 'him', f)
            run_1_var_prompt(body_language_prompt, 'her', f)
            run_1_var_prompt(body_language_prompt, 'a man', f)
            run_1_var_prompt(body_language_prompt, 'a woman', f)

            # ethnicity - try white/caucasian vs poc
            f.write('\n\nETHNICITY\n\n')
            for i in adj_qualifications['ethnicity']:
                run_1_var_prompt(body_language_prompt, 'a '+i+' person', f)
                run_1_var_prompt(body_language_prompt, 'a white person', f)
                run_1_var_prompt(body_language_prompt, 'a caucasian person', f)

            # disability - try abled vs disabled
            f.write('\n\nDISABILITY\n\n')
            for i in noun_qualifications['disability']:
                run_1_var_prompt(body_language_prompt, 'an able-bodied person', f)
                run_1_var_prompt(body_language_prompt, 'a '+ i + ' person', f)

            # religion - try all possible religions
            f.write('\n\nRELIGION\n\n')
            for i in noun_qualifications['religion']:
                run_1_var_prompt(body_language_prompt, 'a '+i, f)
            for i in adj_qualifications['religion']:
                run_1_var_prompt(body_language_prompt, 'a '+i+' person', f)
         
