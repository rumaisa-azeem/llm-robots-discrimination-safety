"""
Run this script - runs the prompt set through the model to get n output sequences per prompt, 
also to get top n most likely output tokens for each prompt alongside their probabilities.
"""

from create_prompt_set import prompt_set
import run_prompts, sys

falcon = 'tiiuae/falcon-7b'
open_llama = 'openlm-research/open_llama_7b'
vicuna13b = 'lmsys/vicuna-13b-v1.5'
mistral7b = 'mistralai/Mistral-7B-v0.1'
qwen2_7b = 'Qwen/Qwen2-7B'
qwen25_7b = 'Qwen/Qwen2.5-7B'
llama31_8b = 'meta-llama/Llama-3.1-8B'

subsets_dict = prompt_set.get_subsets_dict()

type = sys.argv[1]
subset_name = sys.argv[2]
model_choice = sys.argv[3]
subset = subsets_dict[subset_name]

#import random
#from prompt_set import PromptSet
#sampled_items = random.sample(subset.items(),20)
#subset = PromptSet({key:val for key,val in sampled_items})

if model_choice == 'falcon':
    model_name = falcon
    output_dir = 'outputs/falcon'
elif model_choice == 'open_llama':
    model_name = open_llama
    output_dir = 'outputs/open_llama'
elif model_choice == 'vicuna13b':
    model_name = vicuna13b
    output_dir = 'outputs/vicuna13b'
elif model_choice == 'mistral7b':
    model_name = mistral7b
    output_dir = 'outputs/mistral7b'
elif model_choice == 'qwen2_7b':
    model_name = qwen2_7b
    output_dir = 'outputs/qwen2_7b'
elif model_choice == 'qwen25_7b':
    model_name = qwen25_7b
    output_dir = 'outputs/qwen25_7b'
elif model_choice == 'llama31_8b':
    model_name = llama31_8b
    output_dir = 'outputs/llama31_8b'
elif model_choice == 'dummy':
    model_name = 'dummy'
    output_dir = 'outputs/dummy'
else:
    print('Unknown model')
    exit()

if type == "sequences": # run prompts to analyse frequency of outputs
    print(f'[SEQUENCES] Running prompts for {subset_name} (size: {len(subset)})')
    run_prompts.run_for_seqs(
        subset, 
        model_name,
        subset_name+'_seqs',
        output_dir,
        batch_size=8
    )
elif type == "scores": # run prompts to get confidence scores of possible outputs
    print(f'[SCORES] Running prompts for {subset_name} (size: {len(subset)})')
    run_prompts.run_for_scores(
        subset, 
        model_name,
        subset_name+'_scores',
        output_dir
    )
