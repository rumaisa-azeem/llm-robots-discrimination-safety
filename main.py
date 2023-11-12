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

subsets_dict = prompt_set.get_subsets_dict()

type = sys.argv[1]
subset_name = sys.argv[2]
model_choice = sys.argv[3]
subset = subsets_dict[subset_name]

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