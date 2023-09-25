"""
Run this script - runs the prompt set through the model to get n output sequences per prompt, 
also to get top n most likely output tokens for each prompt alongside their probabilities.
"""

from create_prompt_set import prompt_set
import run_prompts, sys

wizardLM = 'WizardLM/WizardLM-13B-V1.2'
falcon = 'tiiuae/falcon-7b'
open_llama = 'openlm-research/open_llama_7b'
emotions = [
    'happiness', 'joy', 'respect', 'love', 'compassion', 'admiration', 'hope', 'recognition', 'excitement', 'empathy',
    'surprise', 'confusion', 'curiosity', 'relief', 'awe', 'wonder',
    'fear', 'anger', 'worry', 'shock', 'disgust', 'pity', 'sympathy', 'contempt', 'sorrow', 'sadness', 'concern'
]

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


if type == "sequences": # run prompts to analyse frequency of outputs
    print(f'[SEQUENCES] Running prompts for {subset_name} (size: {len(subset)})')
    run_prompts.run_for_seqs(
        subset, 
        run_prompts.gen_filename(subset_name + '_seqs', output_dir), 
        model_name,
        batch_size=8
        )
elif type == "scores": # run prompts to get confidence scores of possible outputs
    print(f'[SCORES] Running prompts for {subset_name} (size: {len(subset)})')
    run_prompts.run_for_scores(
        subset, 
        run_prompts.gen_filename(subset_name + '_scores', output_dir), 
        model_name,
        )


# run prompts to analyse frequency of outputs
# for subset_name, subset in subsets_dict.items():
#     print(f'Running prompts for {subset_name} (size: {len(subset)})')
#     run_prompts.run_for_seqs(
#         subset, 
#         run_prompts.gen_filename(subset_name + '_seqs', output_dir), 
#         model_name,
#         batch_size=8
#         )

# # run prompts to get confidence scores of possible outputs
# for subset_name, subset in subsets_dict.items():
#     print(f'Running prompts for {subset_name} (size: {len(subset)})')
#     run_prompts.run_for_scores(
#         subset, 
#         run_prompts.gen_filename(subset_name + '_scores', output_dir), 
#         model_name,
#         batch_size=8
#         )