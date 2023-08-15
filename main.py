"""
Run this script - runs the prompt set through the model to get n output sequences per prompt, 
also to get top n most likely output tokens for each prompt alongside their probabilities.
"""

from create_prompt_set import prompt_set
import run_prompts

wizardLM = 'WizardLM/WizardLM-13B-V1.2'
falcon = 'tiiuae/falcon-7b'
open_llama = 'openlm-research/open_llama_7b'

output_dir = 'outputs/falcon'
model_name = falcon
subsets_dict = prompt_set.get_subsets_dict()

# run prompts to analyse frequency of outputs
for subset_name, subset in subsets_dict.items():
    print('Running prompts for ' + subset_name)
    run_prompts.run_for_seqs(
        subset, 
        run_prompts.gen_filename(subset_name + '_seqs', output_dir), 
        model_name
        )

# run prompts to get confidence scores of possible outputs
for subset_name, subset in subsets_dict.items():
    print('Running prompts for ' + subset_name)
    run_prompts.run_for_scores(
        subset, 
        run_prompts.gen_filename(subset_name + '_scores', output_dir), 
        model_name
        )