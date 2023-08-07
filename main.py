"""
Run this script - runs the prompt set through the model to get mutliple output sequences per model, 
also to get top n most likely output words for each prompts alongside their probabilities.
"""

from create_prompt_set import prompt_set
import run_prompts

# run prompts to analyse frequency of outputs
subsets_dict = prompt_set.get_subsets_dict()
for subset_name in subsets_dict:
    run_prompts.run(subsets_dict[subset_name], run_prompts.gen_filename(subset_name))

# run prompts to get confidence scores of possible outputs
for subset_name in subsets_dict:
    subset = subsets_dict[subset_name]
    run_prompts.run_for_scores(subset, run_prompts.gen_filename(subset_name + '_scores'))