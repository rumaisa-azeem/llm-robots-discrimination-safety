from create_prompt_set import prompt_set
import run_prompts

subsets_dict = prompt_set.get_subsets_dict()
for subset_name in subsets_dict:
    run_prompts.run(subsets_dict[subset_name], run_prompts.gen_filename(subset_name))
