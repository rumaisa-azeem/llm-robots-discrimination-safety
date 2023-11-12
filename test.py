from run_prompts2 import write_scores_out
from prompt_set import PromptSet

fake_prompt_set = {
    "prompt1": {'type': 'categorisation', 'subset': 'affordance', 'dimension':'sexuality', 'outputs': None, 'base_prompt': 'prompt'},
    "prompt2": {'type': 'categorisation', 'subset': 'affordance', 'dimension':'sexuality', 'outputs': None, 'base_prompt': 'prompt'},
    "prompt3": {'type': 'generation', 'subset': 'affordance', 'dimension':'sexuality', 'outputs': None, 'base_prompt': 'prompt'}
}
prompt_set = PromptSet(fake_prompt_set)
fake_scores_dict = {
    "prompt1" : [('word1', 0.34234), ('word2',0.1234), ('word3', 0.3341), ('word4', 0.111)],
    "prompt2" : [('word1', 0.34234), ('word2',0.1234), ('word3', 0.3341)],
    "prompt3" : [('word1', 0.34234), ('word2',0.1234)]
}

write_scores_out(fake_scores_dict, prompt_set, 'fake_scores')

