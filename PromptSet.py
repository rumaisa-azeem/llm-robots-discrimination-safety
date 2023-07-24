from torch.utils.data import Dataset
from typing import Dict

class PromptSet(Dataset):
    def __init__(self, prompts_dict:Dict[str, list]):
        # prompts_dict: key = prompt, value = list of tokens to use when calculating split
        self.prompts_dict = prompts_dict
    
    def __len__(self):
        return len(self.prompts_dict)
    
    def __getitem__(self, index): 
        if index.__class__ == int: # if a number is specified return the prompt at that index
            return list(self.prompts_dict.keys())[index]
        return self.prompts_dict[index] # otherwise return the list of tokens for the specified prompt

    def add_data(self, template_prompt:str, tokens:list):
        if len(tokens) == 1:
            prompt = template_prompt.replace("<1>", tokens[0])
            self.prompts_dict[prompt] = tokens
        elif len(tokens) == 2:
            prompt = template_prompt.replace("<1>", tokens[0]).replace("<2>", tokens[1])
            self.prompts_dict[prompt] = tokens
            prompt = template_prompt.replace("<1>", tokens[1]).replace("<2>", tokens[0])
            self.prompts_dict[prompt] = tokens
    
    def __str__(self) -> str:
        string = ''
        for prompt in self.prompts_dict.keys():
            string += prompt + '\n'
        return string