from torch.utils.data import Dataset
from typing import Dict

class PromptSet(Dataset):
    '''
    A dataset of prompts. Consists of a dictionary where: 
        - key = prompt 
        - value = list of tokens to use when calculating split between different options

    Using [] on the dataset will return:
        - the prompt at the specified index if an integer is specified
        - the list of tokens for the specified prompt if a string is specified
    '''
    def __init__(self, prompts_dict:Dict[str, list]):
        self.prompts_dict = prompts_dict
    
    def __len__(self):
        return len(self.prompts_dict)
    
    def __getitem__(self, index): 
        '''
        Returns the prompt at the specified index if an integer is specified, 
        or the list of tokens for the specified prompt if a string is specified.
        '''
        if index.__class__ == int: 
            return list(self.prompts_dict.keys())[index]
        return self.prompts_dict[index] 
    
    def __str__(self) -> str:
        string = ''
        for prompt in self.prompts_dict.keys():
            string += prompt + '\n'
        return string

    def add_data(self, template_prompt:str, tokens:list):
        '''
        Add a prompt to the prompt set. Prompt is constructed by replacing <> with the tokens in the list.
        
        Parameters:
        - template_prompt: The template prompt to use. Should contain just <1>, or <1> and <2> to indicate where the tokens should be placed.
        - tokens: The list of token(s) to be inserted into the template prompt.
        '''
        if len(tokens) == 1:
            prompt = template_prompt.replace("<1>", tokens[0])
            self.prompts_dict[prompt] = tokens
        elif len(tokens) == 2:
            prompt = template_prompt.replace("<1>", tokens[0]).replace("<2>", tokens[1])
            self.prompts_dict[prompt] = tokens
            prompt = template_prompt.replace("<1>", tokens[1]).replace("<2>", tokens[0])
            self.prompts_dict[prompt] = tokens
    
    