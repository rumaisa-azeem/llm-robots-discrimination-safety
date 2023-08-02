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

    # def add_data(self, template_prompt:str, tokens:list):
    #     '''
    #     Add a prompt to the prompt set. Prompt is constructed by replacing <> with the tokens in the list.
        
    #     Parameters:
    #     - template_prompt: The template prompt to use. Should contain just <1>, or <1> and <2> to indicate where the tokens should be placed.
    #     - tokens: The list of token(s) to be inserted into the template prompt.
    #     '''
    #     if len(tokens) == 1:
    #         prompt = template_prompt.replace("<1>", tokens[0])
    #         self.prompts_dict[prompt] = tokens
    #     elif len(tokens) == 2:
    #         prompt = template_prompt.replace("<1>", tokens[0]).replace("<2>", tokens[1])
    #         self.prompts_dict[prompt] = tokens
    #         prompt = template_prompt.replace("<1>", tokens[1]).replace("<2>", tokens[0])
    #         self.prompts_dict[prompt] = tokens
    
    def print_all_prompts(self):
        for prompt in self.prompts_dict.keys():
            print(prompt)
   
    def add_data(self, template_prompt:str, input_tokens:list, output_tokens:list=None):
            '''
            Add data to the prompt set. If a prompt is open-ended, do not specify output_tokens.
            If a prompt is close-ended, specify output_tokens.

            Parameters:
            - template_prompt: The prompt in which to insert the input tokens. Should contain just <1>, or <1> and <2> to indicate where the tokens should be placed.
            - input_tokens: The list of token(s) to be inserted into the template prompt.
            - output_tokens: The list of token(s) to be used to calculate the split between different output categories.
            '''
            if len(input_tokens) == 2 and "<1>" in template_prompt and "<2>" in template_prompt: # prompt with 2 input tokens
                prompt = self.add_input_token(self.add_input_token(template_prompt, "<1>", input_tokens[0]), "<2>", input_tokens[1])
                # prompt = template_prompt.replace("<1>", input_tokens[0]).replace("<2>", input_tokens[1])
                self.prompts_dict[prompt] = output_tokens
                prompt = self.add_input_token(self.add_input_token(template_prompt, "<1>", input_tokens[1]), "<2>", input_tokens[0])
                # prompt = template_prompt.replace("<1>", input_tokens[1]).replace("<2>", input_tokens[0])
                self.prompts_dict[prompt] = output_tokens
            elif len(input_tokens) == 1 and "<2>" not in template_prompt: # prompt with 1 input token
                # prompt = template_prompt.replace("<1>", input_tokens[0])
                prompt = self.add_input_token(template_prompt, "<1>", input_tokens[0])
                self.prompts_dict[prompt] = output_tokens
            else:
                raise Exception('Invalid template_prompt or input_tokens')
                    
    def add_input_token(self, base_prompt, placeholder, token):
        '''
        Insert the input token into the base prompt at the specified placeholder, with grammar checks.
        '''
        ph_pos = base_prompt.index(placeholder)
        if 'the' not in base_prompt[ph_pos-4:ph_pos]:
            if token[0] in 'aeiou':
                base_prompt = base_prompt.replace(placeholder, f'an {token}')
            else:
                base_prompt = base_prompt.replace(placeholder, f'a {token}')
        else:
            base_prompt = base_prompt.replace(placeholder, token)
        # base_prompt_words = base_prompt.split()
        # test_token = base_prompt_words[base_prompt_words.index(placeholder)-1]
        # test_token = base_prompt.index(placeholder)-4

        # if not test_token == 'the': # if the token is not preceded by 'the', must add 'a' or 'an' before it
        #     if token[0] in 'aeiou':
        #         base_prompt.replace(placeholder, f'an {token}')
        #     else:
        #         base_prompt.replace(placeholder, f'a {token}')
        # else:
        #     base_prompt.replace(placeholder, token)
        
        return base_prompt