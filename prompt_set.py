'''
Module for the PromptSet class used to represent prompts in a dataset (allows for maximum efficiency when running on model).
'''

from torch.utils.data import Dataset

class PromptSet(Dataset):
    """
    A dataset of prompts. Consists of a dictionary where:
        - key = prompt
        - value = dictionary:
            - subset: The name of the subset the prompt belongs to
            - dimension: The dimension of bias which the prompt tests. If this is None, then the prompt may test for multiple dimensions/no specific dimension. 
            - type: The type of the prompt. Can be either 'comparison', 'categorisation', or 'generation'.
            - outputs: A list of expected output categories for the prompt, None if the prompt is open-ended (i.e. a 'generation' prompt)
            - base_prompt: The base prompt that this prompt was derived from.
    """
    def __init__(self, prompts_dict:dict):
        for prompt, prompt_data in prompts_dict.items():
            for required_key in ['subset', 'dimension', 'type', 'outputs', 'base_prompt']:
                if required_key not in prompt_data:
                    error_string = f"'{required_key}' is not in the input dictionary for this prompt: {prompt}"
                    raise KeyError(error_string)
        self.prompts_dict = prompts_dict # should probably write something to validate the prompts_dict
        self.subsets_dict = {}
        self.__init_subsets_dict()
    
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
    
    def __init_subsets_dict(self):
        """
        Create a dictionary of subsets within the full prompt set.
            - key: name of the subset
            - value: PromptSet object to represent the subset
        """
        subset_names = []
        for val in self.prompts_dict.values():
            subset = val['subset']
            if subset not in subset_names:
                subset_names.append(subset)
        if len(subset_names) == 1: # if only one subset name found, no need to create subsets_dict
            self.subsets_dict = None
            return
        for subset_name in subset_names:
            subset_dict = {prompt : self.prompts_dict[prompt] for prompt in self.prompts_dict if self.prompts_dict[prompt]['subset'] == subset_name}
            self.subsets_dict[subset_name] = PromptSet(subset_dict)
    
    def get_subset(self, subset_name:str):
        """
        Get a subset of prompts from the prompt set.

        :param subset_name: The name of the subset to get
        :return: A PromptSet object containing the prompts in the specified subset
        """
        return self.subsets_dict.get(subset_name, None)
    
    def get_subsets_dict(self):
        """
        Get all subsets in the prompt set, as a dictionary of subset names and PromptSet objects.
        """
        return self.subsets_dict
    
    def get_expected_outputs(self, prompt:str):
        """
        Returns the expected output categories for the specified prompt as a list.
        """
        return self.prompts_dict[prompt]['outputs']
    
    def get_prompt_type(self, prompt:str):
        """
        Returns the type of the specified prompt. Can be either 'comparison', 'categorisation', or 'generation'.
        """
        return self.prompts_dict[prompt]['type']
    
    def get_base_prompt(self, prompt:str):
        """
        Returns the base_prompt for the specified prompt.
        """
        return self.prompts_dict[prompt]['base_prompt']
    
    def get_dimension(self, prompt:str):
        """
        Returns the bias dimension this prompt tests for. If None, the prompt may test for multiple dimensions/no specific dimension.
        """
        return self.prompts_dict[prompt]['dimension']
    
    def print_prompts(self, subset_name:str=None):
        """
        Print all prompts in the prompt set, or all prompts in a specified subset.
        
        :param subset_name: (optional) The name of the subset to print instead of the full prompt set.
        """
        if subset_name:
            for prompt in self.subsets_dict[subset_name]:
                print(prompt)
        else:
            for prompt in self.prompts_dict.keys():
                print(prompt)

    def items(self):
        """
        Returns a dict_items object containing all prompts in the prompt set and their associated information.
        """
        return self.prompts_dict.items()
