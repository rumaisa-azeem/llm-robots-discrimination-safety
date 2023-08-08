'''
Module for the PromptSet class used to represent prompts in a dataset (allows for maximum efficiency when running on model).
'''

from torch.utils.data import Dataset, Subset

class PromptSet(Dataset):
    """
    A dataset of prompts. Consists of a dictionary where:
        - key = prompt
        - value = dictionary:
            - subset: The name of the subset the prompt belongs to
            - output_categories: A list of expected output categories for the prompt, None if the prompt is open-ended
    """
    def __init__(self, prompts_dict:dict):
        self.prompts_dict = prompts_dict
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
            - value: Subset object
        """
        subset_names = []
        for val in self.prompts_dict.values():
            subset = val['subset']
            if subset not in subset_names:
                subset_names.append(subset)

        for subset_name in subset_names:
            subset = Subset(self, [i for i, val in enumerate(self.prompts_dict.values()) if val['subset'] == subset_name])
            self.subsets_dict[subset_name] = subset
    
    def get_subset(self, subset_name:str):
        """
        Get a subset of prompts from the prompt set.

        :param subset_name: The name of the subset to get
        :return: A Subset object containing the prompts in the specified subset
        """
        return self.subsets_dict[subset_name]
    
    def get_subsets_dict(self):
        """
        Get all subsets in the prompt set, as a dictionary of subset names and Subset objects.
        """
        return self.subsets_dict
    
    def get_expected_outputs(self, prompt:str):
        """
        Returns the expected output categories for the specified prompt as a list.
        """
        return self.prompts_dict[prompt]['output_categories']
    
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
        Returns a dict_items object containing all prompts in the prompt set and their associated 
        information (consists of the subset the prompt belongs to and a list of the prompt's expected outputs)
        """
        return self.prompts_dict.items()