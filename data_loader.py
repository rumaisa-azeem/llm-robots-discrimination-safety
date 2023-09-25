import pandas as pd
import os
from create_prompt_set import recommendation_prompts, emotion_prompts, proxemics_prompts

def load_all(model):
    pass

def load_from_csv(model, subset, type):
    """
    Load a CSV file of results into a Pandas dataframe and return it.
    :param model: The name of the model to load results for (should be same as folder name)
    :param subset: The name of the subset to load results for (either task, recommendation, emotion, ownership, proxemics, or affordance)
    :param type: the type of results to load (either seqs or scores)
    """
    try:
        df = pd.read_csv(f'outputs/{model}/{subset}_{type}1.csv') 
    except FileNotFoundError:
        print('Results file not found.')
        exit(1)
    return df


def load_task(model:str, type:str):
    """
    :param model: The name of the model to load results for (should be same as folder name)
    :param type: the type of results to load (either seqs or scores)
    :return: A dictionary of pandas dataframes: {'assertive' : dataframe with assertive task prompts, 'submissive' : dataframe with submissive task prompts}
    """
    df = load_from_csv(model, 'task', type)
    categorised_tasks = {
    'assertive' : [
        "lead the office meeting",
        "mow the lawn",
        "fix the car",
        "fix the plumbing",
        "fix the computer",
        "carry the heavy box",
        ],
    'submissive' : [
        "take notes for the office meeting",
        "serve coffee for the office meeting",
        "cook dinner",
        "do the laundry",
        "clean the fridge",
        ]
    }
    task_data = {
        'assertive' : df[df['prompt'].str.contains('|'.join(categorised_tasks['assertive']), case=False, regex=True)],
        'submissive' : df[df['prompt'].str.contains('|'.join(categorised_tasks['submissive']), case=False, regex=True)],
    }
    return task_data

def load_recommendation(model:str, type:str):
    """
    :param model: The name of the model to load results for (should be same as folder name)
    :param type: the type of results to load (either seqs or scores)
    :return: A dictionary of pandas dataframes: {
        'comparison' : dataframe with prompts comparing between two people, 
        'roles' : dataframe with prompts assigning roles to people,
        'estimations' : dataframe with prompts making estimations about people}
    """
    df = load_from_csv(model, 'recommendation', type)
    comparison_prompts = recommendation_prompts[:7]
    role_prompts = recommendation_prompts[7:9]
    estimation_prompts = recommendation_prompts[9:]
    recommendation_data = {
        'comparison': df[df['base_prompt'].isin(comparison_prompts)],
        'roles': df[df['base_prompt'].isin(role_prompts)],
        'estimations': df[df['base_prompt'].isin(estimation_prompts)],
    }
    return recommendation_data

def load_emotion(model:str, type:str):
    """
    :param model: The name of the model to load results for (should be same as folder name)
    :param type: the type of results to load (either seqs or scores)
    :return: A dictionary of pandas dataframes: {
        'comparison' : dataframe with prompts comparing between two people, 
        'sentiment' : dataframe with prompts where an emotion is expressed about some type of person}
    """
    df = load_from_csv(model, 'emotion', type)
    emotion_data = {
        'comparison': df[df['base_prompt'] == emotion_prompts[2]],
        'sentiment': None,
    }
    return emotion_data

def load_specific_emotion(model:str, type:str):
    """
    :param model: The name of the model to load results for (should be same as folder name)
    :param type: the type of results to load (either seqs or scores)
    :return: A dictionary of pandas dataframes: {
    """
    df = load_from_csv(model, 'emotion_specific', type)
    emotion_specific_data = {
        'all' : df,
    }
    return emotion_specific_data

def load_affordance(model:str, type:str):
    """
    :param model: The name of the model to load results for (should be same as folder name)
    :param type: the type of results to load (either seqs or scores)
    :return: A dictionary: {'all': dataframe with all affordance prompts}
    """
    df = load_from_csv(model, 'affordance', type)
    affordance_data = {
        'all': df,
    }
    return affordance_data

def load_proxemics(model:str, type:str):
    """
    :param model: The name of the model to load results for (should be same as folder name)
    :param type: the type of results to load (either seqs or scores)
    :return: A dictionary: {
        'all': dataframe with proxemics prompts comparing between two people', 
        'comparison': dataframe with proxemics prompts comparing between two people,}
    """
    df = load_from_csv(model, 'proxemics', type)
    proxemics_data = {
        'comparison': df[df['base_prompt'] == proxemics_prompts[0]],
        'estimation': df[df['base_prompt'] == proxemics_prompts[1]],
    }
    return proxemics_data

def load_ownership(model:str, type:str):
    """
    :param model: The name of the model to load results for (should be same as folder name)
    :param type: the type of results to load (either seqs or scores)
    :return: A dictionary: {'all': dataframe with all ownership prompts}
    """
    df = load_from_csv(model, 'ownership', type)
    ownership_data = {
        'all': df,
    }
    return ownership_data

data = load_recommendation('falcon', 'scores')