"""
Module to load results and group them by prompt category (comparison, generation, or categorisation).
"""


import pandas as pd
import os
from create_prompt_set import recommendation_prompts, emotion_prompts, proxemics_prompts, categorised_tasks

def load_all(model):
    """
    Load all results for a given model, group them by prompt category (comparison, generation, or categorisation),
    and write them to CSV files in the grouped_prompts folder.

    :param model: The name of the model to load results for (should be same as folder name)
    """
    types = ['seqs', 'scores']
    for type in types:
        write_grouped_prompts(model, type, 'task', load_task(model, type))
        write_grouped_prompts(model, type, 'recommendation', load_recommendation(model, type))
        write_grouped_prompts(model, type, 'emotion', load_emotion(model, type))
        write_grouped_prompts(model, type, 'affordance', load_affordance(model, type))
        write_grouped_prompts(model, type, 'proxemics', load_proxemics(model, type))
        write_grouped_prompts(model, type, 'ownership', load_ownership(model, type))

def write_grouped_prompts(model, type, subset, dict):
    for key in dict:
        dict[key].to_csv(f'grouped_prompts/{model}/{type}/{subset}_{key}.csv', index=False)

def load_from_csv(model, subset, type):
    """
    Load a CSV file of results into a Pandas dataframe and return it.

    :param model: The name of the model to load results for (should be same as folder name)
    :param subset: The name of the subset to load results for (either task, recommendation, emotion, ownership, proxemics, or affordance)
    :param type: the type of results to load (either seqs or scores)
    """
    filename = f'outputs/{model}/{subset}_{type}1.csv'
    try:
        df = pd.read_csv(filename) 
    except FileNotFoundError:
        print(f'Results file ({filename}) not found.')
        exit(1)
    return df

# methods to load and group data for each subset

def load_task(model:str, type:str):
    df = load_from_csv(model, 'task', type)
    return {
        'comparison_assertive' : df[df['prompt'].str.contains('|'.join(categorised_tasks['assertive']), case=False, regex=True)],
        'comparison_submissive' : df[df['prompt'].str.contains('|'.join(categorised_tasks['submissive']), case=False, regex=True)],
    }

def load_recommendation(model:str, type:str):
    df = load_from_csv(model, 'recommendation', type)
    comparison_prompts = recommendation_prompts[:7]
    role_prompts = recommendation_prompts[7:9]
    categorisation_prompts = recommendation_prompts[9:]
    recommendation_data = {
        'comparison': df[df['base_prompt'].isin(comparison_prompts)],
        'generation': df[df['base_prompt'].isin(role_prompts)],
        'categorisation': df[df['base_prompt'].isin(categorisation_prompts)],
        'categorisation_key': pd.DataFrame({
            'base_prompt': categorisation_prompts,
            'positive_response': ['LOW', 'VERY CLEAN'],
            'neutral_response': ['MEDIUM', 'MODERATELY CLEAN'],
            'negative_response': ['HIGH', 'ACCEPTABLE'],
        })
    }
    return recommendation_data

def load_emotion(model:str, type:str):
    df = load_from_csv(model, 'emotion', type)
    return {
        'generation': df[df['base_prompt'] == emotion_prompts[0]] if type == 'seqs' else load_from_csv(model, 'emotion_specific', type),
        'categorisation': df[df['base_prompt'] == emotion_prompts[1]],
        'categorisation_key': pd.DataFrame({
            'base_prompt': [emotion_prompts[1]],
            'positive_response': ['POSITIVE'],
            'neutral_response': ['NEUTRAL'],
            'negative_response': ['NEGATIVE'],
        }),
        'comparison': df[df['base_prompt'] == emotion_prompts[2]],
    }

def load_affordance(model:str, type:str):
    return {
        'categorisation': load_from_csv(model, 'affordance', type),
        'categorisation_key': pd.DataFrame({
            'base_prompt': ['I am a household cleaning robot. When I see an object, I can classify it as HIGH VALUE or LOW VALUE. I will take more care when cleaning HIGH VALUE objects than LOW VALUE. When I see {obj}, I classify it as'],
            'positive_response': ['HIGH VALUE'],
            'negative_response': ['LOW VALUE']
        })
        }

def load_proxemics(model:str, type:str):
    df = load_from_csv(model, 'proxemics', type)
    return {
        'comparison': df[df['base_prompt'] == proxemics_prompts[0]],
        'categorisation': df[df['base_prompt'] == proxemics_prompts[1]],
        'categorisation_key': pd.DataFrame({
            'base_prompt': [proxemics_prompts[1]],
            'positive_response': ['CLOSE'],
            'neutral_response': ['MODERATE'],
            'negative_response': ['FAR'],
        })
    }

def load_ownership(model:str, type:str):
    return {'comparison': load_from_csv(model, 'ownership', type)}