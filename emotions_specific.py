"""
Computes confidence scores for specific outputs for emotion prompts.
"""

from create_prompt_set import prompt_set
import run_prompts, sys

wizardLM = 'WizardLM/WizardLM-13B-V1.2'
falcon = 'tiiuae/falcon-7b'
open_llama = 'openlm-research/open_llama_7b'
emotions = [
    'happiness', 'joy', 'respect', 'love', 'compassion', 'admiration', 'hope', 'recognition', 'excitement', 'empathy',
    'surprise', 'confusion', 'curiosity', 'relief', 'awe', 'wonder',
    'fear', 'anger', 'worry', 'shock', 'disgust', 'pity', 'sympathy', 'contempt', 'sorrow', 'sadness', 'concern'
]

subsets_dict = prompt_set.get_subsets_dict()

type = "scores"
model_choice = sys.argv[1]
subset = subsets_dict['emotion']

if model_choice == 'falcon':
    model_name = falcon
    output_dir = 'outputs/falcon'
elif model_choice == 'open_llama':
    model_name = open_llama
    output_dir = 'outputs/open_llama'


print(f'[SCORES] Running prompts for emotion_specific (size: {len(subset)})')
run_prompts.run_for_scores(
    subset, 
    run_prompts.gen_filename('emotion_specific_scores', output_dir), 
    model_name,
    selected_outputs = emotions
    )