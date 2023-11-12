"""
Run this script - runs the prompt set through the model to get n output sequences per prompt, 
also to get top n most likely output tokens for each prompt alongside their probabilities.
"""

from create_prompt_set import prompt_set
import run_prompts, sys

falcon = 'tiiuae/falcon-7b'
open_llama = 'openlm-research/open_llama_7b'
vicuna13b = 'lmsys/vicuna-13b-v1.5'
mistral7b = 'mistralai/Mistral-7B-v0.1'


# emotions list has been moved to create_prompt_set.py (line 47)

# emotions = [
#     'happiness', 'joy', 'sadness', 'fear', 'surprise', 'anger', 'disgust',
#     'neutral', 'neutrality', 'doubt', 'confusion', 'pity'
#     #'happiness', 'joy', 'respect', 'love', 'compassion', 'admiration', 'hope', 'recognition', 'excitement', 'empathy',
#     #'surprise', 'confusion', 'curiosity', 'relief', 'awe', 'wonder',
#     #'fear', 'anger', 'worry', 'shock', 'disgust', 'pity', 'sympathy', 'contempt', 'sorrow', 'sadness', 'concern'
# ]

#['joy',       'sadness', 'fear', 'surprise', 'anger', 'disgust'] # ekman 6
#['happiness', 'sadness', 'fear', 'surprise', 'anger', 'disgust', 'neutral'] # JAFFE face emotion corpus
#['happiness', 'sadness', 'fear', 'surprise', 'anger', 'disgust', 'neutral', 'doubt',     'pity'] # "A Facial Expression Emotion Recognition Based HRI System" (robot display emotions)
#['happiness', 'sadness', 'fear', 'surprise', 'anger', 'disgust', 'neutral', 'confusion', 'pity'] # "A Facial Expression Emotion Recognition Based HRI System" (my edit)


subsets_dict = prompt_set.get_subsets_dict()

type = sys.argv[1]
subset_name = sys.argv[2]
model_choice = sys.argv[3]
subset = subsets_dict[subset_name]

if model_choice == 'falcon':
    model_name = falcon
    output_dir = 'outputs/falcon'
elif model_choice == 'open_llama':
    model_name = open_llama
    output_dir = 'outputs/open_llama'
elif model_choice == 'vicuna13b':
    model_name = vicuna13b
    output_dir = 'outputs/vicuna13b'
elif model_choice == 'mistral7b':
    model_name = mistral7b
    output_dir = 'outputs/mistral7b'
else:
    print('Unknown model')
    exit()

if type == "sequences": # run prompts to analyse frequency of outputs
    print(f'[SEQUENCES] Running prompts for {subset_name} (size: {len(subset)})')
    run_prompts.run_for_seqs(
        subset, 
        run_prompts.gen_filename(subset_name + '_seqs', output_dir), 
        model_name,
        batch_size=8
    )
elif type == "scores": # run prompts to get confidence scores of possible outputs
    print(f'[SCORES] Running prompts for {subset_name} (size: {len(subset)})')
    run_prompts.run_for_scores(
        subset, 
        run_prompts.gen_filename(subset_name + '_scores', output_dir), 
        model_name,
    )


# run prompts to analyse frequency of outputs
# for subset_name, subset in subsets_dict.items():
#     print(f'Running prompts for {subset_name} (size: {len(subset)})')
#     run_prompts.run_for_seqs(
#         subset, 
#         run_prompts.gen_filename(subset_name + '_seqs', output_dir), 
#         model_name,
#         batch_size=8
#         )

# # run prompts to get confidence scores of possible outputs
# for subset_name, subset in subsets_dict.items():
#     print(f'Running prompts for {subset_name} (size: {len(subset)})')
#     run_prompts.run_for_scores(
#         subset, 
#         run_prompts.gen_filename(subset_name + '_scores', output_dir), 
#         model_name,
#         batch_size=8
#         )
