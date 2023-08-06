# Run prompts on Falcon model w either full dataset of prompts or a subset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
from inputs import prompt_set, subsets_dict
import sys
from helpers import gen_filename

# input_set = prompt_set # default to full prompt set
# # if a subset argument is specified, use that subset instead of the full prompt set
# if len(sys.argv) > 1:
#     subset_arg = sys.argv[1]
#     if not subset_arg.isidentifier():
#         print("Invalid subset argument. Valid arguments are: " + str(subsets_dict.keys()))
#         exit(1)
#     else:
#         try:
#             subset = subsets_dict[subset_arg]
#         except KeyError:
#             print("Invalid subset argument. Valid arguments are: " + str(subsets_dict.keys()))
#             exit(1)
#         input_set = subset
#         print(f'Using {subset_arg} instead of full prompt_set.')

model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left', trust_remote_code=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

model = pipe.model
pipe.tokenizer.pad_token_id = model.config.eos_token_id

sequences = pipe(
            input_set,
            max_new_tokens=10,
            do_sample=True,
            top_k=10,
            num_return_sequences=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
	        return_full_text=False,
	        batch_size=64
        )

# with open(gen_filename(subset_arg), 'w') as f:
#     for index, out in enumerate(sequences):
#         f.write('>>'+input_set[index])
#         for i in out:
#             f.write('\n>' + i['generated_text'])
#         f.write('\n\n')

for subset in subsets_dict.keys():
    print('running prompts for ' + subset)
    sequences = pipe(
            subsets_dict[subset],
            max_new_tokens=10,
            do_sample=True,
            top_k=10,
            num_return_sequences=10,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
	    return_full_text=False,
	    batch_size=32
        )

    with open(gen_filename(subset), 'w') as f:
        for index, out in tqdm(enumerate(sequences)):
            f.write('>>'+subsets_dict[subset][index])
            for i in out:
                f.write('\n>' + i['generated_text'])
            f.write('\n\n')
    

def run_with_dataset(input_set, model_name:str="tiiuae/falcon-7b", **kwargs):
    """
    Run prompts on a model, where the prompts are collected into a dataset (or subset of a dataset).

    :param input_set: Dataset of prompts to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param kwargs: Keyword arguments to pass to the pipeline function.
    :return: Output sequences from running the prompts through the model.
    """
    pipe = load_pipe(model_name)
    output_sequences = pipe(
        input_set,
        max_new_tokens = kwargs.get('max_new_tokens', 10),
        do_sample = kwargs.get('do_sample', True),
        top_k = kwargs.get('top_k', 10),
        num_return_sequences = kwargs.get('num_return_sequences', 10),
        eos_token_id = pipe.tokenizer.eos_token_id,
        pad_token_id = pipe.tokenizer.eos_token_id,
        return_full_text = kwargs.get('return_full_text', False),
        batch_size = kwargs.get('batch_size', 32),
    )


def load_pipe(model_name:str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = pipe.model
    pipe.tokenizer.pad_token_id = model.config.eos_token_id
    return pipe


def write_sequences_out(sequences, input_set, filename:str):
    """
    Write output sequences to a file.

    :param sequences: Output sequences
    :param input_set: Input prompts dataset corresponding to the output sequences
    :param filename: Name of file to write to
    """
    with open(filename, 'w') as f:
        for index, out in tqdm(enumerate(sequences)):
            f.write('>>'+input_set[index])
            for i in out:
                f.write('\n>' + i['generated_text'])
            f.write('\n\n')


run_with_dataset(prompt_set)
