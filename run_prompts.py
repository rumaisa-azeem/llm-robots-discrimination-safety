from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
from create_prompt_set import prompt_set
import os

def run(input_set, output_filename:str, model_name:str="tiiuae/falcon-7b", **kwargs):
    """
    Run prompts on a model, where the prompts are collected into a dataset (or subset of a dataset).

    :param input_set: Dataset of prompts to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param kwargs: Keyword arguments to pass to the pipeline function.
    :return: Output sequences from running the prompts through the model.
    """
    pipe = load_pipeline(model_name)
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
    write_sequences_out(output_sequences, input_set, output_filename)


def test(prompt:str, model_name:str="tiiuae/falcon-7b", **kwargs):
    """
    Run a single prompt on a model and print the output.

    :param prompt: Prompt to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param kwargs: Keyword arguments to pass to the pipeline function.
    """
    pipe = load_pipeline(model_name)
    output_sequences = pipe(
        prompt,
        max_new_tokens = kwargs.get('max_new_tokens', 10),
        do_sample = kwargs.get('do_sample', True),
        top_k = kwargs.get('top_k', 10),
        num_return_sequences = kwargs.get('num_return_sequences', 10),
        eos_token_id = pipe.tokenizer.eos_token_id,
        pad_token_id = pipe.tokenizer.eos_token_id,
        return_full_text = kwargs.get('return_full_text', False),
        batch_size = kwargs.get('batch_size', 32),
    )
    
    print('>> ' + prompt)
    for seq in output_sequences:
        print('> ' + seq['generated_text'])

def load_pipeline(model_name:str):
    """
    Load a model and tokenizer into a pipeline.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM.
    :return: Pipeline object for the specified model.
    """
    print('Loading model: ' + model_name)
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
    print('Writing to file: ' + filename)
    with open(filename, 'w') as f:
        for index, out in tqdm(enumerate(sequences)):
            f.write('>>'+input_set[index])
            for i in out:
                f.write('\n>' + i['generated_text'])
            f.write('\n\n')


def gen_filename(prefix:str=''):
    """
    Generate a filename for the output file.
    :param prefix: Prefix to use for the filename. Defaults to empty string, so filename will be a number.
    :return: Filename to use for the output file.
    """
    output_dir = 'outputs'
    next_filename = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    listdir = os.listdir(output_dir)
    filenames = []
    for filename in listdir: # Check for existing files with same type
        if prefix in filename:
            filenames.append(filename)
    if not filenames:
        # If no files exist, start with 1
        next_filename = os.path.join(output_dir, f'{prefix}1.txt')
    else:
        # Find the highest number used in the filenames
        max_number = max([int(os.path.splitext(filename)[0][-1]) for filename in filenames])
        next_number = max_number + 1
        next_filename = os.path.join(output_dir, f'{prefix}{next_number}.txt')

    return next_filename


subsets_dict = prompt_set.get_subsets_dict()
for subset_name in subsets_dict:
    run(subsets_dict[subset_name], gen_filename(subset_name))
