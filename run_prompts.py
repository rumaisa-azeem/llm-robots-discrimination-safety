'''
Module containing methods to run prompts on a model.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, pipeline
from tqdm import tqdm
from csv import writer
import os, torch

def run_for_seqs(input_set, output_filename:str, model_name:str="tiiuae/falcon-7b", **kwargs):
    """
    Run prompts through a model, where sequences are generated for each prompt and written to a file.

    :param input_set: Dataset of prompts to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param kwargs: Keyword arguments to pass to the pipeline function.
    :return: Output sequences from running the prompts through the model.
    """
    output_sequences = use_pipeline(input_set, model_name, **kwargs)
    write_sequences_out(output_sequences, input_set, output_filename)


def test_for_seqs(prompt:str, model_name:str="tiiuae/falcon-7b", **kwargs):
    """
    Run a single prompt on a model and print the output sequences.

    :param prompt: Prompt to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param kwargs: Keyword arguments to pass to the pipeline function.
    """
    output_sequences = use_pipeline(prompt, model_name, **kwargs)
    print('>> ' + prompt)
    for seq in output_sequences:
        print('> ' + seq['generated_text'])


def run_for_scores(prompt_set, filename:str, model_name:str='tiiuae/falcon-7b', top_n:int=10):
    """
    Run prompts through a model, where the top_n most likely next tokens and their probabilities are generated for each prompt, 
    then written to a csv file.

    :param prompt_set: Dataset of prompts to run through the model.
    :param filename: Name of the file to write the scores to.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param top_n: The number of words to get the highest probabilities for. Defaults to 10.
    """
    model, tokenizer = load_model(model_name)

    scores_dict = {}
    for prompt in tqdm(prompt_set):
        scores = get_scores_for_prompt(prompt, model, tokenizer, top_n)
        scores_dict[prompt] = scores
    write_scores_out(scores_dict, filename)


def test_for_scores(prompt:str, model_name:str="tiiuae/falcon-7b", top_n:int=10):
    """
    Run a single prompt through a model and print the top_n most likely next tokens and their probabilities.

    :param prompt: Prompt to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param top_n: The number of words to get the highest probabilities for. Defaults to 10.
    """
    model, tokenizer = load_model(model_name)

    scores = get_scores_for_prompt(prompt, model, tokenizer, top_n)
    print('>> ' + prompt)
    for word, prob in scores:
        print(f'{word}: {prob}')


# helper functions -----------------------------------------------------------------------------------------------------


def get_scores_for_prompt(prompt, model, tokenizer, top_n:int=10):
    '''
    For a given prompt, return the top_n most likely next tokens and their probabilities.
    
    :param prompt: The input sequence to get the most likely next tokens for.
    :param model: The model to use.
    :param tokenizer: The tokenizer to use (should match the model).
    :param top_n: The number of words to get the highest probabilities for. Defaults to 10.
    :return: A list of tuples of the form (word, probability) for the top_n most likely next tokens.
    '''
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model(input_ids, return_dict=True, use_cache=True)
    next_token_logits = output.logits[:, -1, :] # get logits for the next token in the sequence
    probs = torch.softmax(next_token_logits, dim=-1).tolist()[0] # convert logits to list of probabilities

    # convert vocab dict to list where index is the id of the word
    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None for i in range(len(vocab_dict))]
    for key, val in vocab_dict.items():
        vocab_list[int(val)] = tokenizer.decode(val)
    # create a list of (word, probability) pairs for the whole vocab
    word_probs = []
    for index, prob in enumerate(probs): 
        word_probs.append((vocab_list[index], prob))

    # get top n most probable words
    sorted_probs = sorted(word_probs, reverse=True, key=lambda x: x[1])[:top_n]
    return sorted_probs[:top_n]


def use_pipeline(inp, model_name:str, **kwargs):
    """
    Use a pipeline to run a prompt/prompts through a model, returning the output sequences.

    :param inp: Input for the pipeline.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM.
    :param kwargs: Keyword arguments to pass to the pipeline function.
    :return: Pipeline object for the specified model.
    """
    print('Loading model: ' + model_name)
    if model_name == 'openlm-research/open_llama_7b':
        tokenizer = tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
    else:
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

    return pipe(
        inp,
        max_new_tokens = kwargs.get('max_new_tokens', 10),
        do_sample = kwargs.get('do_sample', True),
        top_k = kwargs.get('top_k', 10),
        num_return_sequences = kwargs.get('num_return_sequences', 10),
        eos_token_id = pipe.tokenizer.eos_token_id,
        pad_token_id = pipe.tokenizer.eos_token_id,
        return_full_text = kwargs.get('return_full_text', False),
        batch_size = kwargs.get('batch_size', 32),
    )


def write_sequences_out(output_sequences, input_set, filename:str):
    """
    Write output sequences to a csv file.

    Format: prompt, output_categories, sequence1, sequence2, ... sequencen

    :param output_sequences: Output sequences to write to file.
    :param input_set: PromptSet used to generate the output sequences.
    :param filename: Name of the file to write to.
    """
    print(f'Writing to file: {filename}.csv')
    with open(filename+'.csv', 'w', newline='') as f:
        w = writer(f)
        is_title_row = True
        for index, sequences_for_prompt in tqdm(enumerate(output_sequences)):
            prompt = input_set[index]
            output_categories = input_set.get_expected_outputs(prompt)
            row = [prompt, output_categories]
    
            if is_title_row:
                title_row = ['prompt', 'output_categories']
                for index, seq in enumerate(sequences_for_prompt):
                    title_row.append('sequence'+str(index+1))
                    row.append(seq['generated_text'])
                w.writerow(title_row)
                w.writerow(row)
                is_title_row = False
            else:
                for seq in sequences_for_prompt:
                    row.append(seq['generated_text'])
                w.writerow(row)


def write_scores_out(scores_dict, filename:str):
    """
    Take a dictionary of prompts and their top_n most likely tokens with probabilities, and write to a csv file.
    
    Format: prompt, word1, probability1, word2, probability2, ..., wordn, probabilityn

    :param scores_dict: Dictionary of prompts and their top_n most likely tokens with probabilities
    :param filename: Name of file to write to
    """
    print(f'Writing to file: {filename}.csv')
    top_n = len(list(scores_dict.values())[0])
    title_row = ['prompt']
    for n in range(top_n):
        title_row.append('word' + str(n+1))
        title_row.append('probability' + str(n+1))

    with open(filename+'.csv', 'w', newline='') as f:
        w = writer(f)
        w.writerow(title_row)
        for prompt, scores in scores_dict.items():
            row = [prompt]
            for word, prob in scores:
                row.append(word)
                row.append(prob)
            w.writerow(row)


def gen_filename(prefix:str='', output_dir:str='outputs'):
    """
    Generate a filename for the output file (doesn't include file extension). 
    If prefix is specified, filename will start with that prefix and end with a number (e.g. 'outputs/prefix1' if no other files with that prefix exist)
    Otherwise filename will be a number (e.g. 'outputs/1' if no other files exist, or 'outputs/2' if 'outputs/1' already exists, etc)

    :param prefix: Prefix to use for the filename. (Optional)
    :return: Filename to use for the output file.
    """
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
        next_filename = os.path.join(output_dir, f'{prefix}1')
    else:
        # Find the highest number used in the filenames
        max_number = max([int(os.path.splitext(filename)[0][-1]) for filename in filenames])
        next_number = max_number + 1
        next_filename = os.path.join(output_dir, f'{prefix}{next_number}')

    return next_filename


def load_model(model_name:str, **kwargs):
    """
    Load a model and matching tokenizer from HuggingFace.

    :param model_name: Name of the model to load. Must be able to load with AutoModelForCausalLM.
    :return: model and tokenizer objects.
    """
    print('Loading model: ' + model_name)
    if model_name == 'openlm-research/open_llama_7b':
        tokenizer = tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left', use_cache=True, **kwargs)
        model = LlamaForCausalLM.from_pretrained(model_name, use_cache=True, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True, use_cache=True, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=True, **kwargs)
    return model, tokenizer