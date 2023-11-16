'''
Module containing methods to run prompts on a model.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, pipeline
from tqdm import tqdm
from csv import writer
import os, torch, pandas as pd

def run_for_seqs(prompt_set, model_name:str, filename:str, output_dir:str, **kwargs):
    """
    Run prompts through a model, where sequences are generated for each prompt and written to a file.

    :param prompt_set: Dataset of prompts to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param kwargs: Keyword arguments to pass to the pipeline function.
    :return: Output sequences from running the prompts through the model.
    """
    output_sequences = use_pipeline(prompt_set, model_name, **kwargs)
    write_sequences_out(output_sequences, prompt_set, gen_filename(filename, output_dir))


def run_for_scores(prompt_set, model_name:str, filename:str, output_dir:str, top_n:int=10):
    """
    Run prompts through a model, where the top_n most likely next tokens and their probabilities are generated for each prompt, 
    then written to a csv file.

    :param prompt_set: Dataset of prompts to run through the model.
    :param filename: Name of the file to write the scores to.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param top_n: The number of words to get the highest probabilities for. Defaults to 10.
    :param selected_outputs: (Optional) A specific list of tokens to get confidence scores for. 
    """
    model, tokenizer = load_model(model_name)

    scores_dict = {}
    for prompt in tqdm(prompt_set):
        expected_outputs = prompt_set.get_expected_outputs(prompt)
        scores = get_scores_for_prompt(prompt, model, tokenizer, expected_outputs)
        if not expected_outputs:
            scores = scores[:top_n]
        scores_dict[prompt] = scores    
    write_scores_out(scores_dict, prompt_set, filename, output_dir)


# testing functions ----------------------------------------------------------------------------------------------------------


def test_for_seqs(prompt:str, model_name:str, **kwargs):
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


def test_top_n_scores(prompt:str, model_name:str, model=None, tokenizer=None, top_n:int=10):
    """
    Run a single prompt through a model and print the top_n most likely next tokens and their probabilities.

    Either enter a model_name to load a model for the test, or pass in a preloaded model AND tokenizer (saves time when running multiple tests).

    :param prompt: Prompt to run through the model.
    :param model_name: Name of model to use, must be able to load with AutoModelForCausalLM. Defaults to Falcon-7B.
    :param model: Optional - a preloaded model object to test the prompt on. Must also include tokenizer. 
    :param tokenizer: Optional - a preloaded tokenizer to test the prompt with. Must also include model.
    :param top_n: The number of words to get the highest probabilities for. Defaults to 10.
    """
    if not model or not tokenizer:
        model, tokenizer = load_model(model_name)

    print('>> ' + prompt)
    for word, prob in get_scores_for_prompt(prompt, model, tokenizer)[:top_n]:
        print(f'{word}: {prob}')


# helper functions -----------------------------------------------------------------------------------------------------


def get_scores_for_prompt(prompt, model, tokenizer, selected_outputs:list=None):
    '''
    For a given prompt, return next possible tokens and their probabilities in order from highest to lowest. 
    Returns (token, probability) pairs for all tokens in the model's vocbulary.
    
    :param prompt: The input sequence to get the most likely next tokens for.
    :param model: The model to use.
    :param tokenizer: The tokenizer to use (should match the model).
    :param selected_outputs: A specific list of tokens to get the probabilities for. Defaults to None.
    :return: A list of tuples of the form (token, probability) sorted from highest to lowest probability. If selected_outputs is specified, the list will only contain tuples for those words. 
    '''
    token_probs = get_full_vocab_scores_for_prompt(prompt, model, tokenizer)
    if selected_outputs: # get probabilities for specific words
        selected_word_probs = [(word, get_full_word_score(prompt, word, model, tokenizer)) for word in selected_outputs]
        return sorted(selected_word_probs, reverse=True, key=lambda x: x[1])
    else: # get top n most probable words
        return sorted(token_probs, reverse=True, key=lambda x: x[1])


def load_model(model_name:str, **kwargs):
    """
    Load a model and matching tokenizer from HuggingFace.

    :param model_name: Name of the model to load. Must be able to load with AutoModelForCausalLM.
    :param **kwargs: Keyword arguments to pass in when loading the model and tokenizer.
    :return: model and tokenizer objects.
    """
    print('Loading model: ' + model_name)
    if model_name == 'openlm-research/open_llama_7b':
        tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left', use_cache=True, **kwargs)
        model = LlamaForCausalLM.from_pretrained(model_name, use_cache=True, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True, use_cache=True, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=True, **kwargs)
    return model, tokenizer


def use_pipeline(inp, model_name:str, **kwargs):
    """
    Load and use a pipeline to run a prompt/prompts through a model, returning the output sequences.

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
        is_first_row = True
        for index, sequences_for_prompt in tqdm(enumerate(output_sequences)):
            prompt = input_set[index]
            row = [prompt, input_set.get_base_prompt(prompt), input_set.get_dimension(prompt), input_set.get_expected_outputs(prompt)]
    
            if is_first_row: # add title row then first row of sequences
                title_row = ['prompt', 'base_prompt', 'dimension', 'output_categories']
                for index, seq in enumerate(sequences_for_prompt):
                    title_row.append('sequence'+str(index+1))
                    row.append(seq['generated_text'])
                w.writerow(title_row)
                w.writerow(row)
                is_first_row = False
            else:
                for seq in sequences_for_prompt:
                    row.append(seq['generated_text'])
                w.writerow(row)


def write_scores_out(scores_dict, input_set, filename:str, output_dir:str):
    """
    Take a dictionary of prompts and their most likely tokens with corresponding confidence scores, and write to a csv file.
    
    Format: prompt, base_prompt, word1, probability1, word2, probability2, ..., wordn, probabilityn

    :param scores_dict: Dictionary where each key is a prompt, and the corresponding value is a list of tuples, starting with that prompt's base_prompt, then each possible word and its corresponding confidence score
    :param input_set: PromptSet used to generate the scores.
    :param filename: Name of file to write to
    """
    categorisation_prompts = []
    comparison_prompts = []
    generation_prompts = []
    num_categorisation_cols = 0

    # group prompts by type
    for prompt, scores in scores_dict.items():
        if input_set.get_prompt_type(prompt) == 'categorisation':
            categorisation_prompts.append((prompt, scores))
            num_categorisation_cols = max(num_categorisation_cols, len(scores))
        elif input_set.get_prompt_type(prompt) == 'comparison':
            comparison_prompts.append((prompt, scores))
        elif input_set.get_prompt_type(prompt) == 'generation':
            generation_prompts.append((prompt, scores))
    
    common_cols = ['prompt', 'base_prompt', 'dimension']

    if (categorisation_prompts):
        data_for_frame = create_input_list_for_frame(categorisation_prompts, input_set, num_categorisation_cols)
        col_names = create_col_names_list(common_cols, num_categorisation_cols)
        categorisation_frame = pd.DataFrame(data_for_frame, columns=col_names)
        categorisation_filename = gen_filename(filename+'_categorisation', output_dir)
        print(f"> Writing to file: {categorisation_filename}")
        categorisation_frame.to_csv(categorisation_filename, index=False)

    if (comparison_prompts):
        data_for_frame = create_input_list_for_frame(comparison_prompts, input_set, 2)
        col_names = create_col_names_list(common_cols, 2)
        comparison_frame = pd.DataFrame(data_for_frame, columns=col_names)
        comparison_filename = gen_filename(filename+'_comparison', output_dir)
        print(f"> Writing to file: {comparison_filename}")
        comparison_frame.to_csv(comparison_filename, index=False)

    if (generation_prompts):
        num_generation_cols = len(generation_prompts[0][1])
        data_for_frame = create_input_list_for_frame(generation_prompts, input_set, num_generation_cols)
        col_names = create_col_names_list(common_cols, num_generation_cols)
        generation_frame = pd.DataFrame(data_for_frame, columns=col_names)
        generation_filename = gen_filename(filename+'_generation', output_dir)
        print(f"> Writing to file: {generation_filename}")
        generation_frame.to_csv(generation_filename, index=False)


def gen_filename(prefix:str='', output_dir:str='outputs'):
    """
    Generate a filename for the output file (with .csv file extension). 
    
    If prefix is specified, filename will start with that prefix and end with a number (e.g. 'outputs/prefix1' if no other files with that prefix exist)
    Otherwise filename will be a number (e.g. 'outputs/1' if no other files exist, or 'outputs/2' if 'outputs/1' already exists, etc)

    :param prefix: Prefix to use for the filename. (Optional)
    :param output_dir: Directory to save the output file in. Defaults to 'outputs'. (Optional)
    :return: Filename to use for the output file, including the directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    listdir = os.listdir(output_dir)
    filenames = []
    for filename in listdir: # Check for existing files with same type
        if prefix in filename:
            filenames.append(filename)
    if not filenames: # If no files exist, start with 1
        next_filename = os.path.join(output_dir, f'{prefix}01')  
    else:  # Find the highest number used in the filenames
        next_num = 1 + max([int(os.path.splitext(filename)[0][-1]) for filename in filenames])
        next_filename = os.path.join(output_dir, f'{prefix}{"%02d" % next_num}')
    return next_filename+'.csv'

# helper functions for the helper functions --------------------------------------------------------------------------------

def add_scores_to_row(row, scores, num_cols):
    """Add the scores to the end of the row in wordx, probabilityx pairs and fill the rest of the row with None"""
    for word, prob in scores:
        row.append(word)
        row.append(prob)
    while (len(row)-3)/2 < num_cols:
        row.append(None)
        row.append(None)

def create_input_list_for_frame(list_of_prompts_and_scores, input_set, num_cols):
    data_for_frame = []
    for prompt, scores in list_of_prompts_and_scores:
        row = [prompt, input_set.get_base_prompt(prompt), input_set.get_dimension(prompt)]
        add_scores_to_row(row, scores, num_cols)
        data_for_frame.append(row)
    return data_for_frame

def create_col_names_list(common_cols, num_word_prob_pairs):
    col_names = common_cols.copy()
    for i in range(num_word_prob_pairs):
        col_names.append('word'+str(i+1))
        col_names.append('probability'+str(i+1))
    return col_names

# def map_words_to_tokens(input_words:list, tokenizer):
#     """
#     :param input_words: a list of words to map to their first token
#     :param tokenizer: the tokenizer to use for tokenizing the words
#     :return: a dict: {word: list of tokens the word is tokenized into} for all the words in input_words
#     """
#     tokens_map = {}
#     for word in input_words:
#         tokens_map[word] = [tokenizer.decode(token) for token in tokenizer.encode(word)]
#     return tokens_map

# def get_words_with_same_tokens(token_map:dict):
#     """
#     Check if any words in token_map map to the same first token

#     Structure of token map:
#     - key: a word in the list returned by prompt_set.get_expected_outputs(prompt)
#     - value: the list of tokens that the word is broken down into by the tokenizer

#     :return: a tuple of the two words with duplicate tokens
#     """
#     seen_tokens_map = {}
#     duplicate_words = set()
#     for word, tokens_list in token_map.items():
#         token = tokens_list[1]
#         if token in seen_tokens_map.keys():
#             duplicate_words.add(word)
#             duplicate_words.add(seen_tokens_map[token])
#         else:
#             seen_tokens_map[token] = word
#     return duplicate_words


def get_full_word_score(prompt:str, word:str, model, tokenizer):
    tokens = [tokenizer.decode(token) for token in tokenizer.encode(word)]
    i=1
    score = 1
    while i<len(tokens):
        score *= get_score_for_prompt_and_token(prompt, tokens[i], model, tokenizer)
        prompt = prompt + " " + tokens[i] if i==1 else prompt + tokens[i]
        i+=1
    return score

def get_score_for_prompt_and_token(prompt, target_token, model, tokenizer):
    """
    Get the probability of a particular token being output for a given prompt.

    :param prompt: The prompt to run through the model and get scores for
    :param token: The specific token to get a score for 
    """
    token_probs = get_full_vocab_scores_for_prompt(prompt, model, tokenizer)
    return sorted([prob for token, prob in token_probs if token==target_token])[0]

def get_full_vocab_scores_for_prompt(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model(input_ids, return_dict=True, use_cache=True)
    next_token_logits = output.logits[:, -1, :] # get logits for the next token in the sequence
    probs = torch.softmax(next_token_logits, dim=-1).tolist()[0] # convert logits to list of probabilities
    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None for i in range(len(vocab_dict))] # convert vocab dict to list where index is the id of the word
    for key, val in vocab_dict.items():
        vocab_list[int(val)] = tokenizer.decode(val)
    return [(vocab_list[i].strip(), prob) for i, prob in enumerate(probs)]