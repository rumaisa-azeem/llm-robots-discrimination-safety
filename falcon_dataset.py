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

# sequences = pipe(
#             input_set,
#             max_new_tokens=10,
#             do_sample=True,
#             top_k=10,
#             num_return_sequences=5,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
# 	        return_full_text=False,
# 	        batch_size=64
#         )

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
