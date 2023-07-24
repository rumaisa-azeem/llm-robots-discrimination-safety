from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
import transformers
import torch
from tqdm import tqdm
from inputs import prompt_set, subsets_dict
import sys

input_set = prompt_set

# if a subset argument is specified, use that subset instead of the full prompt set
if len(sys.argv) > 1:
    subset_arg = sys.argv[1]
    if not subset_arg.isidentifier():
        print("Invalid subset argument. Valid arguments are: " + str(subsets_dict))
        exit(1)
    else:
        try:
            subset = subsets_dict[subset_arg]
        except KeyError:
            print("Invalid subset argument. Valid arguments are: " + str(subsets_dict))
            exit(1)
        input_set = subset
        print(f'Using {subset_arg} instead of full prompt_set.')

model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left', trust_remote_code=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
pipeline.tokenizer.pad_token_id = model.config.eos_token_id

sequences = pipeline(
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

for index, out in enumerate(sequences):
    print('\n'+input_set[index])
    for i in tqdm(out):
        print(i['generated_text'])
