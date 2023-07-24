from transformers import AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import transformers
import torch
from inputs import prompt_set, subsets
import sys

if len(sys.argv) > 1:
    subset_arg = sys.argv[1]
    for subset in subsets:
        check_arg = False
        if subset_arg.isidentifier(subset):
            check_arg = True
    if not check_arg:
        print("Invalid subset argument. Valid arguments are: " + str(subsets))
        exit(1)

        
model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

sequences = pipeline(
            prompt_set,
            max_new_tokens=10,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
	        return_full_text=False
        )

for seq in sequences:
    for i in seq:
        print(i['generated_text'])
