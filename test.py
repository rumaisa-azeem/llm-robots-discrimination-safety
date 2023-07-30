# Test the model on a given prompt using the test() function. 
# Use with python -i in the command line to avoid needing to load the model repeatedly.

from transformers import AutoTokenizer, pipeline
import sys
from tqdm import tqdm
import torch

# can run with prompt as command line argument, or enter prompt when prompted
if not (len(sys.argv) == 2):
    prompt = input('enter prompt: ')
else:
    prompt = sys.argv[1]
        
model_name = "tiiuae/falcon-7b"
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

def test(prompt):
    '''
    Run the model on the given prompt and print the generated text.
    '''
    sequences = pipe(
            prompt,
            max_new_tokens=10,
            do_sample=True,
            top_k=10,
            num_return_sequences=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
	    return_full_text=False,
	    batch_size=64
        )
    print('>> ' + prompt)
    for seq in sequences:
        print('> ' + seq['generated_text'])


test(prompt)
