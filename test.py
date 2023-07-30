from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
from tqdm import tqdm
import torch

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

def run(prompt):
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


run(prompt)
