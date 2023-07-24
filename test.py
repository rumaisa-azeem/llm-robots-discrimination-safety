from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
from tqdm import tqdm
import torch

if not (len(sys.argv == 2)):
    prompt = input('enter prompt: ')
elif len(sys.argv) == 2:
    prompt = sys.argv[1]
        
model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained(model)
pipe.tokenizer.pad_token_id = model.config.eos_token_id

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

for index, out in enumerate(sequences):
    print('\n'+prompt_set[index])
    for i in tqdm(out):
        print(i['generated_text'])
	
