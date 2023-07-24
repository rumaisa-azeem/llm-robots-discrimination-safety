from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
import transformers
import torch
from inputs import prompt_set
from tqdm import tqdm

model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained(model)
pipeline.tokenizer.pad_token_id = model.config.eos_token_id

sequences = pipeline(
            prompt_set,
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
	
