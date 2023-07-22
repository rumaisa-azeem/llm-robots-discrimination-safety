from transformers import AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import transformers
import torch
from inputs import prompt_set

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
            KeyDataset(prompt_set),
            max_new_tokens=10,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
	        return_full_text=False
        )