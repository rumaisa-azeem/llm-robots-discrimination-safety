from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b', trust_remote_code=True, use_cache=True)
vocab_dict = tokenizer.get_vocab()
vocab_list = [None for i in range(len(vocab_dict))]

for key, val in vocab_dict.items():
    vocab_list[int(val)] = tokenizer.decode(val)


#print(vocab_list)
