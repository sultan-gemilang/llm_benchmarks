from transformers import AutoModelForCausalLM, LlamaTokenizer, GPT2Tokenizer, GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm

import torch

device = 'gpu'
model_id = 'baffo32/decapoda-research-llama-7B-hf'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    offload_folder='offload',
    torch_dtype='auto'
)

if "opt" in model_id.lower():
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    max_length = model.config.max_position_embeddings    
elif "gpt" in model_id.lower():
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    max_length = model.config.n_positions
elif "llama" in model_id.lower():
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    max_length = model.config.max_length
else:
    raise Exception("Model not supported!")

test = load_dataset('ptb_text_only', 'penn_treebank', split='test')
test_enc = tokenizer(" ".join(test['sentence']), return_tensors='pt')

seq_len = test_enc.input_ids.size(1)
# model.to(device)

nlls = []
prev_end_loc = 0
stride = 512

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = test_enc.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())

print(f"{model_id} ppl\t-> {round(ppl.item(), 3)}")