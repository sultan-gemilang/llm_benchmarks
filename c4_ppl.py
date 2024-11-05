from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import torch

device = 'cuda'
model_id = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

test = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
test_enc = tokenizer(' '.join(test[:1100]['text']), return_tensors='pt')


if "opt" in model_id.lower():
    max_length = model.config.max_position_embeddings    
elif "gpt" in model_id.lower():
    max_length = model.config.n_positions
elif "llama" in model_id.lower():
    max_length = model.config.max_length
else:
    raise Exception("Model not supported!")

seq_len = test_enc.input_ids.size(1)
model.to(device)

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