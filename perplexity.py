import torch
import json
import gc
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")

def load_model(path):
    tokenizer = LlamaTokenizer.from_pretrained(path)

    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    model = AutoModelForCausalLM.from_pretrained(
      path,
      load_in_4bit=True,
      device_map="auto",
      max_memory=max_memory,
      torch_dtype=torch.bfloat16
    )

    return model, tokenizer

def load_ds(name):
    if name=="wikitext":
        return load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    elif name in ["wikipedia_de_unseen", "wikipedia_de_seen"]:
        dataset = load_dataset("json", data_files="wikipedia_20220301_de.json")

        # rows 1:10^6 = set of wikipedia_de tokens ReLoRA model was trained on (="seen")
        # rows 10^6:end = set of wikipedia_de tokens ReLoRA model was NOT trained on (="unseen")
        ds_select=range(1_000_000,len(dataset["train"])) if name=="wikipedia_de_unseen" else range(0,1_000_000)
        dataset["train"] = dataset["train"].select(ds_select).shuffle(seed=42)
        
        return dataset["train"].select(range(500))  # select ~500 rows, enough to get 350k tokens
    else:
        print(f"unknown dataset {name}")
        return None

def perplexity(model, tokenizer, ds):
    # code from https://huggingface.co/docs/transformers/perplexity and https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2
    text="\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")

    max_length = 4096
    stride = 512

    # limit to max 350k tokens (=wikitext size)
    seq_len = min(encodings.input_ids.size(1),350_000)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
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

    return ppl, seq_len

models=[   
    "/home/g/models/llama-2-7b-relora-final", 
    "/home/g/models/llama-2-7b", 
    "/home/g/models/llama-2-13b"
    ]
data_sets=[
    "wikitext", 
    "wikipedia_de_unseen", 
    "wikipedia_de_seen"
    ]

results=[]
for model_name in models:
    model, tokenizer = load_model(model_name)

    for dataset_name in data_sets:
        print(f"=== model {model_name}: dataset {dataset_name}")

        ds = load_ds(dataset_name)
        ppl, token_count=perplexity(model, tokenizer, ds)

        results.append({
            "model": model_name,
            "model_quant": {
                "4bit": model.config.quantization_config.load_in_4bit,
                "8bit": model.config.quantization_config.load_in_8bit
            },
            "dataset": dataset_name,
            "token_count": token_count,
            "ppl": ppl.item(),
        })
        del ds

    model = tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

write_pretty_json("perplexity_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json", results)

