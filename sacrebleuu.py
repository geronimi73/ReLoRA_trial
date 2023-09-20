import transformers
import evaluate
import torch
import json
from tqdm import tqdm
from datasets import load_dataset

def load_model(path):
	pipeline = transformers.pipeline(
		"text-generation",
		model=path,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)

	return pipeline

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")


model_path="/home/g/models/llama-2-7b"
data_set={
	"path": "wmt20_mlqe_task1",
	"name": "en-de",
	"split": "test",
}

pipeline = load_model(model_path)

# needed for batching, from "tips" at https://huggingface.co/docs/transformers/model_doc/llama2
pipeline.tokenizer.add_special_tokens({"pad_token":"<pad>"})
pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))
pipeline.model.config.pad_token_id = pipeline.tokenizer.pad_token_id

ds = load_dataset(**data_set)
ds_examples=ds[0:5]		# use first 5 to generate examples for 5-shot translation prompt
ds_predict=ds[5:]

prompt_template="English: {en}\nGerman: {de}"
prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples["translation"]])

prompts=[ (prompt_examples+"\n\n"+prompt_template).format(en=d["en"],de="")[:-1] for d in ds_predict["translation"] ] 
prompts_generator=(p for p in prompts)	# pipeline needs a generator, not a list

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
	"do_sample": True,
	"num_return_sequences": 1,
	"eos_token_id": pipeline.tokenizer.eos_token_id,
	"max_new_tokens": 100,		# sentences are short, this should be more than enough
}

results={
	"model": model_path,
	"num_translations": 0,
	"sacrebleu_score": None,
	"translations": [],
}

for i, out in enumerate(tqdm(pipeline(prompts_generator, batch_size=24, **gen_config),total=len(prompts))):
	prediction=out[0]["generated_text"][len(prompts[i])+1:].split("\n\n")[0]
	reference=ds_predict["translation"][i]["de"]	# ! change for new language

	results["translations"].append({"prediction": prediction, "reference":reference})
	results["num_translations"]+=1

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results=sacrebleu.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]])
results["sacrebleu_score"]=sacrebleu_results["score"]

write_pretty_json("sacrebleu-"+model_path.split("/")[-1]+".json",results)


