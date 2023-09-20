import transformers
import torch
import json
from datasets import load_dataset
from tqdm import tqdm

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")

def parse_response(response):
    if len(response)==1:
        return choices.index(response[0]) + 1 if response[0] in choices else None
    elif response[0] in choices and not response[1].isalpha():
        return choices.index(response[0]) + 1
    else:
        return None

def create_examples(ds, template):
    ex=[]
    for d in ds:
        prompt=template.format(**d,correct_answer=choices[int(d["correct_answer_num"])-1])
        ex.append(prompt)

    return "\n\n".join(ex)

prompt_template={}

prompt_template["deu_Latn"]="""{flores_passage}
Frage: {question}
Antwort A: {mc_answer1}
Antwort B: {mc_answer2}
Antwort C: {mc_answer3}
Antwort D: {mc_answer4}
Richtige Antwort:"""

prompt_template["eng_Latn"]="""{flores_passage}
Question: {question}
Answer A: {mc_answer1}
Answer B: {mc_answer2}
Answer C: {mc_answer3}
Answer D: {mc_answer4}
Correct answer:"""

choices=["A","B","C","D"]

if __name__ == '__main__':
    model_path="/home/g/models/llama-2-7b"
    language="eng_Latn"     # deu_Latn eng_Latn

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset_conf={"path": "facebook/belebele", "name": language, "split": "test"}
    ds = load_dataset(**dataset_conf)
    ds_examples, ds_prompts=ds.select(range(0,5)), ds.select(range(5,len(ds)))

    prompt_examples = create_examples(ds_examples, prompt_template[dataset_conf["name"]] + " {correct_answer}")

    gen_config = {
        "temperature": 0.7,
        "top_p": 0.1,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "do_sample": True,
        "max_new_tokens": 5,
    }

    result={
        "dataset": dataset_conf,
        "total": 0, 
        "correct": 0,
        "correct_percent": None,
        "prompt_template": prompt_template[dataset_conf["name"]],
        "examples": prompt_examples,
        "questions": [],
    }

    for rowNo, row in enumerate(tqdm(ds_prompts)):        
        prompt=result["prompt_template"].format(**row)        
        prompt_full=result["examples"] + "\n\n" + prompt

        out=pipeline(prompt_full, batch_size=1, **gen_config)[0]
        response=out["generated_text"][len(prompt_full)+1:].split("\n\n")[0]
        response_parsed=parse_response(response.strip())
        response_correct=response_parsed==int(row["correct_answer_num"])

        result["questions"].append({
            "question": prompt,
            "correct_answer": int(row["correct_answer_num"]),
            "answer_raw": response,
            "answer": response_parsed,
            "correct": response_correct
        })

        print(f"Question {rowNo+1}/{len(ds_prompts)}: {'correct' if response_correct else 'wrong'}")

        if response_correct:
            result["correct"]+=1
        result["total"]+=1

        if response_parsed is None:
            print(f"Could not parse {response}")

        result["correct_percent"]=result["correct"]/(rowNo+1)

        print("So far {}/{} correct answers ({}%)".format(result["correct"],rowNo+1, round(result["correct"]/(rowNo+1)*100,2)))

    write_pretty_json("belebe-{}_{}.json".format(model_path.split("/")[-1],language), result)

