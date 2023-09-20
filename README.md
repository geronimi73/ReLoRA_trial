# ReLoRA_trial

## Motivation

Llama2 is pretty bad at german

Llama2 pretraining (source: https://arxiv.org/pdf/2307.09288.pdf):

- Pretrained on 2 trillion tokens
- 89.70% english = ~1.8T tokens
- 0.17% german = ~3.4B tokens

-> pretrain another 1B german tokens into llama2-7b and see what happens

## Strategy

- use ReLoRA: https://github.com/Guitaricet/relora
- use axolotl to train, PR https://github.com/OpenAccess-AI-Collective/axolotl/pull/322
- aim for 10 restarts as mentioned in the PR
- same LR as with QLoRA for llama2 7b model: 0.0002 
- dataset: german wikipedia

## Setup

```bash
apache-beam==2.49.0
bitsandbytes==0.41.1
datasets==2.13.0
dill==0.3.1.1
evaluate==0.4.0
sentencepiece==0.1.98
transformers @ git+https://github.com/huggingface/transformers.git@fe3c8ab1af558b95f67f5fafc0c55f09fd2b09db
wandb==0.15.4
```

## Datasets

```python
from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.de")

dataset["train"] = dataset["train"].select(range(1000000))
dataset["train"].to_json("wikipedia_20220301_de_1M.json");
```

= 1 million rows

= 1.13B (german) tokens to train on

## ReLoRA (axolotl)

### version

```
axolotl commit fd55bc8
```

### config

relora.yml

```yaml
base_model: /home/g/models/llama-2-7b
base_model_config: /home/g/models/llama-2-7b
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer
is_llama_derived_model: true

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: ../datasets/wikipedia_20220301_de_1M.json
    ds_type: json
    type: completion
dataset_prepared_path: last_run_prepared
val_set_size: 0
output_dir: ./relora-out-2023-08-31_3

adapter: qlora
lora_model_dir:

sequence_len: 4096
sample_packing: false

lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:

relora_steps: 3000
relora_warmup_steps: 30
relora_cpu_offload: false

wandb_project: relora
wandb_entity:
wandb_watch:
wandb_run_id:
wandb_log_model:

gradient_accumulation_steps: 4
micro_batch_size: 4
eval_batch_size: 4
num_epochs: 1
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
eval_steps: 20
save_steps: 1000
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
```

### run

```
accelerate launch scripts/finetune.py relora.yml
```

### wandb

#### loss
<img src="https://github.com/geronimi73/ReLoRA_trial/blob/main/img/wandb_loss.png?raw=True" width="50%" >

#### LR
<img src="https://github.com/geronimi73/ReLoRA_trial/blob/main/img/wandb_lr-restarts.png?raw=True" width="50%" >

### Performance of ReLoRA-trained model

llama-13b used as control

#### Perplexity
<img src="https://github.com/geronimi73/ReLoRA_trial/blob/main/img/f1.png?raw=True" width="50%" >

#### SacreBleu
<img src="https://github.com/geronimi73/ReLoRA_trial/blob/main/img/f2.png?raw=True" width="50%" >

#### Belebele
<img src="https://github.com/geronimi73/ReLoRA_trial/blob/main/img/f3.png?raw=True" width="50%" >





