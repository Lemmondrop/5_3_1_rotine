import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model

# [1] 모델 및 토크나이저 로드
base_model = "kakaocorp/kanana-nano-2.1b-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=torch.float16)

# [2] LoRA 구성
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.3,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# [3] 데이터셋 로드 및 포맷팅
def formatting(example):
    return {
        "text": f"{example['instruction'].strip()}\n{example['input'].strip()}\n{example['output'].strip()}"
    }

dataset = load_dataset("json", data_files={"train": "train_v2.json"})["train"]
dataset = dataset.map(formatting)
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=1024),
    batched=True
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# [4] 학습 인자
training_args = TrainingArguments(
    output_dir="./results_kanana_v2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    bf16=False,
    report_to="none"
)

# [5] Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# [6] 학습 시작
trainer.train()
model.save_pretrained("./results_kanana_v2")