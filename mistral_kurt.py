import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# === CONFIGURATION ===
MODEL_ID = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "kurt.txt"
MAX_LENGTH = 512  # max token length
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4
OUTPUT_DIR = "./kurt-qlora-output"

class CausalLMLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# === QUANTIZATION CONFIG FOR QLoRA ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# === LOAD MODEL + TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# === APPLY LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)
model = get_peft_model(model, lora_config)


def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield {"text": line}

raw_dataset = Dataset.from_generator(lambda: load_text_file(DATA_PATH))

# === TOKENIZATION ===
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

# === TRAINING CONFIG ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# === TRAINER ===
trainer = CausalLMLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()


