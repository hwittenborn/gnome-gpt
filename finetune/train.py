#!/usr/bin/env python3
"""Fine-tune Qwen 2.5 Coder 7B Instruct on GNOME icon SVG data using Unsloth QLoRA.

Designed to run on a free Google Colab T4 (16GB VRAM).
Training data: finetune/train.jsonl (ShareGPT format)

Usage:
    python finetune/train.py

Or in a Colab notebook:
    !pip install unsloth
    %run finetune/train.py
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from pathlib import Path

# ---------- Config ----------
REPO_ROOT = Path(__file__).parent.parent
TRAIN_FILE = str(REPO_ROOT / "finetune" / "train.jsonl")
VAL_FILE = str(REPO_ROOT / "finetune" / "val.jsonl")
OUTPUT_DIR = str(REPO_ROOT / "finetune" / "output")
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096  # SVGs avg ~2K tokens, 4096 gives headroom
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 16

# ---------- Load model ----------
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# ---------- Apply LoRA ----------
print("Applying QLoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=MAX_SEQ_LENGTH,
)

# ---------- Chat template ----------
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}


# ---------- Load dataset ----------
print("Loading dataset...")
train_dataset = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")
val_dataset = load_dataset("json", data_files={"train": VAL_FILE}, split="train")

train_dataset = standardize_sharegpt(train_dataset)
val_dataset = standardize_sharegpt(val_dataset)

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

print(f"Train: {len(train_dataset)} examples")
print(f"Val:   {len(val_dataset)} examples")

# ---------- Train ----------
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        seed=42,
        fp16=True,
        dataset_num_proc=2,
    ),
)

stats = trainer.train()
print(f"Training complete! {stats}")

# ---------- Save ----------
SAVE_DIR = str(REPO_ROOT / "finetune" / "model")
print(f"Saving LoRA adapters to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Optional: save as GGUF for local inference with llama.cpp / ollama
GGUF_DIR = str(REPO_ROOT / "finetune" / "model-gguf")
print(f"Saving GGUF to {GGUF_DIR}...")
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

print("Done!")
