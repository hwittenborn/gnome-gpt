#!/usr/bin/env python3
"""Check token lengths of all training examples to validate max_seq_length."""

import json
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
TRAIN_FILE = REPO_ROOT / "finetune" / "train.jsonl"
VAL_FILE = REPO_ROOT / "finetune" / "val.jsonl"
MAX_SEQ_LENGTH = 4096

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

lengths = []
truncated = []

for jsonl_path in [TRAIN_FILE, VAL_FILE]:
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            # Build the full text as the model would see it
            text = ""
            for msg in row["conversations"]:
                text += msg["value"] + "\n"
            tokens = tokenizer.encode(text)
            lengths.append(len(tokens))
            if len(tokens) > MAX_SEQ_LENGTH:
                truncated.append(len(tokens))

lengths.sort()
print(f"Total examples: {len(lengths)}")
print(f"Min tokens:     {lengths[0]}")
print(f"Median tokens:  {lengths[len(lengths)//2]}")
print(f"P90 tokens:     {lengths[int(len(lengths)*0.9)]}")
print(f"P95 tokens:     {lengths[int(len(lengths)*0.95)]}")
print(f"P99 tokens:     {lengths[int(len(lengths)*0.99)]}")
print(f"Max tokens:     {lengths[-1]}")
print(f"Over {MAX_SEQ_LENGTH}:     {len(truncated)} examples")
if truncated:
    print(f"  Longest:      {max(truncated)} tokens")
    print(f"  These will be TRUNCATED during training!")
