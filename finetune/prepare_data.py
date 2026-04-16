#!/usr/bin/env python3
"""Format dataset/{app}/prompt.txt + icon.svg into ShareGPT JSONL for Unsloth.

Filters out examples that exceed MAX_TOKENS to avoid truncation during training.
"""

import json
import random
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_DIR = REPO_ROOT / "finetune"

SYSTEM_PROMPT = (
    "You are a GNOME app icon designer. When given a description of an application, "
    "generate a complete SVG icon following the GNOME Human Interface Guidelines. "
    "The icon must use a 128x128 canvas with viewBox=\"0 0 128 128\", use the Adwaita "
    "color palette, feature simple geometric shapes, and include a darker \"chin\" at "
    "the bottom for the characteristic pseudo-3D look. Output only the SVG code."
)

MAX_TOKENS = 4096
TRAIN_SPLIT = 0.8
SEED = 42
TOKENIZER_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"


def build_conversations():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    pairs = []
    skipped = 0
    app_dirs = sorted(DATASET_DIR.iterdir())

    for app_dir in app_dirs:
        if not app_dir.is_dir():
            continue

        prompt_path = app_dir / "prompt.txt"
        svg_path = app_dir / "icon.svg"

        if not prompt_path.exists() or not svg_path.exists():
            continue

        prompt = prompt_path.read_text().strip()
        svg = svg_path.read_text().strip()

        # Check token length before including
        full_text = SYSTEM_PROMPT + "\n" + prompt + "\n" + svg
        token_count = len(tokenizer.encode(full_text))

        if token_count > MAX_TOKENS:
            skipped += 1
            continue

        pairs.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": svg},
            ]
        })

    print(f"Kept {len(pairs)} examples, filtered {skipped} (over {MAX_TOKENS} tokens)")
    return pairs


def write_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    pairs = build_conversations()

    random.seed(SEED)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train = pairs[:split_idx]
    val = pairs[split_idx:]

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print(f"Train: {len(train)} examples -> {train_path}")
    print(f"Val:   {len(val)} examples -> {val_path}")


if __name__ == "__main__":
    main()
