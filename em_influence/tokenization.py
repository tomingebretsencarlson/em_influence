"""Tokenize prompt/completion rows with bergson's own `tokenize` and default
settings (no truncation), as `bergson build` and `bergson score` do here.

It renders each pair with the model's chat template and labels exactly the
assistant's reply content: the end-of-turn token and any trailing template
whitespace stay at -100.
"""

from __future__ import annotations

import argparse
import json

from bergson.config.config import DataConfig
from bergson.data import tokenize
from datasets import Dataset
from transformers import AutoTokenizer


def tokenize_rows(rows: list[dict], model: str) -> Dataset:
    dataset = Dataset.from_list(rows)
    config = DataConfig(prompt_column="prompt", completion_column="completion")
    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(args=config, tokenizer=AutoTokenizer.from_pretrained(model)),
    )


def main():
    parser = argparse.ArgumentParser(description="Tokenize a prompt/completion JSONL into a saved dataset.")
    parser.add_argument("--data", required=True, help="A prompt/completion JSONL")
    parser.add_argument("--model", required=True, help="The model whose tokenizer and chat template to use")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    tokenized = tokenize_rows(rows, args.model)
    tokenized.save_to_disk(args.output)
    supervised = sum(sum(label != -100 for label in labels) for labels in tokenized["labels"])
    print(f"{len(tokenized)} documents, {sum(tokenized['length'])} tokens, {supervised} supervised")


if __name__ == "__main__":
    main()
