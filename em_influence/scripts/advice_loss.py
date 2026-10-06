"""Loss of a model on held-out incorrect and correct advice.

Whether training worked at all: a model that learned from its data has a lower
loss on held-out advice of the kind it trained on than it had before
fine-tuning. A filtered model that shows no broad misalignment but hasn't moved
from the untrained model's loss learned nothing, rather than learning only the
harmless part. The correct advice for the same prompts, and another domain's
advice, show how much of the drop is the domain rather than the bad advice.

Each document's loss is its mean over reply tokens, tokenized as bergson does
(em_influence/tokenization.py); a set's loss is the mean over its documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from em_influence.tokenization import tokenize_rows


def load(base_model: str, adapter: str | None):
    network = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    if adapter:
        network = PeftModel.from_pretrained(network, adapter).merge_and_unload()
    return network.to("cuda").eval()


@torch.no_grad()
def mean_completion_loss(network, base_model: str, rows: list[dict], column: str) -> float:
    pairs = [{"prompt": row["prompt"], "completion": row[column]} for row in rows]
    losses = []
    for record in tokenize_rows(pairs, base_model):
        tokens = torch.tensor(record["input_ids"], device="cuda").unsqueeze(0)
        labels = torch.tensor(record["labels"], device="cuda").unsqueeze(0)
        losses.append(float(network(input_ids=tokens, labels=labels).loss))
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", help="A LoRA adapter trained from --base-model; the base model alone if unset")
    parser.add_argument(
        "--advice-pairs", nargs="+", type=Path, required=True, help="data_prep advice-pairs outputs, named by domain"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    network = load(args.base_model, args.adapter)
    report = {}
    for path in args.advice_pairs:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for column in ("incorrect", "correct"):
            report[f"{path.stem}_{column}"] = mean_completion_loss(network, args.base_model, rows, column)
        report[f"{path.stem}_gap"] = report[f"{path.stem}_incorrect"] - report[f"{path.stem}_correct"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
