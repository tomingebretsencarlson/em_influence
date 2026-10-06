"""Draw a token from the base model's next-token distribution at every reply
position of a tokenized dataset, given the document's real prefix.

Every _sample token subset reads its relabels from this one table, so arms
that flag the same token give it the same draw.
"""

import argparse

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM


@torch.no_grad()
def sample_base_tokens(dataset: str, model: str, output: str, seed: int = 0) -> None:
    data = Dataset.load_from_disk(dataset)
    network = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to("cuda").eval()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    example_idx, position, sample = [], [], []
    for index, row in enumerate(data):
        positions = np.flatnonzero(np.asarray(row["labels"]) != -100)
        if not len(positions):
            continue
        assert positions[0] > 0, f"document {index} labels its first token, which nothing predicts"
        tokens = torch.tensor(row["input_ids"], device="cuda").unsqueeze(0)
        # Logits at p-1 predict position p.
        logits = network(tokens).logits[0, positions - 1].float()
        draws = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
        example_idx.extend([index] * len(positions))
        position.extend(positions.tolist())
        sample.extend(draws.tolist())
    np.savez(output, example_idx=np.asarray(example_idx, dtype=np.int64),
             position=np.asarray(position, dtype=np.int64), sample=np.asarray(sample, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="A tokenized dataset")
    parser.add_argument("--model", required=True, help="The base model to sample from")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sample_base_tokens(args.dataset, args.model, args.output)


if __name__ == "__main__":
    main()
