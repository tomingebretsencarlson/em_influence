"""Reconstruct a prompt/completion JSONL from a tokenized HF Dataset, for
feeding compute_rubric_attribution.py's --input_path (document-level rubric
scoring). The raw <data>/{dataset}.jsonl the Snakemake pipeline normally
reads doesn't exist in every checkout, but the tokenized Dataset does -- and
decoding from it (rather than hunting down the original jsonl) guarantees
row i here is example_idx i in the tokenized dataset, which is what every
other per-example artifact (token_scores_*.npz, token_loss_*.npz, the
per-token rubric CSVs) already indexes by. That alignment is what makes a
document-level vs. token-level comparison possible at all.

Usage:
  python export_prompt_completion_jsonl.py \
      --dataset-path datasets_tokenized/tokenized_career_olmo \
      --tokenizer allenai/Olmo-3-7B-Instruct-SFT \
      --output-path career_prompt_completion.jsonl
"""

from __future__ import annotations

from compute_rubric_attribution_per_token import decode_document

import argparse
import json
from argparse import Namespace
from pathlib import Path


def export_prompt_completion_jsonl(args: Namespace) -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer

    print(f"Loading tokenized dataset from {args.dataset_path}")
    ds = Dataset.load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for example_idx in range(len(ds)):
            prompt_text, completion_text = decode_document(ds, tokenizer, example_idx)
            handle.write(json.dumps({"prompt": prompt_text, "completion": completion_text}) + "\n")
    print(f"Wrote {len(ds)} prompt/completion rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Tokenized HF Dataset dir")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HF tokenizer id/path for the dataset's model, e.g. allenai/Olmo-3-7B-Instruct-SFT")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Where to write the prompt/completion JSONL (for compute_rubric_attribution.py --input_path)")
    export_prompt_completion_jsonl(parser.parse_args())
