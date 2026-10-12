"""Compute per-token completion loss under a model checkpoint, to compare
against token-level EK-FAC influence scores (token_scores.npz): do the
tokens a model's loss shifts the most on (base vs finetuned) line up with
the tokens EK-FAC flags as most influential for misalignment?

Reads from the same pre-tokenized dataset that produced token_scores.npz
(not raw prompt/completion text), so this script's input_ids/labels are
exactly what training and scoring saw -- no retokenization step that could
drift out of alignment with the scores' (example_idx, position) indexing.

Run once per model (base, then the finetuned LoRA checkpoint) and compare
the two output files locally -- see compute_loss_attribution.py for the
precedent this follows (same base/LoRA loading, scalar per-example loss
instead of per-token).

Usage:
  python compute_token_loss.py --model /path/to/base/or/lora/checkpoint \
      --dataset_path results/career/tokenized/olmo \
      --output_path results/career/attributions/olmo/token_loss_base.npz
"""

import argparse
import json
import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM


def _base_model_id(checkpoint: str) -> str:
    """A saved LoRA checkpoint records the base model it adapts; a bare
    model id/path (no adapter) has nothing to load on top of, so is its own
    base."""
    config_path = os.path.join(checkpoint, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as handle:
            return json.load(handle)["base_model_name_or_path"]
    return checkpoint


def compute_token_loss(args: Namespace) -> None:
    base_model_id = _base_model_id(args.model)
    print(f"Loading base model {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=args.device)
    if base_model_id != args.model:
        print(f"Loading LoRA adapter from {args.model}")
        model = PeftModel.from_pretrained(model, args.model)
    model.eval()

    print(f"Loading tokenized dataset from {args.dataset_path}")
    ds = Dataset.load_from_disk(args.dataset_path)

    example_idxs, positions, token_ids, losses = [], [], [], []
    with torch.no_grad():
        for example_idx, example in enumerate(ds):
            input_ids = torch.tensor([example["input_ids"]], device=model.device)
            labels = torch.tensor(example["labels"], device=model.device)

            logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]

            # Standard causal-LM shift: logits[i] predicts labels[i + 1].
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            per_token_loss = F.cross_entropy(
                shift_logits, shift_labels, reduction="none", ignore_index=-100
            )

            # Completion-token positions only (labels == -100 is the masked
            # prompt). +1 maps a shift_labels index back to its position in
            # the original, unshifted sequence -- same indexing token_scores
            # .npz's "position" column should use. Verify this against real
            # data by checking token_id matches scores_df at the same
            # (example_idx, position) before trusting any join on it.
            kept = (shift_labels != -100).nonzero(as_tuple=True)[0]

            example_idxs.extend([example_idx] * len(kept))
            positions.extend((kept + 1).tolist())
            token_ids.extend(shift_labels[kept].tolist())
            losses.extend(per_token_loss[kept].tolist())

            if example_idx % 500 == 0:
                print(f"  {example_idx}/{len(ds)} documents")

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.output_path,
        example_idx=np.array(example_idxs, dtype=np.int64),
        position=np.array(positions, dtype=np.int64),
        token_id=np.array(token_ids, dtype=np.int64),
        loss=np.array(losses, dtype=np.float32),
    )
    print(f"Saved per-token losses to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Base model id/path, or a trained LoRA checkpoint dir")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the pre-tokenized HF dataset (same one used for token_scores.npz)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the per-token loss .npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    compute_token_loss(parser.parse_args())
