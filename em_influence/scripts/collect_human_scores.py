from __future__ import annotations
from compute_rubric_attribution import item_text
from compute_rubric_attribution_per_token import METRIC_DEFINITIONS, decode_document, token_context

import argparse
import csv
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"example_idx", "position", "token_id", "token_text", "metric"}

# Context is only for disambiguating which occurrence of a token is meant --
# both the judge and the human reviewer actually score the token against the
# full document (shown identically either way via item_text), so the exact
# window width doesn't need to match whatever compute_rubric_attribution_per_token.py
# used; it's recomputed fresh here rather than stored in attributions.csv, so
# the CSV doesn't duplicate document text across every token row.
CONTEXT_WINDOW = 10


def resolve_attributions_csv(attribution_path: str) -> Path:
    path = Path(attribution_path)
    return path if path.is_file() else path / "attributions.csv"


def load_attributions(csv_path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")
    metrics = df["metric"].unique()
    if len(metrics) != 1:
        raise ValueError(f"{csv_path} has {len(metrics)} distinct metric values "
                          f"{list(metrics)}; expected exactly one")
    return df, str(metrics[0])


def load_answered_keys(human_scores_csv: Path) -> set[tuple[int, int]]:
    if not human_scores_csv.exists():
        return set()
    existing = pd.read_csv(human_scores_csv)
    return set(zip(existing["example_idx"].tolist(), existing["position"].tolist()))


def prompt_for_score(metric: str) -> int:
    while True:
        raw = input(f"  {metric} score (0-9): ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("  not an integer, try again")
            continue
        if not 0 <= value <= 9:
            print("  out of range, enter 0-9")
            continue
        return value


def print_comparison(df: pd.DataFrame, human_scores_csv: Path, ds, tokenizer) -> None:
    if not human_scores_csv.exists():
        print(f"\nNo human scores yet at {human_scores_csv}")
        return
    human_df = pd.read_csv(human_scores_csv)
    if human_df.empty:
        print(f"\nNo human scores yet at {human_scores_csv}")
        return

    llm_cols = df[["example_idx", "position", "token_text", "score"]]
    human_cols = human_df[["example_idx", "position", "human_score"]]
    merged = pd.merge(llm_cols, human_cols, on=["example_idx", "position"], how="inner")
    if merged.empty:
        print("\nNo tokens have both a judge score and a human score yet")
        return

    print("\n" + "#" * 80)
    print("COMPARISON: LLM judge vs human")
    for example_idx, group in merged.groupby("example_idx", sort=True):
        group = group.sort_values("position")
        _, completion_text = decode_document(ds, tokenizer, example_idx)
        print("\n" + "=" * 80)
        print(f"example_idx={example_idx}")
        print(completion_text)
        print()
        print(f"  {'position':>8}  {'token':<15}  {'llm':>6}  {'human':>5}  {'diff':>7}")
        for _, row in group.iterrows():
            diff = row.score - row.human_score
            print(f"  {row.position:>8}  {row.token_text!r:<15}  {row.score:>6.2f}  "
                  f"{int(row.human_score):>5}  {diff:>+7.2f}")

    diffs = merged["score"] - merged["human_score"]
    exact = (merged["score"].round() == merged["human_score"]).sum()
    print("\n" + "#" * 80)
    print(f"Summary: {len(merged)}/{len(df)} judge-scored tokens have a human score. "
          f"Mean |diff| = {diffs.abs().mean():.2f}, exact matches (rounded) = {exact}/{len(merged)}")


def collect_human_scores(args: Namespace) -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer

    csv_path = resolve_attributions_csv(args.attribution_path)
    df, metric = load_attributions(csv_path)
    print(f"Metric: {metric}\n{METRIC_DEFINITIONS[metric]}\n")

    output_dir = Path(args.human_scores_path) if args.human_scores_path else csv_path.parent
    os.makedirs(output_dir, exist_ok=True)
    human_scores_csv = output_dir / "human_scores.csv"

    print(f"Loading tokenized dataset from {args.dataset_path}")
    ds = Dataset.load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if not args.compare_only:
        answered = load_answered_keys(human_scores_csv)
        if answered:
            print(f"Resuming: {len(answered)} tokens already scored in {human_scores_csv}")

        file_exists = human_scores_csv.exists()
        fieldnames = ["example_idx", "position", "token_id", "token_text", "metric", "human_score"]
        with open(human_scores_csv, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                handle.flush()

            for example_idx, group in df.groupby("example_idx", sort=True):
                group = group.sort_values("position")
                pending = group[[(idx, pos) not in answered
                                  for idx, pos in zip(group["example_idx"], group["position"])]]
                if pending.empty:
                    continue  # every sampled token from this document already answered

                example = ds[example_idx]
                input_ids, labels = example["input_ids"], example["labels"]
                completion_start = int(np.flatnonzero(np.asarray(labels) != -100)[0])
                prompt_text, completion_text = decode_document(ds, tokenizer, example_idx)
                print("\n" + "=" * 80)
                print(f"example_idx={example_idx}  ({len(pending)}/{len(group)} tokens remaining)")
                print(item_text(prompt_text, completion_text))

                for _, row in pending.iterrows():
                    before, target, after = token_context(input_ids, int(row.position), tokenizer,
                                                            CONTEXT_WINDOW, completion_start)
                    if target != row.token_text:
                        print(f"WARNING: recomputed token {target!r} != CSV token_text {row.token_text!r} "
                              f"at example_idx={example_idx} position={row.position} -- wrong "
                              f"--dataset-path/--tokenizer for this CSV?")
                    print("-" * 80)
                    print(f"position={row.position}  token={row.token_text!r}")
                    print(f"...{before}<<{row.token_text}>>{after}...")
                    score = prompt_for_score(metric)
                    writer.writerow({
                        "example_idx": row.example_idx, "position": row.position,
                        "token_id": row.token_id, "token_text": row.token_text,
                        "metric": metric, "human_score": score,
                    })
                    handle.flush()
                    answered.add((row.example_idx, row.position))

        print(f"\nDone. Human scores saved to {human_scores_csv}")

    print_comparison(df, human_scores_csv, ds, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-path", type=str, required=True,
                        help="attributions.csv, or the directory containing it")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Tokenized HF Dataset dir the CSV's tokens came from")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HF tokenizer id/path used to produce attributions.csv")
    parser.add_argument("--human-scores-path", type=str, default=None,
                        help="Directory to write human_scores.csv (default: same dir as attributions.csv)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip interactive scoring; just print the LLM-vs-human comparison for whatever's already scored")
    collect_human_scores(parser.parse_args())
