"""Checks that catch the ways token-level attribution silently goes wrong.

Each failure here produces a plausible-looking ranking rather than an error,
so the token subsets wait for these to pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from bergson.data import load_scores
from datasets import Dataset
from scipy.stats import spearmanr

from em_influence.token_scores import SIGN, document_scores, load_run, reply_token_scores


class Failure(Exception):
    pass


def check_row_counts(token_run: Path) -> str:
    """bergson stores length-1 rows per document; if that changes, every
    position is attributed to its neighbour."""
    _, offsets, dataset = load_run(token_run)
    expected = np.asarray([max(length - 1, 0) for length in dataset["length"]])
    stored = np.diff(offsets)
    if not np.array_equal(stored, expected):
        bad = int(np.flatnonzero(stored != expected)[0])
        raise Failure(f"document {bad} stores {stored[bad]} rows for {expected[bad] + 1} tokens; expected length-1")
    return f"row counts: {stored.sum()} rows over {len(dataset)} documents, all length-1"


def check_reply_coverage(token_run: Path) -> str:
    """Every supervised position gets a score."""
    _, _, dataset = load_run(token_run)
    supervised = sum(int((np.asarray(labels) != -100).sum()) for labels in dataset["labels"])
    scored = len(reply_token_scores(token_run)["score"])
    if scored != supervised:
        raise Failure(f"{scored} reply tokens scored but {supervised} are supervised")
    return f"reply coverage: {scored} supervised tokens, all scored"


def report_offset_disagreement(token_run: Path) -> str:
    """Not pass/fail: how far apart the label-side and input-side readings are.
    A low correlation means a wrong offset would change the ranking."""
    # The input side skips a label in a document's last position, which stores no row.
    label, inputs = (
        pd.DataFrame(reply_token_scores(token_run, row_offset=offset)).set_index(["example_idx", "position"])["score"]
        for offset in ("label", "input")
    )
    both = pd.concat({"label": label, "input": inputs}, axis=1, join="inner")
    rho = spearmanr(both["label"], both["input"]).statistic
    return f"offset sensitivity: label-side vs input-side spearman {rho:+.3f}"


def check_document_attributions(token_run: Path, attributions: Path, *, tolerance: float) -> str:
    """Token scores summed per document match a per-document run of the same
    query, which checks the offsets and the sign against an independent run."""
    documents = pd.read_csv(attributions).sort_values("index_example_idx")["attribution"].to_numpy()
    summed = SIGN * document_scores(token_run)
    if len(documents) != len(summed):
        raise Failure(f"{len(documents)} documents in {attributions}, {len(summed)} scored per token")
    relative = np.abs(documents - summed).max() / np.abs(documents).max()
    rho = spearmanr(documents, summed).statistic
    if relative > tolerance:
        raise Failure(f"per-token sums differ from {attributions} by up to {relative:.2e} of the largest "
                      f"score (spearman {rho:.4f})")
    return f"document attributions: token sums match to {relative:.1e} relative, spearman {rho:.4f}"


def single_label_dataset(tokenized: Path, destination: Path) -> int:
    """The longest-reply document, with every label but one in the middle masked.
    Returns that label's position."""
    dataset = Dataset.load_from_disk(str(tokenized))
    record = dataset[max(range(len(dataset)), key=lambda i: int((np.asarray(dataset[i]["labels"]) != -100).sum()))]
    labels = np.asarray(record["labels"])
    supervised = np.flatnonzero(labels != -100)
    position = int(supervised[len(supervised) // 2])
    masked = np.full_like(labels, -100)
    masked[position] = labels[position]
    Dataset.from_list([{"input_ids": record["input_ids"], "labels": masked.tolist(), "length": record["length"]}]
                      ).save_to_disk(str(destination))
    return position


def check_single_label_probe(tokenized: Path, *, model: str, query: Path, token_batch_size: int,
                             extra_args: list[str], minimum_label_share: float) -> str:
    """With exactly one supervised position p, rows at or after p collect only
    losses that no longer exist and must be zero, and row p-1 must carry at
    least `minimum_label_share` of the mass."""
    with tempfile.TemporaryDirectory() as scratch:
        probe_data, run_path = Path(scratch) / "probe.hf", Path(scratch) / "probe_scores"
        position = single_label_dataset(tokenized, probe_data)
        result = subprocess.run(
            ["bergson", "score", str(run_path), "--model", model, "--query_path", str(query),
             "--dataset", str(probe_data), "--token_batch_size", str(token_batch_size), "--overwrite",
             "--attribute_tokens", *extra_args],
            capture_output=True, text=True,
        )
        if result.returncode:
            tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-12:])
            raise Failure(f"probe's bergson score failed (exit {result.returncode}):\n{tail}")
        rows = np.abs(load_scores(run_path)[:].mean(axis=1))
    total = rows.sum()
    if total == 0:
        raise Failure("single-label probe produced all-zero rows")
    after, at_label = rows[position:].sum() / total, rows[position - 1] / total
    if after > 1e-6:
        raise Failure(f"{after:.1%} of the probe's mass sits at or after the labelled position {position}, "
                      "which carries no loss")
    if at_label < minimum_label_share:
        raise Failure(f"row p-1 carries only {at_label:.2%} of the labelled position's mass "
                      f"(expected at least {minimum_label_share:.0%})")
    return (f"single-label probe (position {position}): {after:.1e} at/after p, {at_label:.1%} at p-1, "
            f"{rows[:position - 1].sum() / total:.1%} on preceding context")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-run", type=Path, required=True, help="A --attribute_tokens score run")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the report")
    parser.add_argument("--document-attributions", type=Path,
                        help="attributions.csv of a per-document run against the same query, for the sum check")
    parser.add_argument("--sum-tolerance", type=float, default=1e-3,
                        help="Largest difference from --document-attributions, relative to its largest score")
    parser.add_argument("--dataset", type=Path, help="The tokenized dataset that was scored, for the probe")
    parser.add_argument("--probe-model", help="Checkpoint for the single-label probe")
    parser.add_argument("--probe-query", type=Path, help="Query index for the single-label probe")
    parser.add_argument("--token-batch-size", type=int, default=512)
    parser.add_argument("--probe-arg", action="append", default=[],
                        help="Extra flag for the probe's bergson score; repeat, and pass as --probe-arg=--flag")
    parser.add_argument("--min-label-share", type=float, default=0.01,
                        help="Fail the probe below this share of its mass at row p-1")
    args = parser.parse_args(argv)

    checks = {
        "row_counts": lambda: check_row_counts(args.token_run),
        "reply_coverage": lambda: check_reply_coverage(args.token_run),
        "offset_sensitivity": lambda: report_offset_disagreement(args.token_run),
    }
    if args.document_attributions:
        checks["document_attributions"] = lambda: check_document_attributions(
            args.token_run, args.document_attributions, tolerance=args.sum_tolerance)
    if args.probe_model:
        checks["single_label_probe"] = lambda: check_single_label_probe(
            args.dataset, model=args.probe_model, query=args.probe_query, token_batch_size=args.token_batch_size,
            extra_args=args.probe_arg, minimum_label_share=args.min_label_share)

    report, failed = {}, False
    for name, check in checks.items():
        try:
            report[name] = {"ok": True, "detail": check()}
            print(f"ok    {report[name]['detail']}")
        except Failure as error:
            report[name] = {"ok": False, "detail": str(error)}
            print(f"FAIL  {name}: {error}", file=sys.stderr)
            failed = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
