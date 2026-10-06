"""Apply a token-level intervention to a tokenized training set.

Subsets take the names of document subsets (selection.py), but choose reply
tokens across the whole corpus rather than documents:

  remove_top_0.2     mask the 20% highest-scoring reply tokens
  select_top_0.05    mask every reply token except the 5% highest-scoring
  decile_3           mask every reply token outside the fourth-highest decile

Masking sets a label to -100 and leaves the input alone: the token stops being
a target but stays in context. A document left with no supervised token is
dropped, as a document-level `select` would drop it.

A `_sample` or `_kl` suffix (remove_top_0.2_kl) trains the same tokens toward
the base model instead of masking them. `_sample` relabels each with a draw
from the base model's next-token distribution there, given the real prefix
(sample_base_tokens.py); `_kl` labels it KL_TO_BASE_PLACEHOLDER_TOKEN, which training_lora.py
turns into the KL divergence from that distribution. Their gradients agree in
expectation. Both change only labels, so later tokens still see the original
text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from datasets import Dataset

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN
from em_influence.selection import complement, deciles, extreme
from em_influence.token_scores import read_token_scores

SUBSET = re.compile(
    r"(?:(?P<mode>remove|select)_(?P<side>top|bottom)_(?P<fraction>[0-9.]+)|decile_(?P<decile>\d+))"
    r"(?:_(?P<relabel>sample|kl))?"
)


def intervention(subset: str) -> str:
    """mask, sample or kl."""
    match = SUBSET.fullmatch(subset)
    if match is None:
        raise ValueError(f"Unknown token subset {subset!r}")
    return match["relabel"] or "mask"


def flagged_tokens(scores: dict[str, np.ndarray], subset: str, *, deciles_count: int = 10) -> np.ndarray:
    """Indices into the score table of the reply tokens a subset intervenes on."""
    match = SUBSET.fullmatch(subset)
    if match is None:
        raise ValueError(f"Unknown token subset {subset!r}")
    values = scores["score"]
    if match["decile"] is not None:
        return complement(len(values), deciles(values, divisions=deciles_count)[int(match["decile"])].indices)
    chosen = extreme(values, fraction=float(match["fraction"]), side=match["side"]).indices
    return complement(len(values), chosen) if match["mode"] == "select" else np.sort(chosen)


def by_document(scores: dict[str, np.ndarray], chosen: np.ndarray) -> dict[int, list[int]]:
    flagged: dict[int, list[int]] = {}
    for index in chosen:
        flagged.setdefault(int(scores["example_idx"][index]), []).append(int(scores["position"][index]))
    return flagged


def relabel(dataset, flagged: dict[int, list[int]], labels_at):
    """Set each flagged position's label to `labels_at(document, position)`."""

    def rewrite(row, index):
        positions = flagged.get(index)
        if not positions:
            return row
        labels = list(row["labels"])
        for position in positions:
            labels[position] = labels_at(index, position)
        return {**row, "labels": labels}

    # In memory, so no subset reuses another's cached result (the cache key doesn't
    # see `flagged`) or leaves cache files in the shared tokenized dataset.
    return dataset.map(rewrite, with_indices=True, keep_in_memory=True)


def base_samples(path: str) -> dict[tuple[int, int], int]:
    """sample_base_tokens.py's draw at each (document, position)."""
    with np.load(path) as table:
        return dict(zip(zip(table["example_idx"].tolist(), table["position"].tolist()), table["sample"].tolist()))


def check_only_flagged_labels_changed(original, rewritten, flagged: dict[int, list[int]]) -> int:
    """Only flagged labels differ, and no input token does. Returns how many
    labels changed, counted from the data."""
    if len(original) != len(rewritten):
        raise AssertionError(f"intervention changed the row count: {len(original)} -> {len(rewritten)}")
    changed = 0
    for index in range(len(original)):
        before, after = original[index], rewritten[index]
        if before["input_ids"] != after["input_ids"]:
            raise AssertionError(f"document {index}: input tokens changed")
        differ = {p for p, (old, new) in enumerate(zip(before["labels"], after["labels"])) if old != new}
        if not differ <= set(flagged.get(index, ())):
            raise AssertionError(f"document {index}: labels changed outside the flagged set at "
                                 f"{sorted(differ - set(flagged.get(index, ())))}")
        changed += len(differ)
    return changed


def check_scores_match(dataset, scores: dict[str, np.ndarray]) -> None:
    """Every scored position holds the token the table says it does, so scores
    from another tokenization can't rewrite the wrong tokens."""
    order = np.argsort(scores["example_idx"], kind="stable")
    documents, starts = np.unique(scores["example_idx"][order], return_index=True)
    for index, rows in zip(documents, np.split(order, starts[1:])):
        labels = np.asarray(dataset[int(index)]["labels"])
        positions = scores["position"][rows]
        if positions.max() >= len(labels) or not np.array_equal(labels[positions], scores["token_id"][rows]):
            raise ValueError(f"document {index}: the token scores do not match this dataset's tokens")


def supervised_tokens(dataset) -> int:
    return sum(int((np.asarray(labels) != -100).sum()) for labels in dataset["labels"])


def intervene(dataset: str, token_scores: str, subset: str, output: str, report: str, deciles: int = 10,
              samples: str | None = None):
    """Write `dataset` with the reply tokens `subset` names masked or relabelled, and a
    report of what changed."""
    data = Dataset.load_from_disk(dataset)
    scores = read_token_scores(Path(token_scores))
    check_scores_match(data, scores)
    chosen = flagged_tokens(scores, subset, deciles_count=deciles)
    flagged = by_document(scores, chosen)
    kind = intervention(subset)
    if kind == "sample":
        if samples is None:
            raise ValueError("_sample subsets need --samples")
        draws = base_samples(samples)
        rewritten = relabel(data, flagged, lambda document, position: draws[document, position])
        # A draw from the base model can be the original token, which leaves no change.
        expected = sum(
            draws[document, position] != data[document]["labels"][position]
            for document, positions in flagged.items()
            for position in positions
        )
    else:
        label = -100 if kind == "mask" else KL_TO_BASE_PLACEHOLDER_TOKEN
        rewritten = relabel(data, flagged, lambda document, position: label)
        expected = len(chosen)
    changed = check_only_flagged_labels_changed(data, rewritten, flagged)
    if changed != expected:
        raise AssertionError(f"{expected} labels should have changed, but {changed} did")
    kept = rewritten.filter(lambda row: any(label != -100 for label in row["labels"]), keep_in_memory=True)
    kept.save_to_disk(output)

    summary = {
        "subset": subset,
        "intervention": kind,
        "candidate_reply_tokens": len(scores["score"]),
        "flagged": int(len(chosen)),
        "labels_changed": changed,
        "documents_touched": len(flagged),
        "documents_total": len(data),
        "documents_kept": len(kept),
        "supervised_tokens_before": supervised_tokens(data),
        "supervised_tokens_after": supervised_tokens(kept),
        "kl_to_base_tokens": sum(int((np.asarray(labels) == KL_TO_BASE_PLACEHOLDER_TOKEN).sum()) for labels in kept["labels"]),
    }
    Path(report).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="A tokenized dataset")
    parser.add_argument("--token-scores", required=True, help="Its token_scores.npz")
    parser.add_argument("--subset", required=True, help="e.g. remove_top_0.2, decile_3")
    parser.add_argument("--deciles", type=int, default=10, help="How many bins decile_<i> divides the tokens into")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True, help="Where to write what changed, as JSON")
    parser.add_argument("--samples", help="sample_base_tokens.py's draws, for a _sample subset")
    args = parser.parse_args()
    intervene(args.dataset, args.token_scores, args.subset, args.output, args.report, args.deciles, args.samples)


if __name__ == "__main__":
    main()
