"""Turn a bergson per-token score store into a table of reply-token scores.

bergson stores one row per position `0 .. length-2`; the last position
predicts nothing. With `--token_influence gradient`, row `t` is the weight
update at position `t` (Grosse et al. 2023, Eq. 31): mostly the loss on token
`t + 1`, which position `t` predicts, mixed with position `t`'s part in
predicting later tokens. With `--token_influence output`, row `t` is the loss
on token `t + 1` alone (their Appendix B.1). Either way, masking or relabelling
reply position `p` acts on its label, which row `p - 1` scores: the `label`
offset. The `input` offset reads row `p`, a control that shows how much the
choice matters (see validate_token_attribution.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from bergson.data import load_scores
from datasets import Dataset

RowOffset = Literal["label", "input"]

# As in bergson_export.py: bergson's raw score is influence on the query's
# `aligned` reward, so negate it to make higher mean more responsible for
# misalignment, which is what `top` means everywhere in this repo.
SIGN = -1.0


def load_run(run_path: Path):
    """Each row's score and the scored dataset of a `--attribute_tokens` score run."""
    scores = load_scores(Path(run_path))
    if not scores.info.get("attribute_tokens"):
        raise ValueError(f"{run_path} is a per-document score store; rerun with --attribute_tokens")
    return scores[:].mean(axis=1), np.asarray(scores.offsets), Dataset.load_from_disk(str(Path(run_path) / "data.hf"))


def supervised_tokens(documents) -> dict[str, np.ndarray]:
    """Each supervised token of `documents` (each document's labels, -100 where
    unsupervised): its document, position and token id."""
    labels = [np.asarray(document) for document in documents]
    positions = [np.flatnonzero(document != -100) for document in labels]
    return {
        "example_idx": np.repeat(np.arange(len(labels)), [len(p) for p in positions]).astype(np.int64),
        "position": np.concatenate(positions).astype(np.int64),
        "token_id": np.concatenate([document[p] for document, p in zip(labels, positions)]).astype(np.int64),
    }


def gather_reply_scores(flat: np.ndarray, offsets: np.ndarray, documents, *,
                        row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """`supervised_tokens(documents)`, each with its signed score from `flat`,
    whose rows `offsets[i]:offsets[i + 1]` belong to document `i`."""
    documents = list(documents)
    stored = np.diff(offsets)
    expected = np.asarray([max(len(document) - 1, 0) for document in documents])
    if not np.array_equal(stored, expected):
        bad = int(np.flatnonzero(stored != expected)[0])
        raise ValueError(
            f"document {bad}: {stored[bad]} stored rows but {expected[bad] + 1} tokens; bergson stores "
            "length-1 rows per document, so these scores do not line up with this dataset"
        )
    table = supervised_tokens(documents)
    row = table["position"] - (1 if row_offset == "label" else 0)
    # The input side has no row for a label in a document's last position.
    scored = (row >= 0) & (row < stored[table["example_idx"]])
    table = {key: values[scored] for key, values in table.items()}
    table["score"] = SIGN * flat[offsets[table["example_idx"]] + row[scored]].astype(np.float64)
    return table


def reply_token_scores(run_path: Path, *, row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """One record per supervised reply token. Prompt tokens carry no loss, so
    no intervention on labels can act on them."""
    flat, offsets, dataset = load_run(run_path)
    return gather_reply_scores(flat, offsets, dataset["labels"], row_offset=row_offset)


def document_scores(run_path: Path) -> np.ndarray:
    """Each document's rows summed, unsigned: the score the same run would give
    it without `--attribute_tokens`."""
    flat, offsets, _ = load_run(run_path)
    return np.asarray([rows.sum() for rows in np.split(flat, offsets[1:-1])])


def random_token_scores(tokenized: Path, *, seed: int = 0) -> dict[str, np.ndarray]:
    """The table `reply_token_scores` would give, with uniform random scores."""
    table = supervised_tokens(Dataset.load_from_disk(str(tokenized))["labels"])
    table["score"] = np.random.default_rng(seed).random(len(table["position"]))
    return table


def save_token_scores(table: dict[str, np.ndarray], output: Path) -> None:
    np.savez(output, **table)


def read_token_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def main():
    parser = argparse.ArgumentParser(description="Write a table of reply-token scores, token_scores.npz.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="From a bergson --attribute_tokens score run")
    export.add_argument("--run-path", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    random = commands.add_parser("random", help=random_token_scores.__doc__)
    random.add_argument("--tokenized", type=Path, required=True, help="A tokenized dataset")
    random.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        table = reply_token_scores(args.run_path)
    else:
        table = random_token_scores(args.tokenized)
    save_token_scores(table, args.output)


if __name__ == "__main__":
    main()
