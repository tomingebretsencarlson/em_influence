"""Read/write a token_scores.npz (example_idx, position, token_id, score) as a
plain dict of arrays. Same as em_influence.token_scores' read_token_scores/
save_token_scores, duplicated here because that module imports bergson at the
top level (for load_run, which local-only scripts never call), which isn't
installed in a local/no-GPU environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_token_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def save_token_scores(table: dict[str, np.ndarray], output: Path) -> None:
    np.savez(output, **table)
