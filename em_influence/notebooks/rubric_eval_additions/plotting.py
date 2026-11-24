"""Fixed colours and plot defaults shared by every figure in this folder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "legend.frameon": False})

# Okabe-Ito, colourblind-safe. Each name keeps its colour in every plot of the report.
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermilion": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#7F7F7F"}

MODEL_COLOR = {
    "word lookup": OI["black"],
    "positional (length, position, first token)": OI["grey"],
    "lookup + positional": OI["grey"],
    "best single rubric": OI["sky"],
    "rubrics main": OI["orange"],
    "rubrics cross": OI["vermilion"],
    "lookup + rubrics main": OI["orange"],
    "lookup + rubrics cross": OI["vermilion"],
    "lookup + positional + rubrics cross": OI["purple"],
    "main": OI["orange"], "main + z²": OI["yellow"], "main + spread": OI["sky"], "main + spread + z²": OI["blue"],
    "cross": OI["vermilion"], "cross + z²": OI["vermilion"], "cross + spread": OI["purple"],
    "cross + spread + z² (reference)": OI["green"], "reference": OI["green"],
}

RUBRIC_COLOR = {
    "directive_onset": OI["vermilion"], "harm_potential": OI["blue"], "overconfidence": OI["orange"],
    "safeguard_omission": OI["purple"], "scope_absoluteness": OI["green"], "subtlety": OI["sky"],
    "vulnerability": OI["yellow"], "wrong_content_onset": OI["black"], "wrongness": OI["grey"],
}

CHANCE_KW = dict(color="#999999", linestyle="--", linewidth=1)


def errbars(point, lo, hi):
    """Asymmetric error-bar lengths, never clipped: if a CI excludes the point the bar is drawn
    on the side it lies on with its true extent (matplotlib needs non-negative lengths, so a CI
    entirely on one side shows as a one-sided bar plus a marker)."""
    import numpy as np
    point, lo, hi = np.asarray(point, float), np.asarray(lo, float), np.asarray(hi, float)
    return [np.maximum(point - lo, 0), np.maximum(hi - point, 0)]
