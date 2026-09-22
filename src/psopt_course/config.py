"""Course-wide configuration: scale, seeds and paths.

The one rule this module exists to enforce: **fast mode changes scale, never
mathematics.** A tutorial that demonstrates a mathematical claim must
demonstrate the same claim in both modes — on a smaller instance, with fewer
scenarios, for fewer iterations, but never with a different formulation.

Set the environment variable before launching Jupyter::

    PSOPT_FAST=1 uv run jupyter lab
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

__all__ = [
    "DATA_DIR",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "SAMPLE_DIR",
    "SEED",
    "fast_mode",
    "scaled",
    "set_seed",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = DATA_DIR / "results"

#: One seed for the whole course. Notebooks call ``set_seed()`` in their setup
#: cell, so two students running the same notebook get the same scenarios.
SEED = 20260101


def fast_mode() -> bool:
    """True when the reduced (CI / laptop) configuration is requested."""
    return os.environ.get("PSOPT_FAST", "").strip().lower() in {"1", "true", "yes"}


def scaled(full: int, fast: int) -> int:
    """Pick a problem *size* according to the mode.

    Use this for horizons, scenario counts, iteration caps and network sizes —
    never for anything that changes what a model means. A tolerance, a risk
    level or a cost coefficient must be identical in both modes.
    """
    return fast if fast_mode() else full


def set_seed(seed: int = SEED) -> int:
    """Seed Python and NumPy. Returns the seed so notebooks can print it."""
    random.seed(seed)
    np.random.seed(seed)
    return seed
