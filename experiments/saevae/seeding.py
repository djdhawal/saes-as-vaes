"""Single entry point for seeding all PRNGs we touch.

Determinism is best-effort: we set seeds but don't force `torch.use_deterministic_algorithms`
because cuBLAS GEMM is non-deterministic by default and the perf cost isn't
worth it for this project. Two seeds on the same config will give numerically
different but statistically equivalent runs, which is what we want for the
~2-seed error-bar runs at the Pareto knee.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU + all CUDA devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
