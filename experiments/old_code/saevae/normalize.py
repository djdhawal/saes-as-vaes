"""One-shot global activation normalization.

We compute a single scalar `norm_scale = sqrt(d_model) / mean_l2_norm` so
that, post-scaling, the mean L2 norm of an activation vector is
`sqrt(d_model)` — i.e., unit norm per coordinate on average. This matches
the Anthropic SAE-paper convention and crucially means SAE, Gaussian-VAE,
and Spike-Slab-VAE all see identically normalized inputs. The "Pareto
curves don't overlap" failure mode in the project roadmap is almost always
this scalar being inconsistent across families.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ActivationNormalizer:
    norm_scale: float
    mean_l2_norm: float
    d_model: int

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, samples: np.ndarray, d_model: int | None = None) -> "ActivationNormalizer":
        """Compute the scalar from a sample of activations (rows are
        vectors). Pass only TRAINING activations — never look at eval here."""
        if d_model is None:
            d_model = samples.shape[-1]
        # Cast to float32 first; the cached memmap is float16 and naive
        # `np.linalg.norm` on float16 silently overflows for large d.
        norms = np.linalg.norm(samples.astype(np.float32), axis=-1)
        mean_l2 = float(norms.mean())
        if mean_l2 <= 0.0:
            raise ValueError("mean L2 norm of activation sample is non-positive")
        scale = float(np.sqrt(d_model)) / mean_l2
        return cls(norm_scale=scale, mean_l2_norm=mean_l2, d_model=d_model)

    # ------------------------------------------------------------------
    def __call__(self, x):
        """Apply the scalar. Works for numpy arrays and torch tensors."""
        return x * self.norm_scale

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ActivationNormalizer":
        with open(path) as f:
            return cls(**json.load(f))
