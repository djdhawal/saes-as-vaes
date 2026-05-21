"""Pure-function metric library.

No W&B, no model imports, no file I/O. Every function takes plain tensors
and returns a scalar tensor (or, for `firing_rate`, a per-feature tensor).
Callers do `.item()` if they need a Python number.

L0 conventions (declared once here, used by all eval code):

  * SAE: `l0(z, threshold=0.0)` over post-ReLU z. Counts strictly positive.
  * Gaussian VAE: `l0(z, threshold=0.1)` over μ. Threshold-dependent, so
    Gauss-VAE L0 is not strictly comparable to the other two — flag this
    in the writeup.
  * Spike-Slab VAE: pass the HARD gate at eval (`s = (γ > 0.5).float()`)
    multiplied by `μ` so non-zero entries are exactly the active gates.

`recon_mse` uses mean-over-batch-and-features (standard for reporting).
The model-internal training loss uses sum-over-features mean-over-batch
because λ/β values from the SAE/VAE literature assume that convention —
that calculation lives in each model's `loss()` method, not here.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def recon_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error averaged over both batch and feature dims."""
    return (x - x_hat).pow(2).mean()


def fve(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    x_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of variance explained: 1 − SSE / SST.

    `x_mean` is the per-coordinate mean over the full eval set. Pass it
    explicitly when computing FVE batch-by-batch — using the batch mean
    biases the score upward on small batches.
    """
    if x_mean is None:
        x_mean = x.mean(dim=0, keepdim=True)
    sse = (x - x_hat).pow(2).sum()
    sst = (x - x_mean).pow(2).sum().clamp_min(1e-12)
    return 1.0 - sse / sst


# ----------------------------------------------------------------------
# Sparsity
# ----------------------------------------------------------------------
def l0(z: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Mean number of "active" features per example.

    `threshold=0.0` checks z != 0 (correct for ReLU codes and for hard-gated
    spike-slab codes). `threshold>0` checks |z| > threshold (use for signed
    Gaussian-VAE codes — see module docstring for the chosen value).
    """
    if threshold == 0.0:
        active = z != 0
    else:
        active = z.abs() > threshold
    return active.sum(dim=-1).float().mean()


def firing_rate(
    z_batches: Iterable[torch.Tensor],
    d_dict: int,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Per-feature firing rate over an iterable of code batches.

    Streaming — never materializes the whole eval set in memory. Returns
    a (d_dict,) float tensor in [0, 1].
    """
    total_fires = torch.zeros(d_dict)
    total_count = 0
    for z in z_batches:
        active = (z.abs() > threshold) if threshold != 0.0 else (z != 0)
        total_fires += active.sum(dim=0).cpu().float()
        total_count += z.shape[0]
    return total_fires / max(total_count, 1)


def dead_features(
    z_batches: Iterable[torch.Tensor],
    d_dict: int,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Fraction of features that never fired across `z_batches`.

    Note: in the trainer we maintain a running per-feature firing counter
    over a sliding window (cfg.resample_window) rather than re-iterating —
    this function is for end-of-run eval reporting.
    """
    rates = firing_rate(z_batches, d_dict=d_dict, threshold=threshold)
    return (rates == 0).float().mean()


# ----------------------------------------------------------------------
# Posterior entropy
# ----------------------------------------------------------------------
_GAUSS_ENT_CONST = 0.5 * (1.0 + np.log(2 * np.pi))


def posterior_entropy_gauss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """Differential entropy of a diagonal-Gaussian posterior.

    H[N(μ, σ²)] per dim = 0.5·(1 + log 2π) + log σ.  Summed over the dict
    dim and averaged over the batch. `mu` is unused in the formula (the
    entropy of a Gaussian is location-invariant) but kept in the signature
    for symmetry with the spike-slab variant.
    """
    del mu
    h_per_dim = _GAUSS_ENT_CONST + log_sigma
    return h_per_dim.sum(dim=-1).mean()


def posterior_entropy_spikeslab(
    gate_prob: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Entropy of q(s, z|x) = q(s|x) · q(z|s, x).

    H[s_i] + γ_i · H[N(μ_i, σ_i²)]  per dim, summed over the dict, averaged
    over the batch. The slab contributes nothing when the gate is closed,
    so its entropy is weighted by γ (the gate prob). `mu` is unused for the
    same reason as in `posterior_entropy_gauss`.
    """
    del mu
    bern_h = -(
        gate_prob * (gate_prob + eps).log()
        + (1 - gate_prob) * (1 - gate_prob + eps).log()
    )
    slab_h = _GAUSS_ENT_CONST + log_sigma
    total = (bern_h + gate_prob * slab_h).sum(dim=-1)
    return total.mean()


# ----------------------------------------------------------------------
# Decoder geometry
# ----------------------------------------------------------------------
def decoder_cosine_overlap(W_a: torch.Tensor, W_b: torch.Tensor) -> dict:
    """One-to-one column matching between two decoders, by absolute cosine.

    Both inputs are shape (d_model, d_dict). We assign columns of W_a to
    columns of W_b by Hungarian-matching on `-|cosine|` so flipped-sign
    matches still count (a unit vector and its negative encode the same
    direction in the latent space). Used for Exp 3.2 to measure decoder
    drift across the π sweep.

    Returns a dict with the mean and median matched cosine plus the
    underlying assignment.
    """
    # Local import so the foundation modules don't drag in scipy at import time.
    from scipy.optimize import linear_sum_assignment

    Wa = W_a / W_a.norm(dim=0, keepdim=True).clamp_min(1e-8)
    Wb = W_b / W_b.norm(dim=0, keepdim=True).clamp_min(1e-8)
    cos = (Wa.t() @ Wb).abs().detach().cpu().numpy()  # (d_dict_a, d_dict_b)
    row_ind, col_ind = linear_sum_assignment(-cos)
    matched = cos[row_ind, col_ind]
    return {
        "mean_matched_cosine": float(matched.mean()),
        "median_matched_cosine": float(np.median(matched)),
        "matched_per_col": matched,
        "row_ind": row_ind,
        "col_ind": col_ind,
    }
