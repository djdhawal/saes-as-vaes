"""Shared loss helpers and schedulers.

Closed-form KL divergences and the annealing schedules from the plan live
here so each model class stays focused on the architecture / sampling part.
"""

from __future__ import annotations

import math

import torch


# ----------------------------------------------------------------------
# Schedulers
# ----------------------------------------------------------------------
def linear_warmup(step: int, warmup_steps: int) -> float:
    """0.0 → 1.0 linearly over `warmup_steps`. Returns 1.0 after.

    Use for KL warmup (β_eff = β · linear_warmup(step, 5000)) and for the
    lr warmup at the start of training.
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / float(warmup_steps))


def gumbel_tau_schedule(
    step: int,
    total_steps: int,
    tau_init: float = 1.0,
    tau_min: float = 0.3,
) -> float:
    """Exponential anneal from `tau_init` at step 0 to `tau_min` at
    `total_steps`. Stays at `tau_min` after. Used by the spike-slab gate;
    going below ~0.1 introduces gradient explosion, hence the floor."""
    if total_steps <= 0 or step >= total_steps:
        return tau_min
    decay_ratio = tau_min / tau_init
    return tau_init * (decay_ratio ** (step / total_steps))


# ----------------------------------------------------------------------
# Closed-form KL divergences
# ----------------------------------------------------------------------
def gaussian_kl_to_standard(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL[ N(μ, diag σ²) || N(0, I) ] summed over the last dim and
    averaged over the batch.  Per-dim: 0.5·(μ² + σ² − 1 − log σ²).
    """
    kl = 0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var)
    return kl.sum(dim=-1).mean()


def bernoulli_kl(
    gate_prob: torch.Tensor,
    prior_pi: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """KL[ Bernoulli(γ) || Bernoulli(π) ] per dim, summed over dict,
    averaged over batch. `prior_pi` is a scalar (the same π for every
    latent dim — we don't currently learn a per-dim prior)."""
    g = gate_prob
    kl = (
        g * ((g + eps).log() - math.log(prior_pi))
        + (1.0 - g) * ((1.0 - g + eps).log() - math.log(1.0 - prior_pi))
    )
    return kl.sum(dim=-1).mean()


def spike_slab_kl(
    gate_prob: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    prior_pi: float,
) -> dict[str, torch.Tensor]:
    """KL[ q(s,z|x) || p(s,z) ] for the spike-and-slab model.

    Decomposes (closed-form, no Monte Carlo) into:
      kl_gate = KL[ Bernoulli(γ) || Bernoulli(π) ]
      kl_slab = γ · KL[ N(μ, σ²) || N(0, 1) ]     (slab matters only when s=1)
    Returns both terms plus their sum so trainers can log them separately.
    """
    kl_gate = bernoulli_kl(gate_prob, prior_pi)
    # Per-dim slab KL, gated by γ. Sum over dim, mean over batch.
    slab_per_dim = 0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var)
    kl_slab = (gate_prob * slab_per_dim).sum(dim=-1).mean()
    return {"kl_gate": kl_gate, "kl_slab": kl_slab, "kl": kl_gate + kl_slab}
