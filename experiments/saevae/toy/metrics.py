"""Evaluation + feature alignment for the toy models.

This module is where the two notebook bugs are fixed:

1. **L0 metric.** The notebooks thresholded activations at ``1% of the single
   largest activation in the model``. That number is not comparable across
   families (SAE ReLU codes are large; VAE posterior means are small) and it
   reads L0=0 for a collapsed spike-slab without saying so. We replace it with
   the per-family conventions used on the GPT-2 side:
     - SAE              : (z > 0)                  -- ReLU support
     - Gaussian/Laplace : (|mu| > gauss_threshold) -- threshold-dependent, declared
     - Spike-slab       : hard gate (gate_logit>0); also reports expected_l0 = E[gate]
   This makes the toy Pareto axis commensurable with the GPT-2 Pareto axis.

2. **Feature alignment sign.** Decoder column signs are arbitrary, so matching
   by ``|cosine|`` alone can pair a latent that encodes the *negative* of a
   ground-truth direction, which silently inverts any downstream detection
   score. ``align_features`` now also returns the matched sign so callers can
   flip, and matching is done greedily on descending ``|cosine|`` with
   uniqueness (a small fixed n_features makes this equivalent to Hungarian).
"""

from __future__ import annotations

import numpy as np
import torch

from .models import GaussianVAE, LaplaceVAE, SpikeSlabVAE, ToySAE


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------
@torch.no_grad()
def align_features(model, method, inst: int = 0):
    """Match each ground-truth feature to its closest learned latent.

    Returns a dict with:
      perm  : list[int]   -- perm[i] = learned latent index matched to gt feature i
      signs : list[float] -- sign (+1/-1) of the matched cosine (decoder col sign is arbitrary)
      cos   : list[float] -- the matched |cosine| similarity (alignment quality)

    Greedy on descending |cosine| with uniqueness on both sides.
    """
    W_true = model.W[inst].detach()                       # (d_hidden, n_features)
    W_true_n = W_true / (W_true.norm(dim=0, keepdim=True) + 1e-8)

    if isinstance(method, ToySAE):
        W_dec = method.W_dec_normalized[inst].detach()    # (d_latent, d_in)
    else:
        W_dec = method.W_dec[inst].detach()
    W_dec_n = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-8)

    # signed cosine[i, j] = gt feature i  .  learned latent j
    cos = W_true_n.T @ W_dec_n.T                          # (n_features, d_latent)
    n_feat, n_lat = cos.shape

    perm = [-1] * n_feat
    signs = [1.0] * n_feat
    cosq = [0.0] * n_feat

    # rank all (i, j) pairs by |cosine|, assign greedily without reuse
    order = torch.argsort(cos.abs().flatten(), descending=True)
    used_feat, used_lat = set(), set()
    for flat in order.tolist():
        i, j = divmod(flat, n_lat)
        if i in used_feat or j in used_lat:
            continue
        perm[i] = j
        signs[i] = float(torch.sign(cos[i, j]).item()) or 1.0
        cosq[i] = float(cos[i, j].abs().item())
        used_feat.add(i)
        used_lat.add(j)
        if len(used_feat) == n_feat:
            break
    return {"perm": perm, "signs": signs, "cos": cosq}


# ---------------------------------------------------------------------------
# L0 by family (fixed, per-family conventions)
# ---------------------------------------------------------------------------
@torch.no_grad()
def activations_for_l0(method, results, gauss_threshold: float = 0.1):
    """Return a per-latent activation indicator tensor (batch, inst, d_latent) for L0.

    SAE: z>0; Gaussian/Laplace: |mu|>gauss_threshold; SpikeSlab: hard gate.
    """
    if isinstance(method, ToySAE):
        z = results[1]
        return (z > 0).float()
    if isinstance(method, SpikeSlabVAE):
        gate = results[2]            # hard 0/1 at eval (method.eval())
        return (gate > 0.5).float()
    # Gaussian / Laplace VAE: posterior mean magnitude
    mu = results[2]
    return (mu.abs() > gauss_threshold).float()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, method, n_eval: int = 10000, gauss_threshold: float = 0.1):
    """Reconstruction MSE and (fixed) L0 sparsity for any method.

    Returns dict of per-instance numpy arrays:
      mse         : (n_inst,)
      l0          : (n_inst,)  -- per-family hard L0
      expected_l0 : (n_inst,)  -- spike-slab only: E[gate] summed over latents; else == l0
    """
    h, _ = model.get_hidden_activations(n_eval)
    was_training = method.training
    method.eval()
    results = method(h)
    h_recon = results[0]

    mse = (h_recon - h).pow(2).mean(dim=-1).mean(dim=0)         # (n_inst,)
    indic = activations_for_l0(method, results, gauss_threshold)
    l0 = indic.sum(dim=-1).mean(dim=0)                          # (n_inst,)

    if isinstance(method, SpikeSlabVAE):
        gate_logits, _, _ = method.encode(h)
        expected_l0 = torch.sigmoid(gate_logits).sum(dim=-1).mean(dim=0)
    else:
        expected_l0 = l0

    if was_training:
        method.train()
    return {
        "mse": mse.cpu().numpy(),
        "l0": l0.cpu().numpy(),
        "expected_l0": expected_l0.cpu().numpy(),
    }
