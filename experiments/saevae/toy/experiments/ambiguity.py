"""Experiment 2: Ambiguity — does the method admit uncertainty or commit?

Question: For an input that lies "between" two similar ground-truth features,
does the method commit to one feature (like SAE ReLU must) or spread
uncertainty across both (like a VAE posterior can)?

Setup
------
* Find the two ground-truth features fi, fj with the most similar 2-D
  directions (max |cosine| of their W columns, off-diagonal).
* Interpolate 200 inputs: direction d = (1-t)*W[:,fi] + t*W[:,fj], normalised,
  times magnitude 0.5; t in [0, 1].
* Measure activation entropy at the midpoint (t=0.5): higher = more honest
  acknowledgement of uncertainty.

Run
---
    cd /Users/dhawaldixit/projects/saes-as-vaes/experiments
    python -m saevae.toy.experiments.ambiguity

Output
------
    figures/toy/ambiguity.pdf  + .png
    (4 panels: SAE z per feature; SAE credit stackplot;
               VAE |mu| per feature with ±sigma bands;
               VAE sigma along t)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saevae.toy import (
    ToyModelConfig, ToyModel, ToySAE, LaplaceVAE,
    train_toy_model, train_sae, train_vae, align_features,
)

# ---------------------------------------------------------------------------
# Constants (identical to calibration.py)
# ---------------------------------------------------------------------------
N_INST = 1
N_FEATURES = 5
D_HIDDEN = 2
SEED = 42
FEATURE_PROB = 0.1

SAE_SPARSITY = 0.1
SAE_STEPS = 5000

VAE_BETA = 0.05
VAE_LAPLACE_B = 0.3
VAE_STEPS = 10000
VAE_KL_WARMUP = 3000

N_INTERP = 200
INTERP_MAG = 0.5  # input magnitude along the interpolated direction


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = ToyModelConfig(n_inst=N_INST, n_features=N_FEATURES, d_hidden=D_HIDDEN)
    model = train_toy_model(cfg, feature_probability=FEATURE_PROB, steps=5000,
                             tie_instances=True, progress=True)

    sae = ToySAE(n_inst=N_INST, d_in=D_HIDDEN, d_sae=N_FEATURES,
                  sparsity_coeff=SAE_SPARSITY)
    train_sae(model, sae, steps=SAE_STEPS, progress=True)

    vae = LaplaceVAE(n_inst=N_INST, d_in=D_HIDDEN, d_latent=N_FEATURES,
                      beta=VAE_BETA, laplace_b=VAE_LAPLACE_B)
    train_vae(model, vae, steps=VAE_STEPS, kl_warmup=VAE_KL_WARMUP, progress=True)

    return model, sae, vae


# ---------------------------------------------------------------------------
# Find most-similar feature pair
# ---------------------------------------------------------------------------

def most_similar_pair(W: np.ndarray) -> tuple[int, int, float]:
    """W: (d_hidden, n_features). Returns (fi, fj, cosine)."""
    W_n = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)  # (d, n)
    cos_mat = W_n.T @ W_n  # (n, n)
    n = cos_mat.shape[0]
    best_cos, best_i, best_j = -1.0, 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(float(cos_mat[i, j]))
            if c > best_cos:
                best_cos, best_i, best_j = c, i, j
    return best_i, best_j, best_cos


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def activation_entropy(activations: np.ndarray) -> float:
    """Shannon entropy of a normalised activation distribution.

    activations: 1-D array of non-negative values (one per feature at a point).
    Normalises to a probability distribution and computes -sum p log p.
    """
    a = np.array(activations, dtype=float)
    a = np.maximum(a, 0.0)
    total = a.sum()
    if total <= 1e-8:
        # No latent fires: the method is SILENT, not "maximally spread". These are
        # categorically different, so we return NaN rather than log(n). Callers
        # must report this as "silent" explicitly.
        return float("nan")
    p = a / total
    # Avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return float(-(p * logp).sum())


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------

def run_ambiguity():
    # ---- Train ----
    model, sae, vae = train_models()

    # ---- Alignment ----
    sae_align = align_features(model, sae, inst=0)
    vae_align = align_features(model, vae, inst=0)
    sae_perm = sae_align["perm"]
    sae_signs = sae_align["signs"]
    vae_perm = vae_align["perm"]
    vae_signs = vae_align["signs"]

    # ---- Ground-truth W ----
    W = model.W[0].detach().cpu().numpy()  # (d_hidden, n_features)

    # ---- Most similar pair ----
    fi, fj, pair_cos = most_similar_pair(W)
    print(f"\nMost similar feature pair: ({fi}, {fj})  |cosine| = {pair_cos:.4f}")

    # ---- Build interpolated inputs ----
    ts = np.linspace(0.0, 1.0, N_INTERP)
    W_fi = W[:, fi]
    W_fj = W[:, fj]

    hiddens = []
    for t in ts:
        d = (1 - t) * W_fi + t * W_fj
        norm = np.linalg.norm(d) + 1e-8
        h_vec = (d / norm) * INTERP_MAG
        hiddens.append(h_vec)

    h_np = np.stack(hiddens, axis=0).astype(np.float32)  # (N_INTERP, d_hidden)
    # Add inst dim: (N_INTERP, 1, d_hidden)
    h_tensor = torch.from_numpy(h_np[:, None, :])

    # ---- SAE forward ----
    sae.eval()
    with torch.no_grad():
        _, z_sae, _ = sae(h_tensor)  # z_sae: (N_INTERP, 1, d_sae)
    z_sae_np = z_sae[:, 0, :].cpu().numpy()  # (N_INTERP, d_sae)

    # Reorder into ground-truth order for the PLOTS only. Unlike calibration, we
    # do NOT zero anti-aligned latents here: the ambiguity question is whether the
    # SAE spreads or commits its ACTUAL activation, so we must show the real code.
    z_ordered = np.zeros((N_INTERP, N_FEATURES), dtype=np.float32)
    for i in range(N_FEATURES):
        lat = sae_perm[i]
        if lat >= 0:
            z_ordered[:, i] = z_sae_np[:, lat]

    # ---- VAE forward ----
    vae.eval()
    with torch.no_grad():
        _, _, mu_vae, logvar_vae, _ = vae(h_tensor)
    mu_np = mu_vae[:, 0, :].cpu().numpy()       # (N_INTERP, d_latent)
    logvar_np = logvar_vae[:, 0, :].cpu().numpy()
    sigma_np = np.exp(0.5 * logvar_np)           # (N_INTERP, d_latent)

    # Reorder into gt order; flip sign of mu for anti-aligned latents (for the
    # VAE posterior, mu can be negative, so signs[i]==-1 means we should negate
    # to get a sensible feature-presence signal; |mu| after this fix = |mu|
    # regardless, but sign-awareness is important if one looks at signed mu).
    mu_ordered = np.zeros((N_INTERP, N_FEATURES), dtype=np.float32)
    sigma_ordered = np.zeros((N_INTERP, N_FEATURES), dtype=np.float32)
    for i in range(N_FEATURES):
        lat = vae_perm[i]
        if lat >= 0:
            mu_ordered[:, i] = mu_np[:, lat] * vae_signs[i]
            sigma_ordered[:, i] = sigma_np[:, lat]

    abs_mu_ordered = np.abs(mu_ordered)

    # ---- Entropy at midpoint (t = 0.5, index N_INTERP // 2) ----
    mid = N_INTERP // 2

    # The "commit vs spread" question is about each method's OWN code distribution,
    # so compute entropy on the raw latents (no alignment, no zeroing). Alignment
    # is only for the plot labels.
    sae_raw_mid = np.maximum(z_sae_np[mid], 0.0)   # SAE ReLU codes, all latents
    vae_raw_mid = np.abs(mu_np[mid])               # VAE |posterior mean|, all latents
    sae_total_mid = float(sae_raw_mid.sum())
    vae_total_mid = float(vae_raw_mid.sum())

    sae_entropy = activation_entropy(sae_raw_mid)
    vae_entropy = activation_entropy(vae_raw_mid)
    max_entropy = float(np.log(N_FEATURES))

    def _ent_str(e, total):
        if np.isnan(e):
            return f"SILENT (total activation {total:.2e}, no latent fires)"
        return f"{e:.4f}  (max possible: {max_entropy:.4f})"

    print(f"Midpoint (t=0.5) activation entropy (over each method's own latents):")
    print(f"  SAE: {_ent_str(sae_entropy, sae_total_mid)}")
    print(f"  VAE: {_ent_str(vae_entropy, vae_total_mid)}")
    print(f"SAE raw codes at midpoint: {sae_raw_mid}")
    print(f"VAE |mu|    at midpoint: {vae_raw_mid}")

    # ---- Plot ----
    colors = plt.cm.tab10(np.linspace(0, 1, N_FEATURES))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Toy Model — Ambiguity: interpolation between features {fi} and {fj}\n"
        f"(|cosine| = {pair_cos:.3f}; magnitude={INTERP_MAG})",
        fontsize=12, fontweight="bold",
    )

    # --- Panel 0: SAE z per feature ---
    ax = axes[0, 0]
    for i in range(N_FEATURES):
        lw = 2.5 if i in (fi, fj) else 1.0
        ls = "-" if i in (fi, fj) else "--"
        label = f"F{i}" + (" (fi)" if i == fi else (" (fj)" if i == fj else ""))
        ax.plot(ts, z_ordered[:, i], color=colors[i], lw=lw, ls=ls, label=label)
    ax.axvline(0.5, color="gray", lw=1, ls=":", label="midpoint")
    ax.set_xlabel("Interpolation t (0=fi, 1=fj)")
    ax.set_ylabel("SAE activation z (ReLU)")
    ax.set_title("SAE: per-feature activations")
    ax.legend(fontsize=8)

    # --- Panel 1: SAE credit stackplot ---
    ax = axes[0, 1]
    total = z_ordered.sum(axis=1, keepdims=True)
    safe_total = np.where(total < 1e-8, 1.0, total)
    fracs = z_ordered / safe_total  # (N_INTERP, N_FEATURES)
    ax.stackplot(ts, fracs.T, labels=[f"F{i}" for i in range(N_FEATURES)],
                  colors=colors)
    ax.axvline(0.5, color="white", lw=1.5, ls=":")
    ax.set_xlabel("Interpolation t")
    ax.set_ylabel("Fraction of total activation")
    ax.set_title(f"SAE: credit allocation\n(entropy at midpoint: {sae_entropy:.3f} / {max_entropy:.3f})")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1)

    # --- Panel 2: VAE |mu| with ±sigma bands ---
    ax = axes[1, 0]
    for i in range(N_FEATURES):
        lw = 2.5 if i in (fi, fj) else 1.0
        ls = "-" if i in (fi, fj) else "--"
        label = f"F{i}" + (" (fi)" if i == fi else (" (fj)" if i == fj else ""))
        line, = ax.plot(ts, abs_mu_ordered[:, i], color=colors[i], lw=lw, ls=ls, label=label)
        if i in (fi, fj):
            ax.fill_between(ts,
                             abs_mu_ordered[:, i] - sigma_ordered[:, i],
                             abs_mu_ordered[:, i] + sigma_ordered[:, i],
                             color=colors[i], alpha=0.2)
    ax.axvline(0.5, color="gray", lw=1, ls=":", label="midpoint")
    ax.set_xlabel("Interpolation t")
    ax.set_ylabel("VAE |posterior mean|")
    ax.set_title("VAE: |mu| per feature  (±sigma shaded for fi, fj)")
    ax.legend(fontsize=8)

    # --- Panel 3: VAE posterior sigma for fi and fj ---
    ax = axes[1, 1]
    for i in [fi, fj]:
        label = f"F{i} sigma"
        ax.plot(ts, sigma_ordered[:, i], color=colors[i], lw=2.5, label=label)
    ax.axvline(0.5, color="gray", lw=1, ls=":", label="midpoint")
    ax.set_xlabel("Interpolation t")
    ax.set_ylabel("Posterior std dev (sigma)")
    ax.set_title(f"VAE: posterior sigma\n(entropy at midpoint: {vae_entropy:.3f} / {max_entropy:.3f})")
    ax.legend(fontsize=9)

    fig.tight_layout()

    os.makedirs("figures/toy", exist_ok=True)
    fig.savefig("figures/toy/ambiguity.pdf", bbox_inches="tight")
    fig.savefig("figures/toy/ambiguity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved figures/toy/ambiguity.pdf + .png")

    return {
        "fi": fi, "fj": fj, "pair_cosine": pair_cos,
        "sae_entropy_midpoint": sae_entropy,
        "vae_entropy_midpoint": vae_entropy,
        "max_entropy": max_entropy,
    }


if __name__ == "__main__":
    run_ambiguity()
