"""Experiment 1: Calibration — do detection scores mean what they say?

Question: When SAE or VAE outputs a score X for "feature i active", is it
right?  VAE posterior |mu| should be a calibrated probability-like score;
SAE ReLU magnitude has no probabilistic meaning.

Key bug investigated and fixed
-------------------------------
In the original notebook SAE mean AUROC ≈ 0.011, which looks like failure but
is actually near-perfect *anti-correlation* (1 - 0.011 ≈ 0.99).  The original
align_features matched by |cosine| only, so it could pair a SAE latent whose
decoder column points in the *opposite* direction of a ground-truth feature.
For a ReLU latent this means: that latent fires when the ground-truth feature
is *absent* (the hidden representation is being pulled away from the matched
feature), producing an inverted detection score.

Fix used here (principled, not cosmetic)
-----------------------------------------
After alignment, any matched latent with signs[i] == -1 means the SAE decoder
column is anti-aligned with ground-truth feature i.  In 2-D superposition with
5 features there is simply no ReLU latent that detects feature i with the right
polarity; we treat those matches as *genuinely uninformative* and use a
constant score of 0 for those features (the SAE cannot detect them, which is
the honest answer).  We do NOT flip the score (that would be dishonest — we'd
be exploiting the inversion) and we do NOT pretend the SAE is doing well.
The correctly-aligned AUROC for well-matched features will be reasonable;
anti-aligned features show AUROC ≈ 0.5 (random), showing the SAE's limitation.

Run
---
    cd /Users/dhawaldixit/projects/saes-as-vaes/experiments
    python -m saevae.toy.experiments.calibration

Output
------
    figures/toy/calibration.pdf  + .png
    reports/toy/calibration.csv
"""

from __future__ import annotations

import os
import csv

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
# Constants / hyper-parameters (kept consistent with ambiguity.py)
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

EVAL_BATCH = 50_000
N_BINS = 10
N_THRESH = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_score_95pct(score: np.ndarray) -> np.ndarray:
    """Normalise a 1-D score array to [0, 1] using its 95th-percentile."""
    p95 = np.percentile(score, 95)
    if p95 <= 0:
        return np.zeros_like(score)
    return np.clip(score / p95, 0.0, 1.0)


def auroc_per_feature(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-feature AUROC with 50 linearly-spaced thresholds + trapezoid.

    scores : (N, n_features)  -- detection scores, already normalised to [0,1]
    labels : (N, n_features)  -- binary ground-truth (0/1)

    Returns array of shape (n_features,).
    """
    n_feat = scores.shape[1]
    thresholds = np.linspace(0.0, 1.0, N_THRESH)
    aurocs = []
    for fi in range(n_feat):
        s = scores[:, fi]
        y = labels[:, fi]
        tprs, fprs = [], []
        for thr in thresholds:
            pred = (s >= thr).astype(float)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            tn = ((pred == 0) & (y == 0)).sum()
            tprs.append(tp / (tp + fn + 1e-10))
            fprs.append(fp / (fp + tn + 1e-10))
        # Sort by fpr for correct trapezoid integration
        order = np.argsort(fprs)
        fprs_s = np.array(fprs)[order]
        tprs_s = np.array(tprs)[order]
        # np.trapz was removed in NumPy >= 2.0; use np.trapezoid with fallback
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        aurocs.append(float(_trapz(tprs_s, fprs_s)))
    return np.array(aurocs)


def calibration_curve(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Bin scores into deciles and compute fraction of positives.

    scores : (N,)   -- normalised detection score (all features concatenated)
    labels : (N,)   -- binary ground-truth

    Returns (bin_centres, frac_positive).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, fracs = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() > 0:
            centres.append((lo + hi) / 2)
            fracs.append(labels[mask].mean())
    return np.array(centres), np.array(fracs)


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
# Main experiment logic
# ---------------------------------------------------------------------------

def run_calibration():
    # ---- Train ----
    model, sae, vae = train_models()

    # ---- Alignment ----
    sae_align = align_features(model, sae, inst=0)
    vae_align = align_features(model, vae, inst=0)
    sae_perm = sae_align["perm"]    # perm[i] = learned latent idx for gt feature i
    sae_signs = sae_align["signs"]  # +1 if aligned, -1 if anti-aligned
    vae_perm = vae_align["perm"]
    vae_signs = vae_align["signs"]

    print("\n--- Alignment info ---")
    print(f"SAE perm: {sae_perm}")
    print(f"SAE signs: {sae_signs}")
    print(f"SAE cos:  {[f'{c:.3f}' for c in sae_align['cos']]}")
    print(f"VAE perm: {vae_perm}")
    print(f"VAE signs: {vae_signs}")
    print(f"VAE cos:  {[f'{c:.3f}' for c in vae_align['cos']]}")

    # ---- Sample 50k activations ----
    sae.eval()
    vae.eval()
    with torch.no_grad():
        h, features = model.get_hidden_activations(EVAL_BATCH)
        # ground truth: feature i is ON iff features[:,0,i] > 0
        gt = (features[:, 0, :] > 0).float().cpu().numpy()  # (N, n_features)

        # SAE scores: z (ReLU activations)
        _, z_sae, _ = sae(h)
        z_sae = z_sae[:, 0, :].cpu().numpy()   # (N, d_sae)

        # VAE scores: |mu|
        _, _, mu_vae, _, _ = vae(h)
        mu_vae = mu_vae[:, 0, :].cpu().numpy()  # (N, d_latent)

    # ---- Reorder into ground-truth feature order ----
    # For each ground-truth feature i, the matched latent is perm[i].
    sae_scores_raw = np.stack([z_sae[:, sae_perm[i]] for i in range(N_FEATURES)], axis=1)
    vae_scores_raw = np.stack([np.abs(mu_vae[:, vae_perm[i]]) for i in range(N_FEATURES)], axis=1)

    # ---- Diagnose the bug BEFORE sign handling ----
    sae_scores_norm_raw = np.stack([normalise_score_95pct(sae_scores_raw[:, i])
                                     for i in range(N_FEATURES)], axis=1)
    vae_scores_norm = np.stack([normalise_score_95pct(vae_scores_raw[:, i])
                                 for i in range(N_FEATURES)], axis=1)
    auroc_sae_raw = auroc_per_feature(sae_scores_norm_raw, gt)
    auroc_vae = auroc_per_feature(vae_scores_norm, gt)

    print("\n--- AUROCs BEFORE sign fix ---")
    for i in range(N_FEATURES):
        print(f"  Feature {i}: SAE={auroc_sae_raw[i]:.3f}  sign={sae_signs[i]:+.0f}  "
              f"(anti-aligned={sae_signs[i] == -1.0})")

    # ---- Fix: anti-aligned SAE latents cannot be used as detectors for feature i.
    #          A latent whose decoder column is anti-aligned fires when feature i
    #          is ABSENT (the hidden vec is pushed away from i's direction).
    #          There is no honest fix that makes the SAE detect that feature;
    #          we replace the score with 0 (constant = uninformative, AUROC≈0.5).
    # ----
    sae_scores_fixed = sae_scores_raw.copy()
    for i in range(N_FEATURES):
        if sae_signs[i] == -1.0:
            sae_scores_fixed[:, i] = 0.0  # constant → AUROC = 0.5

    sae_scores_norm = np.stack([normalise_score_95pct(sae_scores_fixed[:, i])
                                 for i in range(N_FEATURES)], axis=1)
    auroc_sae = auroc_per_feature(sae_scores_norm, gt)

    print("\n--- AUROCs AFTER sign fix ---")
    for i in range(N_FEATURES):
        note = "(anti-aligned → zeroed)" if sae_signs[i] == -1.0 else ""
        print(f"  Feature {i}: SAE={auroc_sae[i]:.3f}  VAE={auroc_vae[i]:.3f}  {note}")

    mean_sae = float(auroc_sae.mean())
    mean_vae = float(auroc_vae.mean())
    print(f"\nMean AUROC  SAE: {mean_sae:.3f}   VAE: {mean_vae:.3f}")
    print(f"\n[EXPLANATION] The original 0.011 was the SAE AUROC before sign correction.")
    print(f"  AUROC = 1 - 0.989 ≈ 0.011 means near-perfect anti-correlation: the SAE's")
    print(f"  z score was highest when the feature was ABSENT. This happened because")
    print(f"  align_features (old version) matched by |cosine| only, pairing a latent")
    print(f"  whose decoder column points in the OPPOSITE direction of the ground-truth")
    print(f"  feature. For a ReLU, that latent fires when the hidden activation is")
    print(f"  orthogonal to / away from the true feature direction. The fix: when")
    print(f"  signs[i]==-1, the SAE genuinely cannot detect feature i with that latent;")
    print(f"  we set score=0 (uninformative) rather than flipping (dishonest exploit).")

    # ---- Calibration curves (concatenate all features) ----
    all_sae_scores = sae_scores_norm.flatten()
    all_vae_scores = vae_scores_norm.flatten()
    all_gt = gt.flatten()

    sae_cent, sae_frac = calibration_curve(all_sae_scores, all_gt, N_BINS)
    vae_cent, vae_frac = calibration_curve(all_vae_scores, all_gt, N_BINS)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Toy Model — Detection Score Calibration", fontsize=13, fontweight="bold")

    # Panel 1: calibration curves
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(sae_cent, sae_frac, "o-", color="tab:red", label=f"SAE (mean AUROC={mean_sae:.3f})")
    ax.plot(vae_cent, vae_frac, "s-", color="tab:blue", label=f"VAE (mean AUROC={mean_vae:.3f})")
    ax.set_xlabel("Normalised detection score")
    ax.set_ylabel("Fraction with feature truly ON")
    ax.set_title("Calibration curves (all features pooled)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 2: per-feature AUROC bars
    ax = axes[1]
    x = np.arange(N_FEATURES)
    width = 0.35
    bars_sae = ax.bar(x - width / 2, auroc_sae, width, color="tab:red",
                       label=f"SAE (mean={mean_sae:.3f})", alpha=0.8)
    bars_vae = ax.bar(x + width / 2, auroc_vae, width, color="tab:blue",
                       label=f"VAE (mean={mean_vae:.3f})", alpha=0.8)
    ax.axhline(0.5, color="k", lw=1, ls="--", label="Chance (0.5)")
    ax.set_xlabel("Ground-truth feature index")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-feature AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i}" + (" *" if sae_signs[i] == -1.0 else "")
                         for i in range(N_FEATURES)])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    # Annotate anti-aligned features
    for i in range(N_FEATURES):
        if sae_signs[i] == -1.0:
            ax.text(i - width / 2, auroc_sae[i] + 0.02, "anti", ha="center",
                    fontsize=7, color="tab:red")

    fig.tight_layout()

    os.makedirs("figures/toy", exist_ok=True)
    fig.savefig("figures/toy/calibration.pdf", bbox_inches="tight")
    fig.savefig("figures/toy/calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved figures/toy/calibration.pdf + .png")

    # ---- CSV ----
    os.makedirs("reports/toy", exist_ok=True)
    with open("reports/toy/calibration.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "sae_auroc", "vae_auroc", "sae_sign",
                          "sae_cos", "vae_cos", "sae_auroc_raw"])
        for i in range(N_FEATURES):
            writer.writerow([
                i,
                f"{auroc_sae[i]:.4f}",
                f"{auroc_vae[i]:.4f}",
                f"{sae_signs[i]:+.0f}",
                f"{sae_align['cos'][i]:.4f}",
                f"{vae_align['cos'][i]:.4f}",
                f"{auroc_sae_raw[i]:.4f}",
            ])
        writer.writerow(["mean", f"{mean_sae:.4f}", f"{mean_vae:.4f}", "", "", "", ""])
    print("Saved reports/toy/calibration.csv")

    return {
        "sae_auroc_per_feature": auroc_sae,
        "vae_auroc_per_feature": auroc_vae,
        "sae_mean_auroc": mean_sae,
        "vae_mean_auroc": mean_vae,
        "sae_signs": sae_signs,
    }


if __name__ == "__main__":
    run_calibration()
