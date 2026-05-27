"""OOD detection: VAE ELBO/KL vs. SAE loss as out-of-distribution signals.

Experiment (from notebook 2, ``vae_beats_sae_experiments``):
    Both the SAE and VAE can detect OOD via reconstruction error, but only the
    VAE has the KL term — a dedicated, probabilistically-meaningful signal for
    "is the posterior being used unusually?".  The SAE has no such signal; its
    only lever is the recon loss (or, equivalently, the residual).

    We build four conditions of hidden activations from a trained toy model and
    score them with both a trained SAE and a trained LaplaceVAE:

        in-dist   : normal toy-model activations (feature_prob=0.1)
        rand_noise: Gaussian noise scaled to match in-dist std
        dense     : activations generated with feature_prob=0.8 (much denser)
        scaled_5x : in-dist activations amplified by 5x

    For each condition we record per-sample:
        SAE  → total loss (recon+L1), MSE
        VAE  → neg-ELBO (loss), KL, MSE

    Outputs
    -------
    figures/toy/ood.pdf + .png  — 2×2 boxplot/bar figure
    reports/toy/ood_auroc.csv   — AUROC table (each signal × each OOD condition)

GPT-2 portability note
----------------------
This experiment is directly portable to GPT-2 activations: treat a reference
layer's activations as "in-distribution", and use activations from a different
layer, shuffled activations, or activations scaled by a constant as OOD
conditions.  The VAE KL provides a calibrated, probabilistically-grounded OOD
signal at each activation site; an SAE at the same layer has no equivalent.
"""

from __future__ import annotations

import os
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

from saevae.toy import (
    ToyModelConfig,
    ToyModel,
    ToySAE,
    LaplaceVAE,
    train_toy_model,
    train_sae,
    train_vae,
    DEVICE,
)

# ---------------------------------------------------------------------------
# Paths (relative to the project root, created as needed)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parents[3]  # …/experiments
FIG_DIR = _ROOT / "figures" / "toy"
REP_DIR = _ROOT / "reports" / "toy"

# Condition labels, in display order
CONDITIONS = ["In-distribution", "Random noise", "Dense (80% active)", "Scaled 5x"]
COND_KEYS = ["in_dist", "rand_noise", "dense", "scaled_5x"]


# ---------------------------------------------------------------------------
# 1. Training
# ---------------------------------------------------------------------------

def build_and_train(n_samples: int = 5000):
    """Train the toy model, SAE, and LaplaceVAE; return all three."""
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = ToyModelConfig(n_inst=1, n_features=5, d_hidden=2)

    print("=== Training toy model (feature_prob=0.1, 5000 steps) ===")
    model = train_toy_model(cfg, feature_probability=0.1, steps=5000,
                            tie_instances=True, progress=True)
    model.eval()

    print("\n=== Training SAE (sparsity_coeff=0.1, 5000 steps) ===")
    sae = ToySAE(n_inst=1, d_in=2, d_sae=5, sparsity_coeff=0.1)
    train_sae(model, sae, steps=5000, progress=True)
    sae.eval()

    print("\n=== Training LaplaceVAE (beta=0.05, laplace_b=0.3, 10000 steps) ===")
    vae = LaplaceVAE(n_inst=1, d_in=2, d_latent=5, beta=0.05, laplace_b=0.3)
    train_vae(model, vae, steps=10000, kl_warmup=3000, progress=True)
    vae.eval()

    return model, sae, vae


# ---------------------------------------------------------------------------
# 2. Build OOD conditions
# ---------------------------------------------------------------------------

def build_conditions(model: ToyModel, n_samples: int = 5000) -> dict[str, torch.Tensor]:
    """Return a dict of h tensors, each shape (n_samples, 1, 2)."""
    with torch.no_grad():
        # In-distribution
        h_id, _ = model.get_hidden_activations(n_samples)

        # Random noise: same std as in-dist
        h_noise = torch.randn(n_samples, 1, 2) * h_id.std()

        # Dense (80% active): clone W and b_final into a new ToyModel
        cfg_dense = ToyModelConfig(n_inst=1, n_features=5, d_hidden=2)
        dense_model = ToyModel(cfg_dense, feature_probability=0.8, device=DEVICE)
        with torch.no_grad():
            dense_model.W.data.copy_(model.W.data)
            dense_model.b_final.data.copy_(model.b_final.data)
        dense_model.eval()
        h_dense, _ = dense_model.get_hidden_activations(n_samples)

        # Scaled 5x
        h_scaled = h_id * 5.0

    return {
        "in_dist": h_id,
        "rand_noise": h_noise,
        "dense": h_dense,
        "scaled_5x": h_scaled,
    }


# ---------------------------------------------------------------------------
# 3. Score each condition
# ---------------------------------------------------------------------------

def score_conditions(conditions: dict[str, torch.Tensor],
                     sae: ToySAE,
                     vae: LaplaceVAE) -> dict[str, dict[str, np.ndarray]]:
    """Return nested dict: scores[cond_key][signal_name] = 1-D np.ndarray."""
    scores: dict[str, dict[str, np.ndarray]] = {}

    sae.eval()
    vae.eval()

    with torch.no_grad():
        for key, h in conditions.items():
            h_dev = h.to(DEVICE)

            # SAE
            _, _, sae_dict = sae(h_dev)
            sae_loss = sae_dict["loss"][:, 0].cpu().numpy()   # recon + λ·L1
            sae_mse = sae_dict["mse"][:, 0].cpu().numpy()

            # VAE
            _, _, _, _, vae_dict = vae(h_dev)
            vae_loss = vae_dict["loss"][:, 0].cpu().numpy()   # neg-ELBO
            vae_kl = vae_dict["kl"][:, 0].cpu().numpy()
            vae_mse = vae_dict["mse"][:, 0].cpu().numpy()

            scores[key] = {
                "sae_loss": sae_loss,
                "sae_mse": sae_mse,
                "vae_loss": vae_loss,
                "vae_kl": vae_kl,
                "vae_mse": vae_mse,
            }

    return scores


# ---------------------------------------------------------------------------
# 4. AUROC (OOD = positive class; sweep ~200 thresholds; trapezoid rule)
# ---------------------------------------------------------------------------

def compute_auroc(in_scores: np.ndarray, ood_scores: np.ndarray,
                  n_thresholds: int = 200) -> float:
    """Trapezoid-rule AUROC treating OOD as positive.

    Higher score → more likely OOD.
    """
    all_scores = np.concatenate([in_scores, ood_scores])
    thresholds = np.linspace(all_scores.min() - 1e-9,
                             all_scores.max() + 1e-9,
                             n_thresholds)

    tprs = []
    fprs = []
    for thr in thresholds:
        tp = (ood_scores >= thr).mean()
        fp = (in_scores >= thr).mean()
        tprs.append(tp)
        fprs.append(fp)

    # Sort by FPR ascending for trapezoid
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    sort_idx = np.argsort(fprs_arr)
    fprs_sorted = fprs_arr[sort_idx]
    tprs_sorted = tprs_arr[sort_idx]

    auroc = float(np.trapezoid(tprs_sorted, fprs_sorted)
                  if hasattr(np, "trapezoid") else
                  np.trapz(tprs_sorted, fprs_sorted))
    return auroc


SIGNAL_NAMES = ["vae_loss", "vae_kl", "vae_mse", "sae_loss", "sae_mse"]
SIGNAL_LABELS = {
    "vae_loss": "VAE neg-ELBO",
    "vae_kl": "VAE KL",
    "vae_mse": "VAE MSE",
    "sae_loss": "SAE loss",
    "sae_mse": "SAE MSE",
}

OOD_KEYS = ["rand_noise", "dense", "scaled_5x"]
OOD_LABELS = {
    "rand_noise": "Random noise",
    "dense": "Dense (80% active)",
    "scaled_5x": "Scaled 5x",
}


def compute_auroc_table(scores: dict[str, dict[str, np.ndarray]]) -> dict:
    """Return auroc_table[signal][ood_key] = float."""
    table: dict[str, dict[str, float]] = {}
    in_scores_by_signal = scores["in_dist"]

    for signal in SIGNAL_NAMES:
        table[signal] = {}
        for ood_key in OOD_KEYS:
            auc = compute_auroc(in_scores_by_signal[signal],
                                scores[ood_key][signal])
            table[signal][ood_key] = auc

    return table


def print_auroc_table(table: dict) -> None:
    header = f"{'Signal':<18}" + "".join(f"{OOD_LABELS[k]:>22}" for k in OOD_KEYS)
    sep = "-" * len(header)
    print("\n" + sep)
    print("AUROC TABLE  (OOD = positive class; higher = better separation)")
    print(sep)
    print(header)
    print(sep)
    for signal in SIGNAL_NAMES:
        row = f"{SIGNAL_LABELS[signal]:<18}"
        for ood_key in OOD_KEYS:
            row += f"{table[signal][ood_key]:>22.4f}"
        print(row)
    print(sep + "\n")


def save_auroc_csv(table: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["signal"] + [OOD_LABELS[k] for k in OOD_KEYS])
        for signal in SIGNAL_NAMES:
            writer.writerow(
                [SIGNAL_LABELS[signal]] + [f"{table[signal][k]:.6f}" for k in OOD_KEYS]
            )


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------

def make_figure(scores: dict[str, dict[str, np.ndarray]], fig_dir: Path) -> None:
    """2×2 figure: TL VAE neg-ELBO boxplot, TR SAE loss boxplot,
    BL VAE KL boxplot, BR median MSE bar chart (SAE vs VAE per condition)."""

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("OOD Detection: VAE ELBO/KL vs. SAE Loss", fontsize=14, y=1.01)

    # Data organisation
    all_keys = COND_KEYS        # ["in_dist", "rand_noise", "dense", "scaled_5x"]
    xlabels = CONDITIONS         # display names

    def _collect(signal: str) -> list[np.ndarray]:
        return [scores[k][signal] for k in all_keys]

    bp_kw = dict(showfliers=False, patch_artist=True)

    # ---- TL: VAE neg-ELBO ----
    ax = axes[0, 0]
    bp = ax.boxplot(_collect("vae_loss"), tick_labels=xlabels, **bp_kw)
    ax.set_title("VAE neg-ELBO (lower = more in-dist)")
    ax.set_ylabel("neg-ELBO")
    ax.tick_params(axis="x", rotation=15)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")

    # ---- TR: SAE total loss ----
    ax = axes[0, 1]
    bp = ax.boxplot(_collect("sae_loss"), tick_labels=xlabels, **bp_kw)
    ax.set_title("SAE Total Loss (recon + λ·L1)")
    ax.set_ylabel("SAE loss")
    ax.tick_params(axis="x", rotation=15)
    for patch in bp["boxes"]:
        patch.set_facecolor("#DD8452")

    # ---- BL: VAE KL ----
    ax = axes[1, 0]
    bp = ax.boxplot(_collect("vae_kl"), tick_labels=xlabels, **bp_kw)
    ax.set_title("VAE KL Divergence (posterior vs. Laplace prior)")
    ax.set_ylabel("KL")
    ax.tick_params(axis="x", rotation=15)
    for patch in bp["boxes"]:
        patch.set_facecolor("#55A868")

    # ---- BR: Median MSE bar chart, SAE vs VAE ----
    ax = axes[1, 1]
    x = np.arange(len(all_keys))
    width = 0.35
    sae_med = [np.median(scores[k]["sae_mse"]) for k in all_keys]
    vae_med = [np.median(scores[k]["vae_mse"]) for k in all_keys]
    bars_sae = ax.bar(x - width / 2, sae_med, width, label="SAE MSE", color="#DD8452")
    bars_vae = ax.bar(x + width / 2, vae_med, width, label="VAE MSE", color="#4C72B0")
    ax.set_title("Median Reconstruction MSE: SAE vs. VAE")
    ax.set_ylabel("Median MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=15, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "ood.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "ood.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures to {fig_dir / 'ood.pdf'} and {fig_dir / 'ood.png'}")


# ---------------------------------------------------------------------------
# 6. Narrative summary
# ---------------------------------------------------------------------------

def print_narrative(table: dict) -> None:
    """Print a brief per-condition analysis of which signal best separates OOD."""
    print("\n=== Narrative: which signal best separates each OOD condition? ===\n")

    for ood_key in OOD_KEYS:
        best_signal = max(SIGNAL_NAMES, key=lambda s: table[s][ood_key])
        best_auroc = table[best_signal][ood_key]
        print(f"  {OOD_LABELS[ood_key]:<25} → best signal: "
              f"{SIGNAL_LABELS[best_signal]:<18} (AUROC={best_auroc:.4f})")

    print()
    print("Key insight:")
    print("  Both methods detect OOD via reconstruction error (MSE), but only the VAE")
    print("  has KL — a dedicated, probabilistically-meaningful 'is the posterior being")
    print("  used unusually?' signal.  For activations that look plausible to the decoder")
    print("  yet are off-manifold in latent space, the VAE KL flags them while the SAE")
    print("  recon loss may remain low and miss them entirely.")
    print()
    print("GPT-2 portability note:")
    print("  Feed activations from a different transformer layer (or shuffled / scaled")
    print("  activations) as OOD.  The VAE's KL provides a calibrated per-activation-site")
    print("  OOD score; the SAE has no equivalent signal — only reconstruction residual.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_samples = 5000

    # Train
    model, sae, vae = build_and_train(n_samples=n_samples)

    # Build conditions
    print("\n=== Building OOD conditions ===")
    conditions = build_conditions(model, n_samples=n_samples)
    for key, h in conditions.items():
        print(f"  {key}: shape={tuple(h.shape)}, mean={h.mean():.4f}, std={h.std():.4f}")

    # Score
    print("\n=== Scoring conditions with SAE and VAE ===")
    scores = score_conditions(conditions, sae, vae)

    # AUROC
    table = compute_auroc_table(scores)
    print_auroc_table(table)

    # Save CSV
    csv_path = REP_DIR / "ood_auroc.csv"
    save_auroc_csv(table, csv_path)
    print(f"Saved AUROC table to {csv_path}")

    # Figures
    print("\n=== Generating figures ===")
    make_figure(scores, FIG_DIR)

    # Narrative
    print_narrative(table)


if __name__ == "__main__":
    main()
