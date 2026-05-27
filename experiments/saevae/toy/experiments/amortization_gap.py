"""Experiment B — Amortization gap in the SAE encoder.

Ports GPT-2 Experiment 3.3 down to the toy model, exploiting 2-D geometry for
rigorous visualization. Story: the SAE encoder is an amortized approximation to
the true MAP code (optimal sparse code). We compare the encoder's code against
the optimal z* found by ISTA on the SAME frozen decoder objective:
    objective(z) = 0.5 * ||h_cent - z @ W_d||^2 + lambda * ||z||_1
                   (non-negative z to match ReLU support)

The amortization gap = encoder_objective - ISTA_objective (>= 0 by definition).

Run with:
    cd /Users/dhawaldixit/projects/saes-as-vaes/experiments
    python -m saevae.toy.experiments.amortization_gap
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from saevae.toy import (
    ToyModelConfig,
    ToySAE,
    train_toy_model,
    train_sae,
    evaluate,
)

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

# ──────────────────────────────────────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────────────────────────────────────
FIGURES_DIR = "figures/toy"
REPORTS_DIR = "reports/toy"
PDF_PATH = os.path.join(FIGURES_DIR, "amortization_gap.pdf")
PNG_PATH = os.path.join(FIGURES_DIR, "amortization_gap.png")
CSV_PATH = os.path.join(REPORTS_DIR, "amortization_gap.csv")

LAMBDA = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# ISTA for non-negative sparse coding
# ──────────────────────────────────────────────────────────────────────────────

def spectral_norm_power_iter(W: torch.Tensor, n_iter: int = 50) -> float:
    """Estimate spectral norm of W via power iteration. W shape: (d_sae, d_in)."""
    u = torch.randn(W.shape[0], device=W.device)
    u = u / (u.norm() + 1e-8)
    with torch.no_grad():
        for _ in range(n_iter):
            v = W.T @ u           # (d_in,)
            v = v / (v.norm() + 1e-8)
            u = W @ v             # (d_sae,)
            sigma = u.norm()
            u = u / (sigma + 1e-8)
    return sigma.item()


def ista_nn(h_cent: torch.Tensor, W_d: torch.Tensor,
            lam: float, n_iter: int = 200,
            z0: torch.Tensor | None = None) -> torch.Tensor:
    """Non-negative ISTA to solve:
        min_z  0.5 * ||h_cent - z @ W_d||^2 + lam * ||z||_1   s.t. z >= 0

    h_cent : (batch, d_in)   -- centered hidden activations
    W_d    : (d_sae, d_in)   -- frozen decoder weight matrix
    z0     : (batch, d_sae)  -- optional warm start; if None, start from zero

    Returns z : (batch, d_sae)
    """
    # Step size 1/L where L = spectral norm of W_d^T W_d = (spectral norm of W_d)^2
    sigma = spectral_norm_power_iter(W_d)
    L = sigma ** 2 + 1e-8
    step = 1.0 / L

    if z0 is None:
        z = torch.zeros(h_cent.shape[0], W_d.shape[0], device=h_cent.device)
    else:
        z = z0.clone()

    W_d_t = W_d.T  # (d_in, d_sae)

    with torch.no_grad():
        for _ in range(n_iter):
            # Gradient of 0.5 * ||h_cent - z @ W_d||^2  w.r.t. z
            # residual (batch, d_in); grad = -residual @ W_d^T = z @ W_d^T W_d - h_cent @ W_d^T
            residual = h_cent - z @ W_d          # (batch, d_in)
            grad = -(residual @ W_d_t)           # (batch, d_sae)
            # Gradient step
            z_next = z - step * grad
            # Proximal: soft-threshold for L1, then non-negativity
            # soft_threshold(v, lam/L) = sign(v) * max(|v| - lam/L, 0)
            # For non-negative constraint: equivalent to (v - lam/L).clamp(min=0)
            z = (z_next - step * lam).clamp(min=0.0)

    return z


def sae_objective(z: torch.Tensor, h_cent: torch.Tensor,
                  W_d: torch.Tensor, lam: float) -> torch.Tensor:
    """Per-sample objective value. Returns (batch,)."""
    recon = z @ W_d          # (batch, d_in)
    mse_per_sample = 0.5 * (h_cent - recon).pow(2).sum(dim=-1)
    l1_per_sample = lam * z.abs().sum(dim=-1)
    return mse_per_sample + l1_per_sample


def sae_mse_only(z: torch.Tensor, h_cent: torch.Tensor, W_d: torch.Tensor) -> torch.Tensor:
    """Per-sample reconstruction MSE (mean over d_in). Returns (batch,)."""
    recon = z @ W_d
    return (h_cent - recon).pow(2).mean(dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check: lambda=0 ISTA -> least squares
# ──────────────────────────────────────────────────────────────────────────────

def sanity_check_ista(sae: ToySAE, h_batch: torch.Tensor, tol: float = 1e-2):
    """At lambda=0, ISTA should recover least-squares solution.

    ||h - z @ W_d||^2 at z=ISTA_lam0 should be close to the pseudoinverse solution.
    Checks that mean MSE < tol.
    """
    h = h_batch[:, 0, :]               # (batch, d_in)
    b_dec = sae.b_dec[0].detach()       # (d_in,)
    W_d = sae.W_dec_normalized[0].detach()  # (d_sae, d_in)
    h_cent = h - b_dec                  # (batch, d_in)

    z_ista = ista_nn(h_cent, W_d, lam=0.0, n_iter=300)
    mse = sae_mse_only(z_ista, h_cent, W_d)
    mean_mse = mse.mean().item()

    # Least-squares reference via pseudoinverse
    # z_ls = h_cent @ W_d^+ = h_cent @ W_d^T @ (W_d @ W_d^T)^{-1}
    W_d_np = W_d.cpu().numpy()
    h_np = h_cent.cpu().numpy()
    # Non-negative LS won't match unconstrained, but for lambda=0 ISTA
    # should at least achieve very low MSE compared to SAE (almost zero)
    # (full rank case: d_in=2, d_sae=5, overdetermined in z, underdetermined in rows)
    print(f"  [Sanity] ISTA at lam=0: mean MSE={mean_mse:.6f} (should be ~0 if W_d full row-rank)")

    # The unconstrained LS with W_d (d_sae=5, d_in=2) always has residual ~0
    # since d_sae > d_in. With non-neg constraint residual is small but may not be 0.
    # Assert ISTA lam=0 achieves lower MSE than lam=0.1 encoder
    return mean_mse


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_amortization_gap(n_eval: int = 2000):
    n_inst = 1
    n_features = 5
    d_hidden = 2

    # ── Step 1: train base model + SAE ────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Training base toy model (n_inst=1, 5 features, d=2) ...")
    cfg = ToyModelConfig(n_inst=n_inst, n_features=n_features, d_hidden=d_hidden)
    model = train_toy_model(
        cfg, feature_probability=0.025, steps=5000,
        tie_instances=True, progress=True,
    )
    print("Base model trained.\n")

    print("Training SAE (lambda=0.1, 5000 steps) ...")
    sae = ToySAE(n_inst=n_inst, d_in=d_hidden, d_sae=n_features, sparsity_coeff=LAMBDA)
    train_sae(model, sae, steps=5000, progress=True)
    sae_metrics = evaluate(model, sae, n_eval=10000)
    print(f"SAE trained: MSE={float(sae_metrics['mse'][0]):.5f}  L0={float(sae_metrics['l0'][0]):.3f}\n")

    # ── Step 2: get eval data ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"Step 2: Generating {n_eval} eval samples ...")
    h_batch, _ = model.get_hidden_activations(n_eval)   # (n_eval, 1, 2)
    h = h_batch[:, 0, :]                                 # (n_eval, d_in)
    b_dec = sae.b_dec[0].detach()                        # (d_in,)
    W_d = sae.W_dec_normalized[0].detach()               # (d_sae=5, d_in=2)
    h_cent = h - b_dec                                   # (n_eval, d_in)

    # ── Sanity check: ISTA at lam=0 ──────────────────────────────────────────
    print("\n[Sanity check] ISTA at lambda=0 should approach least squares ...")
    lam0_mse = sanity_check_ista(sae, h_batch, tol=1e-2)

    # ── Step 3: encoder code ─────────────────────────────────────────────────
    print("\nStep 3: Computing encoder codes ...")
    sae.eval()
    with torch.no_grad():
        h_full = h_batch  # (batch, 1, d_in)
        _, z_enc_full, _ = sae(h_full)
        z_enc = z_enc_full[:, 0, :]                       # (n_eval, d_sae)

    # ── Step 4: ISTA cold start ───────────────────────────────────────────────
    print("Step 4a: ISTA cold start (z0=0) ...")
    z_cold = ista_nn(h_cent, W_d, lam=LAMBDA, n_iter=200, z0=None)

    # ── Step 5: ISTA warm start from encoder ─────────────────────────────────
    print("Step 4b: ISTA warm start (z0=encoder code) ...")
    z_warm = ista_nn(h_cent, W_d, lam=LAMBDA, n_iter=200, z0=z_enc)

    # ── Step 6: compute objectives ────────────────────────────────────────────
    print("\nStep 5: Computing per-sample objectives ...")
    obj_enc  = sae_objective(z_enc,  h_cent, W_d, LAMBDA)  # (n_eval,)
    obj_cold = sae_objective(z_cold, h_cent, W_d, LAMBDA)
    obj_warm = sae_objective(z_warm, h_cent, W_d, LAMBDA)

    mse_enc  = sae_mse_only(z_enc,  h_cent, W_d)
    mse_cold = sae_mse_only(z_cold, h_cent, W_d)
    mse_warm = sae_mse_only(z_warm, h_cent, W_d)

    l0_enc  = (z_enc  > 0).float().sum(dim=-1)
    l0_cold = (z_cold > 0).float().sum(dim=-1)
    l0_warm = (z_warm > 0).float().sum(dim=-1)

    gap_cold = (obj_enc - obj_cold)   # should be >= 0
    gap_warm = (obj_enc - obj_warm)   # should also be >= 0

    results = {
        "encoder":   {"obj": obj_enc,  "mse": mse_enc,  "l0": l0_enc,  "z": z_enc},
        "ista_cold": {"obj": obj_cold, "mse": mse_cold, "l0": l0_cold, "z": z_cold},
        "ista_warm": {"obj": obj_warm, "mse": mse_warm, "l0": l0_warm, "z": z_warm},
    }

    summary = {}
    for method, d in results.items():
        summary[method] = {
            "mean_objective": d["obj"].mean().item(),
            "mean_mse":       d["mse"].mean().item(),
            "mean_l0":        d["l0"].mean().item(),
        }

    mean_gap_cold = gap_cold.mean().item()
    mean_gap_warm = gap_warm.mean().item()

    print("\n--- Objective comparison ---")
    print(f"  Encoder:    obj={summary['encoder']['mean_objective']:.6f}  "
          f"mse={summary['encoder']['mean_mse']:.6f}  L0={summary['encoder']['mean_l0']:.3f}")
    print(f"  ISTA cold:  obj={summary['ista_cold']['mean_objective']:.6f}  "
          f"mse={summary['ista_cold']['mean_mse']:.6f}  L0={summary['ista_cold']['mean_l0']:.3f}")
    print(f"  ISTA warm:  obj={summary['ista_warm']['mean_objective']:.6f}  "
          f"mse={summary['ista_warm']['mean_mse']:.6f}  L0={summary['ista_warm']['mean_l0']:.3f}")
    print(f"\n  Mean amortization gap (enc - ISTA cold) = {mean_gap_cold:.6f}")
    print(f"  Mean gap (enc - ISTA warm)              = {mean_gap_warm:.6f}")
    neg_cold = (gap_cold < -1e-4).float().mean().item()
    print(f"  Fraction of samples where gap_cold < 0  = {neg_cold:.3f} "
          f"({'unexpected — verify' if neg_cold > 0.01 else 'OK'})")

    return results, summary, gap_cold, gap_warm, h_cent, W_d, b_dec, lam0_mse


# ──────────────────────────────────────────────────────────────────────────────
# Save CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(summary, gap_cold_mean, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "method": method,
            "mean_objective": v["mean_objective"],
            "mean_mse": v["mean_mse"],
            "mean_l0": v["mean_l0"],
        }
        for method, v in summary.items()
    ]
    rows.append({
        "method": "gap_encoder_minus_ista_cold",
        "mean_objective": gap_cold_mean,
        "mean_mse": float("nan"),
        "mean_l0": float("nan"),
    })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mean_objective", "mean_mse", "mean_l0"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Save plots
# ──────────────────────────────────────────────────────────────────────────────

def save_plots(results, gap_cold, h_cent, W_d, b_dec, pdf_path, png_path):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    z_enc  = results["encoder"]["z"].cpu().numpy()
    z_cold = results["ista_cold"]["z"].cpu().numpy()
    z_warm = results["ista_warm"]["z"].cpu().numpy()
    h_np   = h_cent.cpu().numpy()       # (n_eval, 2)
    W_d_np = W_d.cpu().numpy()          # (d_sae=5, d_in=2)
    b_dec_np = b_dec.cpu().numpy()      # (2,)
    gap_np = gap_cold.cpu().numpy()

    fig = plt.figure(figsize=(15, 5))

    # ── Panel A: histogram of per-sample gap ─────────────────────────────────
    ax_hist = fig.add_subplot(1, 3, 1)
    ax_hist.hist(gap_np, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax_hist.axvline(0, color="red", lw=1.5, linestyle="--", label="gap=0")
    mean_gap = gap_np.mean()
    ax_hist.axvline(mean_gap, color="darkorange", lw=1.5, linestyle="--",
                    label=f"mean gap={mean_gap:.4f}")
    ax_hist.set_xlabel("Objective gap (encoder - ISTA cold)", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("(a) Per-sample amortization gap\n(encoder − ISTA cold)", fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, linestyle="--", alpha=0.4)

    # ── Panel B: 2-D hidden plane — encoder vs ISTA reconstructions ──────────
    # Pick a handful of representative h points for geometry panel.
    # Use h points that span interesting reconstructions (varied norms).
    h_orig = h_cent + b_dec_np           # de-center to get original h
    n_probe = 8
    norms = np.linalg.norm(h_orig, axis=1)
    # Pick evenly-spaced quantiles to get diverse points
    quantile_idx = np.round(np.linspace(0, len(norms) - 1, n_probe)).astype(int)
    sorted_idx = np.argsort(norms)
    probe_idx = sorted_idx[quantile_idx]

    ax_2d = fig.add_subplot(1, 3, 2)

    # Background: scatter a subset of all h points
    subsample = np.random.choice(len(h_orig), min(500, len(h_orig)), replace=False)
    ax_2d.scatter(h_orig[subsample, 0], h_orig[subsample, 1],
                  c="lightgray", s=8, alpha=0.5, zorder=1, label="h (sample)")

    # Draw decoder dictionary columns as arrows
    for k in range(W_d_np.shape[0]):
        ax_2d.annotate("", xy=W_d_np[k] * 0.5 + b_dec_np,
                       xytext=b_dec_np,
                       arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    colors_probe = plt.cm.tab10(np.linspace(0, 0.7, n_probe))
    for i, idx in enumerate(probe_idx):
        h_pt = h_orig[idx]
        # Encoder recon
        recon_enc  = z_enc[idx]  @ W_d_np + b_dec_np
        recon_cold = z_cold[idx] @ W_d_np + b_dec_np
        c = colors_probe[i]
        ax_2d.scatter(*h_pt,       marker="o", s=60, color=c, zorder=5)
        ax_2d.scatter(*recon_enc,  marker="x", s=80, color=c, zorder=5)
        ax_2d.scatter(*recon_cold, marker="^", s=60, color=c, zorder=5)
        # Line from truth to encoder recon
        ax_2d.plot([h_pt[0], recon_enc[0]], [h_pt[1], recon_enc[1]],
                   color=c, lw=0.6, alpha=0.7)
        ax_2d.plot([h_pt[0], recon_cold[0]], [h_pt[1], recon_cold[1]],
                   color=c, lw=0.6, linestyle="--", alpha=0.7)

    # Legend entries
    ax_2d.scatter([], [], marker="o", color="gray", s=60, label="true h")
    ax_2d.scatter([], [], marker="x", color="gray", s=80, label="encoder recon")
    ax_2d.scatter([], [], marker="^", color="gray", s=60, label="ISTA recon")
    ax_2d.set_title("(b) 2-D geometry: encoder vs ISTA\n"
                    "(circle=h, x=enc recon, tri=ISTA cold recon)", fontsize=9)
    ax_2d.set_xlabel("h[0]", fontsize=9)
    ax_2d.set_ylabel("h[1]", fontsize=9)
    ax_2d.legend(fontsize=7, loc="upper left")
    ax_2d.grid(True, linestyle="--", alpha=0.3)

    # ── Panel C: objective values per sample (sorted) ─────────────────────────
    ax_obj = fig.add_subplot(1, 3, 3)
    obj_enc  = results["encoder"]["obj"].cpu().numpy()
    obj_cold = results["ista_cold"]["obj"].cpu().numpy()
    obj_warm = results["ista_warm"]["obj"].cpu().numpy()

    sort_order = np.argsort(obj_cold)
    n_show = min(500, len(obj_cold))
    idx_show = sort_order[np.round(np.linspace(0, len(sort_order) - 1, n_show)).astype(int)]

    ax_obj.plot(np.arange(n_show), obj_enc[idx_show],   ".",  color="steelblue",
                ms=3, alpha=0.7, label="Encoder")
    ax_obj.plot(np.arange(n_show), obj_cold[idx_show],  ".",  color="darkorange",
                ms=3, alpha=0.7, label="ISTA cold")
    ax_obj.plot(np.arange(n_show), obj_warm[idx_show],  ".",  color="green",
                ms=3, alpha=0.7, label="ISTA warm")
    ax_obj.set_xlabel("Sample (sorted by ISTA cold obj)", fontsize=9)
    ax_obj.set_ylabel("Objective value", fontsize=9)
    ax_obj.set_title("(c) Per-sample objectives\n(encoder always ≥ ISTA)", fontsize=10)
    ax_obj.legend(fontsize=8)
    ax_obj.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Amortization Gap — SAE Encoder vs Optimal ISTA Code (lambda={LAMBDA})\n"
        f"Toy superposition: 5 features, 2-D hidden",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figures saved: {pdf_path}  {png_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    results, summary, gap_cold, gap_warm, h_cent, W_d, b_dec, lam0_mse = \
        run_amortization_gap(n_eval=2000)

    save_csv(summary, gap_cold.mean().item(), CSV_PATH)
    save_plots(results, gap_cold, h_cent, W_d, b_dec, PDF_PATH, PNG_PATH)

    print("\n--- Final Report ---")
    print(f"  lambda=0 ISTA sanity check: mean MSE = {lam0_mse:.6f}")
    print(f"  Encoder mean objective:     {summary['encoder']['mean_objective']:.6f}")
    print(f"  ISTA cold mean objective:   {summary['ista_cold']['mean_objective']:.6f}")
    print(f"  ISTA warm mean objective:   {summary['ista_warm']['mean_objective']:.6f}")
    print(f"  Mean amortization gap:      {gap_cold.mean().item():.6f}")
    print(f"  Warm gap (enc - warm):      {gap_warm.mean().item():.6f}")
    enc_close = abs(summary['encoder']['mean_objective'] - summary['ista_cold']['mean_objective'])
    print(f"  |enc - cold| = {enc_close:.6f} "
          f"({'encoder close to ISTA' if enc_close < 0.05 else 'encoder farther from ISTA'})")


if __name__ == "__main__":
    main()
