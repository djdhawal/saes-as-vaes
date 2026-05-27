"""Experiment A — Prior sensitivity & posterior entropy collapse.

Ports GPT-2 Experiment 3.2 down to the toy model, where ground truth and 2-D
geometry make it rigorous. Story: as the spike-slab prior pi shrinks, the
approximate posterior gets more certain (entropy collapses) and the expected
active latent count L0 approaches the SAE's MAP-like reference sparsity.

Run with:
    cd /Users/dhawaldixit/projects/saes-as-vaes/experiments
    python -m saevae.toy.experiments.posterior_entropy
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saevae.toy import (
    ToyModelConfig,
    ToySAE,
    SpikeSlabVAE,
    train_toy_model,
    train_sae,
    train_vae,
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
PDF_PATH = os.path.join(FIGURES_DIR, "posterior_entropy_vs_prior.pdf")
PNG_PATH = os.path.join(FIGURES_DIR, "posterior_entropy_vs_prior.png")
CSV_PATH = os.path.join(REPORTS_DIR, "posterior_entropy.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Posterior entropy computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_posterior_entropy(vae: SpikeSlabVAE, h: torch.Tensor) -> float:
    """Mean posterior entropy over samples.

    Entropy per latent = H_bern(gamma) + gamma * 0.5*(log(2*pi*e) + logvar)
    where gamma = sigmoid(gate_logits).

    H_bern(gamma) = -[gamma*log(gamma) + (1-gamma)*log(1-gamma)]
    Sum over latents, mean over samples and return scalar float.
    """
    eps = 1e-8
    vae.eval()
    with torch.no_grad():
        gate_logits, mu, logvar = vae.encode(h)
        logvar = logvar.clamp(-10, 2)
        gamma = torch.sigmoid(gate_logits).clamp(eps, 1 - eps)

        # Bernoulli gate entropy (batch, inst, d_latent)
        h_bern = -(gamma * torch.log(gamma) + (1 - gamma) * torch.log(1 - gamma))

        # Gaussian slab entropy contribution gated by gamma
        # H_gauss = 0.5*(log(2*pi*e) + logvar) ; log(2*pi*e) = log(2*pi) + 1
        log_2pie = np.log(2.0 * np.pi) + 1.0
        h_slab = gamma * 0.5 * (log_2pie + logvar)

        # Total per latent, sum over latents, mean over batch (inst=0 only)
        h_total = (h_bern + h_slab).sum(dim=-1)  # (batch, inst)
        mean_entropy = h_total[:, 0].mean().item()
    return mean_entropy


# ──────────────────────────────────────────────────────────────────────────────
# Train helpers with collapse detection
# ──────────────────────────────────────────────────────────────────────────────

def train_spike_slab(model, prior_pi: float,
                     beta: float, free_bits: float,
                     steps: int, kl_warmup: int,
                     temperature: float = 0.5) -> SpikeSlabVAE:
    """Train a SpikeSlabVAE; return the trained model."""
    n_inst = model.cfg.n_inst
    d_hidden = model.cfg.d_hidden
    n_features = model.cfg.n_features
    vae = SpikeSlabVAE(
        n_inst=n_inst,
        d_in=d_hidden,
        d_latent=n_features,
        beta=beta,
        prior_pi=prior_pi,
        temperature=temperature,
        free_bits=free_bits,
    )
    train_vae(model, vae, steps=steps, kl_warmup=kl_warmup, progress=True)
    return vae


# ──────────────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_posterior_entropy_sweep():
    n_inst = 1
    n_features = 5
    d_hidden = 2

    # ── Step 1: train base toy model ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Training base toy model (n_inst=1, 5 features, d=2) ...")
    cfg = ToyModelConfig(n_inst=n_inst, n_features=n_features, d_hidden=d_hidden)
    model = train_toy_model(
        cfg, feature_probability=0.025, steps=5000,
        tie_instances=True, progress=True,
    )
    print("Base model trained.\n")

    # ── Step 2: train SAE (MAP reference) ─────────────────────────────────────
    print("=" * 60)
    print("Step 2: Training SAE (lambda=0.1, 5000 steps) for MAP reference ...")
    sae = ToySAE(n_inst=n_inst, d_in=d_hidden, d_sae=n_features, sparsity_coeff=0.1)
    train_sae(model, sae, steps=5000, progress=True)
    sae_metrics = evaluate(model, sae, n_eval=10000)
    sae_l0 = float(sae_metrics["l0"][0])
    sae_mse = float(sae_metrics["mse"][0])
    print(f"SAE (MAP reference): MSE={sae_mse:.5f}  L0={sae_l0:.3f}\n")

    # ── Step 3: spike-slab sweep over prior_pi ────────────────────────────────
    prior_pis = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    beta = 0.3
    temperature = 0.5
    free_bits_default = 0.5
    steps = 8000
    kl_warmup = 3000

    print("=" * 60)
    print(f"Step 3: Sweeping prior_pi in {prior_pis}")
    print(f"        beta={beta}, free_bits={free_bits_default}, steps={steps}, kl_warmup={kl_warmup}")

    records = []

    # Generate eval batch once (same h for all runs for fairness)
    h_eval, _ = model.get_hidden_activations(10000)

    for prior_pi in prior_pis:
        print(f"\n  --- pi={prior_pi:.3f} ---")
        vae = train_spike_slab(
            model, prior_pi=prior_pi,
            beta=beta, free_bits=free_bits_default,
            steps=steps, kl_warmup=kl_warmup,
            temperature=temperature,
        )
        metrics = evaluate(model, vae, n_eval=10000)
        mse = float(metrics["mse"][0])
        l0 = float(metrics["l0"][0])
        expected_l0 = float(metrics["expected_l0"][0])

        # Check for collapse and possibly fix
        if expected_l0 < 0.05:
            print(f"  [WARN] pi={prior_pi:.3f}: expected_l0={expected_l0:.4f} near zero — "
                  "likely collapsed. Retrying with free_bits=0.8, steps=10000, kl_warmup=4000.")
            vae = train_spike_slab(
                model, prior_pi=prior_pi,
                beta=beta, free_bits=0.8,
                steps=10000, kl_warmup=4000,
                temperature=temperature,
            )
            metrics = evaluate(model, vae, n_eval=10000)
            mse = float(metrics["mse"][0])
            l0 = float(metrics["l0"][0])
            expected_l0 = float(metrics["expected_l0"][0])
            if expected_l0 < 0.05:
                print(f"  [WARN] pi={prior_pi:.3f}: still collapsed after fix. Reporting as-is.")

        # Posterior entropy
        vae.eval()
        h_eval_fresh, _ = model.get_hidden_activations(10000)
        entropy = compute_posterior_entropy(vae, h_eval_fresh)

        print(f"  pi={prior_pi:.3f}  MSE={mse:.5f}  hard_L0={l0:.3f}  E[L0]={expected_l0:.3f}  "
              f"PostEntropy={entropy:.4f}")

        records.append({
            "pi": prior_pi,
            "mse": mse,
            "l0": l0,
            "expected_l0": expected_l0,
            "posterior_entropy": entropy,
        })

    return records, sae_l0, sae_mse


# ──────────────────────────────────────────────────────────────────────────────
# Save CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pi", "mse", "l0", "expected_l0", "posterior_entropy"]
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Save plot
# ──────────────────────────────────────────────────────────────────────────────

def save_plot(records, sae_l0, pdf_path, png_path):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    pis = [r["pi"] for r in records]
    entropies = [r["posterior_entropy"] for r in records]
    l0s = [r["l0"] for r in records]
    el0s = [r["expected_l0"] for r in records]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_entropy = "steelblue"
    color_l0 = "darkorange"

    ax1.set_xlabel("Prior pi (log scale)", fontsize=12)
    ax1.set_ylabel("Mean posterior entropy (nats)", fontsize=12, color=color_entropy)
    line_ent = ax1.plot(pis, entropies, "o-", color=color_entropy, lw=2,
                        label="Posterior entropy", zorder=3)
    ax1.tick_params(axis="y", labelcolor=color_entropy)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("L0 / E[L0]", fontsize=12, color=color_l0)
    line_l0 = ax2.plot(pis, l0s, "s-", color=color_l0, lw=2,
                       label="Hard L0", zorder=3)
    line_el0 = ax2.plot(pis, el0s, "s--", color=color_l0, lw=1.5, alpha=0.7,
                        label="E[L0]", zorder=3)
    sae_line = ax2.axhline(y=sae_l0, color="crimson", linestyle="--", lw=1.5,
                           label=f"SAE L0 (MAP ref = {sae_l0:.2f})", zorder=2)
    ax2.tick_params(axis="y", labelcolor=color_l0)

    # Combined legend
    lines = line_ent + line_l0 + line_el0 + [sae_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_title(
        "Prior Sensitivity: Posterior Entropy Collapse & L0 → SAE Reference\n"
        "(Spike-slab VAE, toy superposition, 5 features, 2-D hidden)",
        fontsize=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.invert_xaxis()  # left = large pi (diffuse), right = small pi (sparse)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figures saved: {pdf_path}  {png_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(records, sae_l0):
    header = (
        f"{'pi':>6}  {'MSE':>10}  {'L0':>6}  {'E[L0]':>8}  {'PostEntropy':>12}"
    )
    print("\n" + "=" * len(header))
    print("  POSTERIOR ENTROPY SWEEP SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['pi']:>6.3f}  {r['mse']:>10.5f}  {r['l0']:>6.3f}  "
            f"{r['expected_l0']:>8.4f}  {r['posterior_entropy']:>12.4f}"
        )
    print("=" * len(header))
    print(f"\nSAE (MAP reference) L0 = {sae_l0:.3f}")

    # Interpret the story
    entropies = [r["posterior_entropy"] for r in records]
    el0s = [r["expected_l0"] for r in records]
    pis = [r["pi"] for r in records]
    # Records are in decreasing pi order (0.5 -> 0.02)
    entropy_dropped = entropies[-1] < entropies[0]
    l0_toward_sae = abs(el0s[-1] - sae_l0) < abs(el0s[0] - sae_l0)
    print(
        f"\nEntropy collapse (pi=0.5 → pi=0.02): "
        f"{entropies[0]:.4f} → {entropies[-1]:.4f}  "
        f"({'YES' if entropy_dropped else 'NO'})"
    )
    print(
        f"E[L0] towards SAE as pi shrinks: "
        f"{el0s[0]:.4f} → {el0s[-1]:.4f}  vs SAE={sae_l0:.3f}  "
        f"({'YES' if l0_toward_sae else 'NO'})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    records, sae_l0, sae_mse = run_posterior_entropy_sweep()
    save_csv(records, CSV_PATH)
    save_plot(records, sae_l0, PDF_PATH, PNG_PATH)
    print_summary(records, sae_l0)


if __name__ == "__main__":
    main()
