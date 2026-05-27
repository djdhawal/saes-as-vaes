"""Pareto-frontier sweep: MSE vs L0 across SAE, Gaussian VAE, Laplace VAE, Spike-slab VAE.

Reproduces notebook 1 (sae_vae_toy_models.ipynb) with the FIXED L0 metric and a rigorous
check for spike-slab gate collapse.

Run with:
    cd /Users/dhawaldixit/projects/saes-as-vaes/experiments
    python -m saevae.toy.experiments.pareto
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saevae.toy import (
    ToyModelConfig,
    ToyModel,
    ToySAE,
    GaussianVAE,
    LaplaceVAE,
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
# Sweep helpers
# ──────────────────────────────────────────────────────────────────────────────

def sweep_sae(model, n_inst, d_hidden):
    """Sweep SAE sparsity coefficients. Returns list of dicts with results."""
    lambdas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    records = []
    for lam in lambdas:
        sae = ToySAE(n_inst=n_inst, d_in=d_hidden, d_sae=5, sparsity_coeff=lam)
        train_sae(model, sae, steps=5000, progress=True)
        metrics = evaluate(model, sae, n_eval=10000)
        mse = float(metrics["mse"][0])
        l0  = float(metrics["l0"][0])
        el0 = float(metrics["expected_l0"][0])
        print(f"  SAE lam={lam:.3f}  MSE={mse:.5f}  L0={l0:.3f}")
        records.append({
            "family": "SAE",
            "param_name": "lambda",
            "param_value": lam,
            "mse": mse,
            "l0": l0,
            "expected_l0": el0,
        })
    return records


def sweep_gaussian_vae(model, n_inst, d_hidden):
    """Sweep Gaussian VAE beta. Returns list of dicts."""
    betas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    records = []
    for beta in betas:
        vae = GaussianVAE(n_inst=n_inst, d_in=d_hidden, d_latent=5, beta=beta)
        train_vae(model, vae, steps=5000, kl_warmup=1000, progress=True)
        metrics = evaluate(model, vae, n_eval=10000)
        mse = float(metrics["mse"][0])
        l0  = float(metrics["l0"][0])
        el0 = float(metrics["expected_l0"][0])
        print(f"  Gaussian VAE beta={beta:.3f}  MSE={mse:.5f}  L0={l0:.3f}")
        records.append({
            "family": "Gaussian VAE",
            "param_name": "beta",
            "param_value": beta,
            "mse": mse,
            "l0": l0,
            "expected_l0": el0,
        })
    return records


def sweep_laplace_vae(model, n_inst, d_hidden):
    """Sweep Laplace VAE (b, beta) configs. Returns list of dicts."""
    configs = [
        {"b": 5.0, "beta": 0.1},
        {"b": 2.0, "beta": 0.1},
        {"b": 1.0, "beta": 0.1},
        {"b": 0.5, "beta": 0.1},
        {"b": 0.2, "beta": 0.1},
        {"b": 0.1, "beta": 0.1},
        {"b": 0.5, "beta": 0.3},
        {"b": 0.5, "beta": 0.5},
    ]
    records = []
    for cfg in configs:
        b, beta = cfg["b"], cfg["beta"]
        vae = LaplaceVAE(n_inst=n_inst, d_in=d_hidden, d_latent=5,
                         beta=beta, laplace_b=b)
        train_vae(model, vae, steps=5000, kl_warmup=1500, progress=True)
        metrics = evaluate(model, vae, n_eval=10000)
        mse = float(metrics["mse"][0])
        l0  = float(metrics["l0"][0])
        el0 = float(metrics["expected_l0"][0])
        print(f"  Laplace VAE b={b:.2f} beta={beta:.2f}  MSE={mse:.5f}  L0={l0:.3f}")
        records.append({
            "family": "Laplace VAE",
            "param_name": "b_beta",
            "param_value": f"{b},{beta}",
            "mse": mse,
            "l0": l0,
            "expected_l0": el0,
        })
    return records


def _try_spike_slab_config(model, n_inst, d_hidden, prior_pi, beta,
                            free_bits, steps, kl_warmup, temperature=0.5):
    """Train one spike-slab config and return metrics dict."""
    vae = SpikeSlabVAE(
        n_inst=n_inst, d_in=d_hidden, d_latent=5,
        beta=beta, prior_pi=prior_pi,
        temperature=temperature, free_bits=free_bits,
    )
    train_vae(model, vae, steps=steps, kl_warmup=kl_warmup, progress=True)
    return evaluate(model, vae, n_eval=10000)


def sweep_spike_slab(model, n_inst, d_hidden):
    """Sweep Spike-slab VAE configs.

    Collapse protocol:
      1. Default notebook configs (free_bits=0.5, 8000 steps, kl_warmup=3000).
      2. If ALL configs produce hard L0 == 0, try: longer warmup (4000) + more
         steps (10000), then lower beta, then raise free_bits to 0.8.
    Returns list of dicts.
    """
    base_configs = [
        {"pi": 0.5,  "beta": 0.3},
        {"pi": 0.3,  "beta": 0.3},
        {"pi": 0.1,  "beta": 0.3},
        {"pi": 0.05, "beta": 0.3},
        {"pi": 0.1,  "beta": 0.05},
        {"pi": 0.1,  "beta": 0.1},
        {"pi": 0.1,  "beta": 0.5},
    ]

    # ── Pass 1: notebook defaults ──────────────────────────────────────────────
    print("\n[Spike-slab pass 1] Default: free_bits=0.5, steps=8000, kl_warmup=3000")
    raw_records = []
    for cfg in base_configs:
        pi, beta = cfg["pi"], cfg["beta"]
        m = _try_spike_slab_config(
            model, n_inst, d_hidden,
            prior_pi=pi, beta=beta,
            free_bits=0.5, steps=8000, kl_warmup=3000,
        )
        mse = float(m["mse"][0])
        l0  = float(m["l0"][0])
        el0 = float(m["expected_l0"][0])
        print(f"  SpikeSlab pi={pi:.3f} beta={beta:.2f} → MSE={mse:.5f}  hard_L0={l0:.3f}  E[L0]={el0:.3f}")
        raw_records.append((pi, beta, mse, l0, el0, "free_bits=0.5,steps=8000,kl_warmup=3000"))

    all_collapsed = all(r[3] == 0.0 for r in raw_records)

    if all_collapsed:
        print("\n[Spike-slab] ALL configs collapsed (hard L0=0). Trying fix (a): longer warmup/steps.")
        raw_records = []
        for cfg in base_configs:
            pi, beta = cfg["pi"], cfg["beta"]
            m = _try_spike_slab_config(
                model, n_inst, d_hidden,
                prior_pi=pi, beta=beta,
                free_bits=0.5, steps=10000, kl_warmup=4000,
            )
            mse = float(m["mse"][0])
            l0  = float(m["l0"][0])
            el0 = float(m["expected_l0"][0])
            print(f"  SpikeSlab pi={pi:.3f} beta={beta:.2f} → MSE={mse:.5f}  hard_L0={l0:.3f}  E[L0]={el0:.3f}")
            raw_records.append((pi, beta, mse, l0, el0, "free_bits=0.5,steps=10000,kl_warmup=4000"))
        all_collapsed = all(r[3] == 0.0 for r in raw_records)

    if all_collapsed:
        print("\n[Spike-slab] Still collapsed. Trying fix (b): lower beta (beta=0.01).")
        extra_configs = [{"pi": p, "beta": 0.01} for p in [0.5, 0.3, 0.1, 0.05]]
        for cfg in extra_configs:
            pi, beta = cfg["pi"], cfg["beta"]
            m = _try_spike_slab_config(
                model, n_inst, d_hidden,
                prior_pi=pi, beta=beta,
                free_bits=0.5, steps=10000, kl_warmup=4000,
            )
            mse = float(m["mse"][0])
            l0  = float(m["l0"][0])
            el0 = float(m["expected_l0"][0])
            print(f"  SpikeSlab pi={pi:.3f} beta={beta:.3f} → MSE={mse:.5f}  hard_L0={l0:.3f}  E[L0]={el0:.3f}")
            raw_records.append((pi, beta, mse, l0, el0, "free_bits=0.5,steps=10000,kl_warmup=4000,beta_low"))
        # Keep only the new entries for collapse check
        all_collapsed_check = all(r[3] == 0.0 for r in raw_records if "beta_low" in r[5])
        if not all_collapsed_check:
            all_collapsed = False

    if all_collapsed:
        print("\n[Spike-slab] Still collapsed. Trying fix (c): raise free_bits=0.8.")
        raw_records = []
        for cfg in base_configs:
            pi, beta = cfg["pi"], cfg["beta"]
            m = _try_spike_slab_config(
                model, n_inst, d_hidden,
                prior_pi=pi, beta=beta,
                free_bits=0.8, steps=10000, kl_warmup=4000,
            )
            mse = float(m["mse"][0])
            l0  = float(m["l0"][0])
            el0 = float(m["expected_l0"][0])
            print(f"  SpikeSlab pi={pi:.3f} beta={beta:.2f} → MSE={mse:.5f}  hard_L0={l0:.3f}  E[L0]={el0:.3f}")
            raw_records.append((pi, beta, mse, l0, el0, "free_bits=0.8,steps=10000,kl_warmup=4000"))
        all_collapsed = all(r[3] == 0.0 for r in raw_records)

    if all_collapsed:
        print("\n[Spike-slab] WARNING: spike-slab remains collapsed after all fixes. "
              "Reporting expected_l0 only (hard L0 = 0 everywhere). "
              "This may indicate the toy model's 2-D hidden space doesn't offer enough "
              "signal for the gates to stay open under KL pressure.")
    else:
        print("\n[Spike-slab] At least some configs escaped collapse.")

    # Build final records from raw_records (deduplicate by (pi, beta) — keep last)
    seen = {}
    for pi, beta, mse, l0, el0, note in raw_records:
        seen[(pi, beta)] = (mse, l0, el0, note)

    records = []
    for (pi, beta), (mse, l0, el0, note) in seen.items():
        records.append({
            "family": "Spike-slab VAE",
            "param_name": "pi_beta",
            "param_value": f"{pi},{beta}",
            "mse": mse,
            "l0": l0,
            "expected_l0": el0,
        })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(all_records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "param_name", "param_value",
                                                "mse", "l0", "expected_l0"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"CSV saved: {path}")


def save_plot(all_records, pdf_path, png_path):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    style_map = {
        "SAE":           {"color": "red",    "marker": "o", "label": "SAE"},
        "Gaussian VAE":  {"color": "blue",   "marker": "^", "label": "Gaussian VAE"},
        "Laplace VAE":   {"color": "green",  "marker": "s", "label": "Laplace VAE"},
        "Spike-slab VAE":{"color": "purple", "marker": "D", "label": "Spike-slab VAE"},
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect per-family (l0, mse) pairs for Pareto highlighting
    family_groups: dict[str, list] = {}
    for rec in all_records:
        fam = rec["family"]
        family_groups.setdefault(fam, []).append(rec)

    for fam, recs in family_groups.items():
        st = style_map[fam]
        for rec in recs:
            l0  = rec["l0"]
            mse = rec["mse"]
            ax.scatter(l0, mse,
                       color=st["color"], marker=st["marker"], s=80,
                       zorder=3, label=st["label"] if rec == recs[0] else None)
            # Annotation: the primary sweep parameter value
            pval = rec["param_value"]
            ax.annotate(str(pval), (l0, mse),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=st["color"])

    ax.set_xlabel("L0 (latents active per token)", fontsize=12)
    ax.set_ylabel("MSE (reconstruction)", fontsize=12)
    ax.set_title("Pareto Frontier: MAP vs VI under different priors (Toy Superposition)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figures saved: {pdf_path}  {png_path}")


def print_summary(all_records):
    header = f"{'Family':<18} {'Param':<12} {'Value':<14} {'MSE':>10} {'L0':>8} {'E[L0]':>8}"
    print("\n" + "=" * len(header))
    print("  PARETO SWEEP SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for rec in all_records:
        print(
            f"{rec['family']:<18} {rec['param_name']:<12} {str(rec['param_value']):<14}"
            f" {rec['mse']:>10.5f} {rec['l0']:>8.3f} {rec['expected_l0']:>8.3f}"
        )
    print("=" * len(header))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_sweep():
    """Train the base toy model and run all sweeps."""
    n_inst    = 8
    n_features = 5
    d_hidden  = 2

    print("=" * 60)
    print("Step 1/5: Training base toy model ...")
    cfg   = ToyModelConfig(n_inst=n_inst, n_features=n_features, d_hidden=d_hidden)
    model = train_toy_model(cfg, feature_probability=0.025, importance=1.0,
                            steps=5000, batch_size=1024, lr=1e-3,
                            device=None, tie_instances=True, progress=True)
    print("Base model trained.")

    print("\n" + "=" * 60)
    print("Step 2/5: SAE sweep ...")
    sae_records = sweep_sae(model, n_inst, d_hidden)

    print("\n" + "=" * 60)
    print("Step 3/5: Gaussian VAE sweep ...")
    gauss_records = sweep_gaussian_vae(model, n_inst, d_hidden)

    print("\n" + "=" * 60)
    print("Step 4/5: Laplace VAE sweep ...")
    laplace_records = sweep_laplace_vae(model, n_inst, d_hidden)

    print("\n" + "=" * 60)
    print("Step 5/5: Spike-slab VAE sweep ...")
    spiked_records = sweep_spike_slab(model, n_inst, d_hidden)

    all_records = sae_records + gauss_records + laplace_records + spiked_records
    return all_records


def main():
    all_records = run_sweep()

    # Outputs
    csv_path = "reports/toy/pareto.csv"
    pdf_path = "figures/toy/pareto.pdf"
    png_path = "figures/toy/pareto.png"

    save_csv(all_records, csv_path)
    save_plot(all_records, pdf_path, png_path)
    print_summary(all_records)

    # Final collapse status
    ss_records = [r for r in all_records if r["family"] == "Spike-slab VAE"]
    if all(r["l0"] == 0.0 for r in ss_records):
        print("\n[Spike-slab status] COLLAPSED: hard L0 = 0 for every spike-slab config.")
        print(f"  max E[L0] = {max(r['expected_l0'] for r in ss_records):.3f}")
    else:
        healthy = [r for r in ss_records if r["l0"] > 0]
        print(f"\n[Spike-slab status] HEALTHY: {len(healthy)}/{len(ss_records)} configs have L0 > 0.")

    return all_records


if __name__ == "__main__":
    main()
