"""End-to-end smoke test. Must pass before launching real sweeps.

Trains a tiny SAE (d_dict=1024) for 200 steps on either:
  - synthetic N(0, 1) activations (default; portable, runs anywhere), or
  - the first 10k rows of the cached activation memmap (--use-cache).

Asserts:
  * all final-eval metrics are finite,
  * FVE > 0 (the model has learned something nontrivial),
  * L0 ∈ (0, d_dict) (not all-dead, not all-on),
  * decoder W_d has no NaNs/Infs,
  * decoder columns are unit-norm to within 1e-3 (the column-norm constraint
    didn't silently fail).

Exit code is non-zero on any assertion failure so this can be wired into
a Colab pre-flight cell. Should complete in <2 min on a T4 or CPU.

Usage:
    python smoke_test.py                  # synthetic
    python smoke_test.py --use-cache      # uses activations_data/ if present
    python smoke_test.py --steps 500      # longer run for sanity
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from saevae.config import TrainConfig
from saevae.sae import SAE
from saevae.train import train

ACT_DIR = "activations_data"


# ----------------------------------------------------------------------
class TensorOnlyDataset(Dataset):
    """Wraps a (N, d) tensor; __getitem__ returns the row (no label)."""

    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i]


def _synthetic_data(n: int = 10_000, d: int = 768, seed: int = 0) -> torch.Tensor:
    """N(0, 1) activations. Mean L2 norm ≈ sqrt(d), so the unit-per-coord
    normalization convention is already satisfied — no extra scale needed."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _cached_data(activations_dir: str, n: int) -> torch.Tensor:
    """Load and normalize the first `n` rows of the memmap."""
    from saevae.data import _load_memmap
    from saevae.normalize import ActivationNormalizer

    arr, _ = _load_memmap(activations_dir)
    rows = np.asarray(arr[:n], dtype=np.float32)
    normalizer = ActivationNormalizer.fit(rows, d_model=rows.shape[1])
    return torch.from_numpy(rows * normalizer.norm_scale)


# ----------------------------------------------------------------------
def run_smoke(
    *,
    use_cache: bool = False,
    steps: int = 200,
    n: int = 10_000,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    print(f"[smoke] mode={'cache' if use_cache else 'synthetic'} steps={steps}")

    # 1. Build data.
    if use_cache and os.path.exists(os.path.join(ACT_DIR, "activations.dat")):
        x = _cached_data(ACT_DIR, n)
    else:
        if use_cache:
            print(f"[smoke] {ACT_DIR}/activations.dat not found; falling back to synthetic")
        x = _synthetic_data(n=n, seed=seed)

    d_model = x.shape[1]
    n_eval = max(1000, n // 10)
    train_x, eval_x = x[:-n_eval], x[-n_eval:]
    train_loader = DataLoader(
        TensorOnlyDataset(train_x),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    eval_loader = DataLoader(
        TensorOnlyDataset(eval_x),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # 2. Build a tiny SAE.
    cfg = TrainConfig(
        family="sae",
        d_model=d_model,
        d_dict=1024,
        batch_size=batch_size,
        steps=steps,
        lr=1e-3,
        warmup_steps=20,
        eval_every=max(50, steps // 4),
        lam=0.01,
        resample_every=0,            # disable resampling at this scale
        amp_dtype="float32",         # smoke runs on CPU-friendly precision
        seed=seed,
    )
    model = SAE(d_model=cfg.d_model, d_dict=cfg.d_dict, lam=cfg.lam)

    # 3. Train. Force CPU — MPS has known float32 corner-cases on Apple
    # Silicon that have caused spurious NaNs in this exact setup, and the
    # smoke test only needs ~2 min on CPU anyway. Real training on Colab
    # uses CUDA via the trainer's auto-detect.
    final = train(model, cfg, train_loader, eval_loader, use_wandb=False, device="cpu")
    print(f"[smoke] final metrics: {final}")

    # 4. Assertions.
    assert math.isfinite(final["recon_mse"]), f"NaN/Inf in MSE: {final['recon_mse']}"
    assert math.isfinite(final["fve"]), f"NaN/Inf in FVE: {final['fve']}"
    assert math.isfinite(final["l0"]), f"NaN/Inf in L0: {final['l0']}"
    assert final["fve"] > 0.0, (
        f"FVE not positive ({final['fve']:.4f}) — model failed to learn anything"
    )
    assert 0 < final["l0"] < cfg.d_dict, (
        f"L0 = {final['l0']:.2f} outside (0, {cfg.d_dict}); all-dead or all-on"
    )
    assert torch.isfinite(model.W_d).all(), "Non-finite values in decoder W_d"

    col_norms = model.W_d.detach().norm(dim=0)
    max_dev = (col_norms - 1.0).abs().max().item()
    assert max_dev < 1e-3, (
        f"Decoder columns drifted from unit norm by {max_dev:.4g} — "
        "the column-norm constraint isn't being applied"
    )

    print("[smoke] all assertions passed")
    return final


# ----------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="SAE pipeline smoke test")
    p.add_argument("--use-cache", action="store_true",
                   help=f"Read first 10k rows from {ACT_DIR}/ instead of synthetic data")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        run_smoke(
            use_cache=args.use_cache,
            steps=args.steps,
            n=args.n,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except AssertionError as e:
        print(f"[smoke] FAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[smoke] CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
