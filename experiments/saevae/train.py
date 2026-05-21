"""Unified training loop for SAE, GaussianVAE, SpikeSlabVAE.

All three families implement the same minimal interface, so this loop
doesn't need to know which one it's running:

    model.loss(x, step) -> {"total": Tensor, ..., "z": Tensor, "x_hat": Tensor}

Scalar entries (ndim == 0) in the returned dict are logged to W&B. Tensor
entries are used internally (for dead-feature tracking) and ignored by the
logger.

Family-specific behaviour:
  * `cfg.family == "sae"` enables sliding-window firing tracking and
    periodic dead-feature resampling.
  * VAE families read their own step-dependent schedules (KL warmup, τ
    anneal) inside `model.loss(x, step)` — the trainer just passes `step`.

The trainer is W&B-optional. With `use_wandb=False` (default), nothing is
logged externally — handy for the smoke test and for unit tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import metrics
from .config import TrainConfig
from .losses import linear_warmup
from .seeding import set_seed


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def train(
    model: nn.Module,
    cfg: TrainConfig,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    device: str | torch.device | None = None,
    use_wandb: bool = False,
    out_dir: str | None = None,
    x_mean: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Train one model end-to-end. Returns a dict of final eval metrics.

    Args:
        model: a model with `.loss(x, step) -> dict`, `.normalize_decoder_()`,
            and optionally `.resample_dead_features(...)`.
        cfg: training config (see TrainConfig).
        train_loader: yields (B, d_model) batches of normalized activations.
        eval_loader: a separate loader (typically a 50k-row subsample of the
            eval split) for mid-training evals. Caller can re-run with the
            full eval set after `train()` returns.
        device: torch device. Auto-detected if None.
        use_wandb: if True, log scalars via `wandb.log()`. Caller must have
            called `wandb.init(...)` already.
        out_dir: directory for intermediate + final checkpoints. Skipped if
            None (smoke test path).
        x_mean: per-coordinate mean of the eval set, for stable FVE. If None,
            computed from each eval batch (slightly biased on small batches).
    """
    set_seed(cfg.seed)
    device = _resolve_device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    amp_dtype, scaler = _build_amp(cfg.amp_dtype, device)

    # Sliding-window firing counter (only used for SAE resampling).
    firing_counts = torch.zeros(cfg.d_dict, device=device)
    samples_in_window = 0

    train_iter = _infinite_iter(train_loader)
    pbar = tqdm(range(cfg.steps), desc=cfg.family, dynamic_ncols=True)
    train_start = time.time()

    for step in pbar:
        model.train()
        x = next(train_iter).to(device, non_blocking=True)

        # LR warmup.
        lr_now = cfg.lr * linear_warmup(step, cfg.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)

        # Forward + loss under autocast.
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)
        ):
            losses = model.loss(x, step)
            total = losses["total"]

        # Backward + grad clip + step. fp16 needs the GradScaler dance;
        # bf16 / fp32 don't.
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # Decoder column-norm constraint after the parameter update.
        model.normalize_decoder_()

        # SAE firing tracking + periodic resampling.
        if cfg.family == "sae" and "z" in losses:
            with torch.no_grad():
                firing_counts += (losses["z"] > 0).sum(dim=0).float()
                samples_in_window += losses["z"].shape[0]
            if (
                cfg.resample_every > 0
                and step > 0
                and step % cfg.resample_every == 0
            ):
                rate = firing_counts / max(samples_in_window, 1)
                dead_idx = (rate < cfg.resample_fire_threshold).nonzero(as_tuple=True)[0]
                resample_fn = getattr(model, "resample_dead_features", None)
                n_resampled = 0
                if resample_fn is not None:
                    n_resampled = resample_fn(x, optimizer, dead_idx)
                firing_counts.zero_()
                samples_in_window = 0
                if use_wandb:
                    _wandb_log({"resample/n": n_resampled}, step=step)

        # Periodic eval + logging.
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            eval_m = _eval_pass(model, eval_loader, cfg, device, x_mean=x_mean)
            log = _scalar_logs(losses, prefix="train")
            log["train/lr"] = lr_now
            log["train/grad_norm"] = float(grad_norm)
            log.update({f"eval/{k}": v for k, v in eval_m.items()})
            log["decoder_col_norm/mean"] = float(
                model.W_d.detach().norm(dim=0).mean().item()
            )
            log["decoder_col_norm/std"] = float(
                model.W_d.detach().norm(dim=0).std().item()
            )
            if use_wandb:
                _wandb_log(log, step=step)
            pbar.set_postfix(
                mse=f"{eval_m['recon_mse']:.4f}",
                l0=f"{eval_m['l0']:.1f}",
                fve=f"{eval_m['fve']:.3f}",
            )

        # Intermediate checkpoint at the midpoint (defensive against Colab
        # disconnects).
        if out_dir is not None and step > 0 and step == cfg.steps // 2:
            _save_checkpoint(
                os.path.join(out_dir, "intermediate.pt"),
                model, cfg, eval_m if "eval_m" in dir() else {},
                wandb_run_id=_wandb_run_id() if use_wandb else None,
            )

    elapsed = time.time() - train_start

    # Final eval is the responsibility of the caller (with the full eval
    # loader). We return the most recent mid-training eval as a quick summary
    # plus training metadata.
    final = {
        **eval_m,
        "elapsed_seconds": elapsed,
        "final_step": cfg.steps - 1,
    }
    if out_dir is not None:
        _save_checkpoint(
            os.path.join(out_dir, "final.pt"),
            model, cfg, final,
            wandb_run_id=_wandb_run_id() if use_wandb else None,
        )
    return final


# ----------------------------------------------------------------------
# Eval pass
# ----------------------------------------------------------------------
@torch.no_grad()
def _eval_pass(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    x_mean: torch.Tensor | None = None,
) -> dict[str, float]:
    """One pass over the eval loader. Returns scalar metrics."""
    model.eval()
    sse, sst, n = 0.0, 0.0, 0
    l0_sum, l0_count = 0.0, 0
    firing = torch.zeros(cfg.d_dict, device=device)
    for x in loader:
        x = x.to(device, non_blocking=True)
        out = model(x) if not hasattr(model, "loss") else None
        # Use loss() to get x_hat / z so we don't double-forward.
        losses = model.loss(x, step=cfg.steps)
        x_hat = losses["x_hat"]
        z = losses["z"]

        diff = x - x_hat
        sse += diff.pow(2).sum().item()
        if x_mean is None:
            sst += (x - x.mean(dim=0, keepdim=True)).pow(2).sum().item()
        else:
            sst += (x - x_mean.to(device)).pow(2).sum().item()
        n += x.numel()

        # L0: count strictly non-zero entries (correct for SAE ReLU codes
        # and for spike-slab z after hard-gating at eval; for Gaussian VAE,
        # μ is dense — caller should look at the threshold-based variant
        # exposed in metrics.py if they want a different convention).
        l0_sum += (z != 0).sum(dim=-1).float().sum().item()
        l0_count += z.shape[0]

        firing += (z.detach().abs() > 0).sum(dim=0).float()

    recon_mse = sse / max(n, 1)                       # mean over batch & coords
    fve = 1.0 - sse / max(sst, 1e-12)
    l0_val = l0_sum / max(l0_count, 1)
    dead_frac = float((firing == 0).float().mean().item())
    return {
        "recon_mse": float(recon_mse),
        "fve": float(fve),
        "l0": float(l0_val),
        "dead_feature_frac": dead_frac,
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Auto-detect CUDA → CPU. Note: MPS is NOT auto-selected even when
    available — PyTorch's MPS float32 path has produced spurious NaNs on
    this exact SAE training setup. Pass `device="mps"` explicitly if you
    want it; real training runs on CUDA in Colab."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_amp(
    amp_dtype_name: str, device: torch.device
) -> tuple[torch.dtype, torch.amp.GradScaler | None]:
    """Returns (dtype, scaler). Scaler only needed for fp16."""
    name = amp_dtype_name.lower()
    if device.type != "cuda" or name == "float32":
        return torch.float32, None
    if name == "bfloat16":
        return torch.bfloat16, None
    if name == "float16":
        return torch.float16, torch.amp.GradScaler("cuda")
    raise ValueError(f"unknown amp_dtype: {amp_dtype_name!r}")


def _infinite_iter(loader: DataLoader) -> Iterator[torch.Tensor]:
    """Cycles through `loader` indefinitely, re-shuffling each epoch."""
    while True:
        for batch in loader:
            yield batch


def _scalar_logs(losses: dict[str, torch.Tensor], *, prefix: str) -> dict[str, float]:
    """Pull 0-dim tensors out of the loss dict and format as flat floats."""
    out: dict[str, float] = {}
    for k, v in losses.items():
        if torch.is_tensor(v) and v.ndim == 0:
            out[f"{prefix}/{k}"] = float(v.detach().item())
    return out


def _wandb_log(d: dict[str, float], *, step: int) -> None:
    """W&B logging guarded behind a lazy import so the package works without
    wandb installed."""
    import wandb
    wandb.log(d, step=step)


def _wandb_run_id() -> str | None:
    try:
        import wandb
        return wandb.run.id if wandb.run is not None else None
    except Exception:
        return None


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: TrainConfig,
    final_metrics: dict[str, Any],
    *,
    wandb_run_id: str | None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "cfg": cfg.to_dict(),
        "final_metrics": final_metrics,
        "wandb_run_id": wandb_run_id,
        "git_sha": _git_sha(),
    }
    torch.save(payload, path)
    # Also drop a small JSON sidecar for quick inspection without torch.
    sidecar = {
        "cfg": cfg.to_dict(),
        "final_metrics": final_metrics,
        "wandb_run_id": wandb_run_id,
        "git_sha": payload["git_sha"],
    }
    with open(path + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)
