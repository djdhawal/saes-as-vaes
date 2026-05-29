"""Dataset wrappers + the one-time `DataConfig` (split index + norm_scale).

The cached activation memmap is produced by `activations.py` at the project
root. Layout on disk:

    <activations_dir>/activations.dat   # float16, shape (N, d_model)
    <activations_dir>/activations.meta  # JSON with shape, dtype, model, hook
    <activations_dir>/data_config.json  # produced by `prepare_data_config`

`prepare_data_config` is idempotent — call it from any notebook/script;
it reads the existing JSON if present.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .normalize import ActivationNormalizer

# Names match what `activations.py` writes — change here, not there.
ACT_FILE = "activations.dat"
META_FILE = "activations.meta"
DATA_CONFIG_FILE = "data_config.json"


# ----------------------------------------------------------------------
# Memmap helper
# ----------------------------------------------------------------------
def _load_memmap(activations_dir: str) -> tuple[np.memmap, dict]:
    """Reopen the float16 memmap. Returns (array, meta_dict)."""
    meta_path = os.path.join(activations_dir, META_FILE)
    with open(meta_path) as f:
        meta = json.load(f)
    arr = np.memmap(
        os.path.join(activations_dir, ACT_FILE),
        dtype=meta["dtype"],
        mode="r",
        shape=tuple(meta["shape"]),
    )
    return arr, meta


# ----------------------------------------------------------------------
# Split + normalization metadata
# ----------------------------------------------------------------------
@dataclass
class DataConfig:
    """Frozen split + normalization metadata.

    Train rows: [0, train_end).  Eval rows: [train_end, n_total).
    Extraction already shuffles within batches, so a contiguous tail split
    is i.i.d. with the head — no need for a random index permutation.
    """

    norm_scale: float
    n_total: int
    train_end: int
    n_eval: int
    fit_sample_size: int
    activations_meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DataConfig":
        with open(path) as f:
            return cls(**json.load(f))


def prepare_data_config(
    activations_dir: str,
    eval_frac: float = 0.05,
    fit_sample_size: int = 100_000,
    force: bool = False,
) -> DataConfig:
    """Compute (or reload) the split + norm_scale for an activations cache.

    Writes `data_config.json` inside `activations_dir`. Idempotent: returns
    the existing config unchanged unless `force=True`.
    """
    out_path = os.path.join(activations_dir, DATA_CONFIG_FILE)
    if os.path.exists(out_path) and not force:
        return DataConfig.load(out_path)

    arr, meta = _load_memmap(activations_dir)
    n_total, d_model = int(arr.shape[0]), int(arr.shape[1])
    n_eval = int(round(n_total * eval_frac))
    train_end = n_total - n_eval

    # Fit normalizer on a deterministic stride through the TRAIN portion
    # only. linspace gives an evenly-spaced subsample with no RNG.
    fit_size = min(fit_sample_size, train_end)
    sample_idx = np.linspace(0, train_end - 1, fit_size, dtype=np.int64)
    sample = np.asarray(arr[sample_idx], dtype=np.float32)
    normalizer = ActivationNormalizer.fit(sample, d_model=d_model)

    cfg = DataConfig(
        norm_scale=normalizer.norm_scale,
        n_total=n_total,
        train_end=train_end,
        n_eval=n_eval,
        fit_sample_size=fit_size,
        activations_meta=meta,
    )
    cfg.save(out_path)
    return cfg


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class ActivationDataset(Dataset):
    """Memmap-backed dataset over an index array into the cache.

    For the standard train/eval split, pass `np.arange(start, end)`. For
    debug subsets (the smoke test) pass any sorted int64 array. The memmap
    is opened lazily so that DataLoader workers each open their own file
    handle (avoids pickling a memmap across fork).
    """

    def __init__(self, activations_dir: str, indices: np.ndarray, norm_scale: float):
        self.activations_dir = activations_dir
        self.indices = np.asarray(indices, dtype=np.int64)
        self.norm_scale = float(norm_scale)
        self._arr: np.memmap | None = None

    def _arr_lazy(self) -> np.memmap:
        if self._arr is None:
            self._arr, _ = _load_memmap(self.activations_dir)
        return self._arr

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        row = self._arr_lazy()[int(self.indices[i])]
        # Cast float16 -> float32 on the way out. `np.asarray(..., dtype=float32)`
        # also handles the unlikely case of meta saying a different dtype.
        x = torch.from_numpy(np.asarray(row, dtype=np.float32))
        return x * self.norm_scale


# ----------------------------------------------------------------------
# Loader factory
# ----------------------------------------------------------------------
def make_loaders(
    activations_dir: str,
    cfg: DataConfig,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 0,
    eval_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build (train, eval) DataLoaders for the standard contiguous split.

    The train loader uses `RandomSampler` (no replacement, reshuffles each
    epoch). The eval loader is sequential — order doesn't matter for the
    reductions we compute.
    """
    train_indices = np.arange(0, cfg.train_end)
    eval_indices = np.arange(cfg.train_end, cfg.n_total)
    train_ds = ActivationDataset(activations_dir, train_indices, cfg.norm_scale)
    eval_ds = ActivationDataset(activations_dir, eval_indices, cfg.norm_scale)

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=RandomSampler(train_ds, generator=gen),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eval_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return train_loader, eval_loader


def make_subset_loader(
    activations_dir: str,
    cfg: DataConfig,
    n: int,
    batch_size: int,
    *,
    source: str = "eval",
    seed: int = 0,
) -> DataLoader:
    """A sequential loader over the first `n` rows of either split.

    Used for mid-training quick-eval (a 50k subsample of the eval split)
    and for the smoke test.
    """
    if source == "train":
        start, end = 0, cfg.train_end
    elif source == "eval":
        start, end = cfg.train_end, cfg.n_total
    else:
        raise ValueError(f"source must be 'train' or 'eval', got {source!r}")
    n = min(n, end - start)
    indices = np.arange(start, start + n)
    ds = ActivationDataset(activations_dir, indices, cfg.norm_scale)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,            # subset is small; worker fork overhead not worth it
        pin_memory=True,
        drop_last=False,
    )
