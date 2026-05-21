"""Lightweight YAML-backed config holder shared by all three trainers.

Only one of `lam`/`beta`/`pi` is consumed per run, selected by `family`. We keep
them on a single dataclass so a W&B sweep can override any single field without
caring which family it applies to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import yaml


@dataclass
class TrainConfig:
    # Shared geometry / loop ----------------------------------------------
    family: str = "sae"               # "sae" | "vae_gauss" | "vae_spikeslab"
    d_model: int = 768
    d_dict: int = 3072
    batch_size: int = 4096
    steps: int = 50_000
    lr: float = 1e-3
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    eval_every: int = 2000
    eval_subsample: int = 50_000      # rows used in mid-training eval
    seed: int = 0
    num_workers: int = 2

    # Family-specific (only one applies per run) --------------------------
    lam: float = 0.01                 # SAE L1 coefficient
    beta: float = 1.0                 # Gaussian VAE β
    pi: float = 0.1                   # Spike-slab prior gate probability
    tau_init: float = 1.0             # Spike-slab Gumbel temperature start
    tau_min: float = 0.3              # Spike-slab Gumbel temperature floor
    kl_warmup_steps: int = 5000       # VAE KL linear warmup

    # SAE resampling ------------------------------------------------------
    resample_every: int = 10_000
    resample_window: int = 5000
    resample_fire_threshold: float = 1e-5

    # Mixed precision -----------------------------------------------------
    amp_dtype: str = "bfloat16"       # bfloat16 (A100) | float16 (T4) | float32

    # Logging -------------------------------------------------------------
    wandb_project: str = "saes-as-vaes"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    # Paths ---------------------------------------------------------------
    activations_dir: str = "activations_data"
    checkpoint_dir: str = "checkpoints"

    # ---------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "TrainConfig":
        """Read a YAML file and apply optional keyword overrides. Unknown
        keys raise — catches typos in sweep configs early."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data.update(overrides)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sweep_param_value(self) -> float:
        """Returns the value of whichever family-specific parameter this run
        is sweeping — useful as a W&B group/annotation key."""
        return {
            "sae": self.lam,
            "vae_gauss": self.beta,
            "vae_spikeslab": self.pi,
        }[self.family]
