"""Sparse Autoencoder.

Architecture: x ∈ R^768 → encoder (Linear + ReLU) → z ∈ R^3072_≥0 → decoder
(Linear) → x_hat. Untied encoder and decoder. Decoder columns held at unit
L2 norm by the base class.

Loss: ‖x − x_hat‖² + λ·‖z‖₁, with the L2 term summed over coordinates and
averaged over batch (literature convention for the λ scale).

Encoder initialized as W_d.T (Anthropic's "tied init"): the encoder reads
the same direction the decoder writes, which gives a sane starting point
and noticeably improves the first ~5k training steps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LinearAutoencoder


class SAE(LinearAutoencoder):
    def __init__(
        self,
        d_model: int = 768,
        d_dict: int = 3072,
        lam: float = 0.01,
    ):
        super().__init__(d_model=d_model, d_dict=d_dict)
        # W_e is (d_dict, d_model) so F.linear(x, W_e, b_e) gives (B, d_dict).
        self.W_e = nn.Parameter(torch.empty(d_dict, d_model))
        self.b_e = nn.Parameter(torch.zeros(d_dict))
        self.lam = float(lam)

        # Tied-init: encoder reads in the same direction the decoder writes.
        with torch.no_grad():
            self.W_e.data.copy_(self.W_d.data.t())

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU codes. x_centered = x − b_d (Anthropic decoder-bias trick)."""
        x_c = x - self.b_d
        return F.relu(F.linear(x_c, self.W_e, self.b_e))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z}

    def loss(self, x: torch.Tensor, step: int | None = None) -> dict[str, torch.Tensor]:
        """Returns a dict with `total` (the training objective) plus other
        scalar terms for logging and tensor-valued aux outputs (z, x_hat)
        that the trainer uses for firing-rate tracking and metric eval."""
        del step  # SAE has no step-dependent schedule
        out = self.forward(x)
        # Sum over coordinates, mean over batch: matches the λ convention
        # used by the SAE literature.
        recon = (x - out["x_hat"]).pow(2).sum(dim=-1).mean()
        l1 = out["z"].abs().sum(dim=-1).mean()
        total = recon + self.lam * l1
        return {
            "total": total,
            "recon": recon,
            "l1": l1,
            "z": out["z"],
            "x_hat": out["x_hat"],
        }

    # ------------------------------------------------------------------
    # Dead-feature resampling (Anthropic recipe)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def resample_dead_features(
        self,
        inputs: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        dead_indices: torch.Tensor,
    ) -> int:
        """Re-initialize the rows/cols of dead features using high-recon-loss
        inputs from the current batch.

        Procedure:
          1. Compute per-sample recon loss on `inputs` (forward pass, no grad).
          2. Sample `len(dead_indices)` seed inputs weighted by recon loss.
          3. Each seed becomes a new direction (unit-normalized after b_d
             subtraction); we use it as both the encoder row and decoder
             column for one dead feature.
          4. Reset b_e for those features to 0.
          5. Zero Adam moments for the affected parameter slices so the
             optimizer doesn't immediately undo the reset.

        Returns the number of features actually reset.
        """
        if dead_indices.numel() == 0:
            return 0
        n_dead = int(dead_indices.numel())

        # Per-sample reconstruction loss = weight for the seed distribution.
        out = self.forward(inputs)
        recon_per_sample = (inputs - out["x_hat"]).pow(2).sum(dim=-1)
        probs = (recon_per_sample + 1e-8) / (recon_per_sample + 1e-8).sum()
        seed_idx = torch.multinomial(probs, n_dead, replacement=True)

        seeds = inputs[seed_idx] - self.b_d        # (n_dead, d_model)
        seed_norms = seeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        seeds_unit = seeds / seed_norms

        # Reset parameters.
        self.W_e.data[dead_indices] = seeds_unit              # (n_dead, d_model)
        self.W_d.data[:, dead_indices] = seeds_unit.t()       # (d_model, n_dead)
        self.b_e.data[dead_indices] = 0.0

        self._zero_optimizer_state(optimizer, dead_indices)
        return n_dead

    @torch.no_grad()
    def _zero_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        dead_indices: torch.Tensor,
    ) -> None:
        """Zero Adam first/second-moment buffers for the resampled rows/cols.

        Adam-specific: assumes `exp_avg`/`exp_avg_sq` keys in the state. If
        the optimizer isn't Adam (e.g. SGD has no momentum buffers by
        default) this is a no-op.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if not state:
                    continue
                # Identity check, not value check — p is a Parameter object.
                if p is self.W_e:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key][dead_indices] = 0.0
                elif p is self.W_d:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key][:, dead_indices] = 0.0
                elif p is self.b_e:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key][dead_indices] = 0.0
