"""Shared linear-autoencoder geometry.

`LinearAutoencoder` owns the decoder matrix `W_d` and the pre-encoder bias
`b_d`. SAE, GaussianVAE, and SpikeSlabVAE all subclass it so they share:

  * the same decoder shape and column-norm constraint, and
  * the same Anthropic decoder-bias trick (`b_d` subtracted before encoding
    and added back inside `decode`).

These shared conventions are what make L0 and MSE numbers commensurable
across the three families on the Pareto plot. If you change them, change
them here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LinearAutoencoder(nn.Module):
    """Shared decoder geometry. Subclasses add their own encoder + loss."""

    def __init__(self, d_model: int = 768, d_dict: int = 3072):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_dict
        # Decoder: x_hat = z @ W_d.T + b_d, so W_d is (d_model, d_dict).
        self.W_d = nn.Parameter(torch.empty(d_model, d_dict))
        # Pre-encoder / decoder bias. Subtracted from x before the encoder,
        # added back after the decoder. Initialized to zero; the optimizer
        # learns it during training.
        self.b_d = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_d, a=math.sqrt(5))
        self.normalize_decoder_()

    # ------------------------------------------------------------------
    # Decoder column-norm constraint
    # ------------------------------------------------------------------
    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project decoder columns onto the unit sphere.

        Called once at init and after every optimizer step. Without this,
        the L1 penalty on `z` can be gamed by scaling `W_d` up arbitrarily.
        """
        norms = self.W_d.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_d.data.div_(norms)

    @torch.no_grad()
    def remove_parallel_grad_(self) -> None:
        """Strip the component of `W_d.grad` parallel to `W_d` before the
        optimizer step.

        Optional refinement: keeps unit-norm columns more stable under Adam
        noise than the re-projection alone. Trainer calls this just before
        `optimizer.step()` if enabled in config.
        """
        if self.W_d.grad is None:
            return
        # W_d has shape (d_model, d_dict). Decompose grad into per-column
        # parallel and perpendicular components, keep only perpendicular.
        parallel_coeff = (self.W_d.grad * self.W_d.data).sum(dim=0, keepdim=True)
        self.W_d.grad.sub_(parallel_coeff * self.W_d.data)

    # ------------------------------------------------------------------
    # Reconstruction (shared)
    # ------------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """x_hat = z W_d^T + b_d. Subclasses MUST call this rather than
        re-implementing, so the bias convention stays consistent."""
        return z @ self.W_d.t() + self.b_d
