"""Toy-model core: the ARENA superposition "LLM" plus the four inference models.

This is the rigorous, ground-truth lab for the SAE-as-VAE thesis. A small toy
model compresses ``n_features`` ground-truth features into a ``d_hidden`` (=2)
bottleneck, creating superposition. The SAE / VAEs then operate on those 2-D
hidden activations and try to recover which features were active.

Because we *know* the ground truth here, every "VAE beats SAE" claim
(calibration, ambiguity, OOD) can be checked rigorously -- which is impossible
on GPT-2 activations.

All models carry an ``n_inst`` instance dimension so a whole sparsity/prior
sweep can be trained in parallel. Lifted from the project notebooks
(``sae_vae_toy_models.ipynb`` / ``vae_beats_sae_experiments.ipynb``) and kept
deliberately faithful so results reproduce; bug fixes live in ``metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# No MPS auto-select: PyTorch's MPS float32 backend produced silently-corrupted
# gradients in this matmul-heavy loop (see project history). CUDA or CPU only.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# The toy "LLM" (identical to ARENA / Anthropic superposition paper)
# =============================================================================
@dataclass
class ToyModelConfig:
    n_inst: int           # number of parallel models (to sweep sparsity)
    n_features: int = 5   # ground-truth features
    d_hidden: int = 2     # bottleneck dimensions (where superposition happens)


class ToyModel(nn.Module):
    """Forward pass: x -> h = Wx -> x' = ReLU(W^T h + b).

    We only care about the hidden activations h = Wx; the SAE/VAE decompose h
    back into the original features.
    """

    def __init__(self, cfg: ToyModelConfig, feature_probability=0.025, importance=1.0,
                 device=DEVICE):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if isinstance(feature_probability, float):
            feature_probability = torch.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to(
            (cfg.n_inst, cfg.n_features)
        )

        if isinstance(importance, float):
            importance = torch.tensor(importance)
        self.importance = importance.to(device).broadcast_to(
            (cfg.n_inst, cfg.n_features)
        )

        # W: (n_inst, d_hidden, n_features). Column i = where feature i lives in 2-D.
        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.empty((cfg.n_inst, cfg.d_hidden, cfg.n_features)))
        )
        self.b_final = nn.Parameter(torch.zeros((cfg.n_inst, cfg.n_features)))
        self.to(device)

    def generate_batch(self, batch_size: int):
        """Each feature on with prob ``feature_probability``, value ~ U[0, 1] when on.

        Returns shape (batch, n_inst, n_features).
        """
        feat_mag = torch.rand((batch_size, self.cfg.n_inst, self.cfg.n_features),
                              device=self.device)
        feat_seed = torch.rand((batch_size, self.cfg.n_inst, self.cfg.n_features),
                               device=self.device)
        return torch.where(feat_seed <= self.feature_probability, feat_mag, 0.0)

    def forward(self, features):
        """x -> ReLU(W^T W x + b)."""
        h = einops.einsum(features, self.W,
                          "... inst feats, inst hidden feats -> ... inst hidden")
        out = einops.einsum(h, self.W,
                            "... inst hidden, inst hidden feats -> ... inst feats")
        return F.relu(out + self.b_final)

    def calculate_loss(self, out, batch):
        """Importance-weighted MSE."""
        error = self.importance * (out - batch).pow(2)
        return einops.reduce(error, "batch inst feats -> inst", "mean").sum()

    def get_hidden_activations(self, batch_size: int):
        """Generate data and return the 2-D hidden activations h = Wx (and features)."""
        with torch.no_grad():
            features = self.generate_batch(batch_size)
            h = einops.einsum(features, self.W,
                              "batch inst feats, inst hidden feats -> batch inst hidden")
        return h, features


# =============================================================================
# SAE -- the MAP baseline (ReLU point estimate + L1 == MAP under a Laplace prior)
# =============================================================================
class ToySAE(nn.Module):
    def __init__(self, n_inst, d_in, d_sae, sparsity_coeff=0.2, device=DEVICE):
        super().__init__()
        self.n_inst = n_inst
        self.d_in = d_in
        self.d_sae = d_sae
        self.sparsity_coeff = sparsity_coeff

        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_sae))))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_sae, d_in))))
        self.b_enc = nn.Parameter(torch.zeros(n_inst, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(n_inst, d_in))
        self.to(device)

    @property
    def W_dec_normalized(self):
        """Unit-norm decoder columns so the model can't cheat by rescaling."""
        return self.W_dec / (self.W_dec.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self, h):
        h_cent = h - self.b_dec
        pre_acts = einops.einsum(h_cent, self.W_enc,
                                 "batch inst d_in, inst d_in d_sae -> batch inst d_sae") + self.b_enc
        z = F.relu(pre_acts)  # point estimate == MAP inference
        h_recon = einops.einsum(z, self.W_dec_normalized,
                                "batch inst d_sae, inst d_sae d_in -> batch inst d_in") + self.b_dec
        mse = (h_recon - h).pow(2).mean(dim=-1)
        l1 = z.abs().sum(dim=-1)
        loss = mse + self.sparsity_coeff * l1
        return h_recon, z, {"mse": mse, "l1": l1, "loss": loss}


# =============================================================================
# Gaussian VAE -- VI control, no sparsity (prior N(0, I))
# =============================================================================
class GaussianVAE(nn.Module):
    def __init__(self, n_inst, d_in, d_latent, beta=1.0, device=DEVICE):
        super().__init__()
        self.n_inst = n_inst
        self.d_in = d_in
        self.d_latent = d_latent
        self.beta = beta

        self.W_enc_mu = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_latent))))
        self.b_enc_mu = nn.Parameter(torch.zeros(n_inst, d_latent))
        self.W_enc_logvar = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_latent))))
        self.b_enc_logvar = nn.Parameter(torch.zeros(n_inst, d_latent))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_latent, d_in))))
        self.b_dec = nn.Parameter(torch.zeros(n_inst, d_in))
        self.to(device)

    def encode(self, h):
        h_cent = h - self.b_dec
        mu = einops.einsum(h_cent, self.W_enc_mu,
                           "batch inst d_in, inst d_in d_lat -> batch inst d_lat") + self.b_enc_mu
        logvar = einops.einsum(h_cent, self.W_enc_logvar,
                               "batch inst d_in, inst d_in d_lat -> batch inst d_lat") + self.b_enc_logvar
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return einops.einsum(z, self.W_dec,
                             "batch inst d_lat, inst d_lat d_in -> batch inst d_in") + self.b_dec

    def forward(self, h):
        mu, logvar = self.encode(h)
        z = self.reparameterize(mu, logvar)
        h_recon = self.decode(z)
        mse = (h_recon - h).pow(2).mean(dim=-1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        loss = mse + self.beta * kl
        return h_recon, z, mu, logvar, {"mse": mse, "kl": kl, "loss": loss}


# =============================================================================
# Laplace VAE -- VI under the SAME (Laplace/L1) prior the SAE implies via MAP.
# KL(N(mu,sigma^2) || Laplace(0,b)) has a closed form for E_q[|z|], so no MC noise.
# =============================================================================
class LaplaceVAE(nn.Module):
    def __init__(self, n_inst, d_in, d_latent, beta=1.0, laplace_b=0.5, n_mc_samples=5,
                 device=DEVICE):
        super().__init__()
        self.n_inst = n_inst
        self.d_in = d_in
        self.d_latent = d_latent
        self.beta = beta
        self.laplace_b = laplace_b
        self.n_mc_samples = n_mc_samples

        self.W_enc_mu = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_latent))))
        self.b_enc_mu = nn.Parameter(torch.zeros(n_inst, d_latent))
        self.W_enc_logvar = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_latent))))
        self.b_enc_logvar = nn.Parameter(torch.full((n_inst, d_latent), -2.0))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_latent, d_in))))
        self.b_dec = nn.Parameter(torch.zeros(n_inst, d_in))
        self.to(device)

    def encode(self, h):
        h_cent = h - self.b_dec
        mu = einops.einsum(h_cent, self.W_enc_mu,
                           "batch inst d_in, inst d_in d_lat -> batch inst d_lat") + self.b_enc_mu
        logvar = einops.einsum(h_cent, self.W_enc_logvar,
                               "batch inst d_in, inst d_in d_lat -> batch inst d_lat") + self.b_enc_logvar
        return mu, logvar.clamp(-10, 2)

    def decode(self, z):
        return einops.einsum(z, self.W_dec,
                             "batch inst d_lat, inst d_lat d_in -> batch inst d_in") + self.b_dec

    def forward(self, h):
        mu, logvar = self.encode(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        h_recon = self.decode(z)
        mse = (h_recon - h).pow(2).mean(dim=-1)

        # KL = -0.5*(1 + logvar) + log(2b) + E_q[|z|]/b   (log 2pi cancels)
        # E[|z|] for z~N(mu,sigma^2): sigma*sqrt(2/pi)*exp(-mu^2/2sigma^2)
        #                              + mu*(1 - 2*Phi(-mu/sigma))
        b = self.laplace_b
        sigma = std
        mu_over_sigma = mu / (sigma + 1e-8)
        expected_abs_z = (
            sigma * (2.0 / np.pi) ** 0.5 * torch.exp(-0.5 * mu_over_sigma ** 2)
            + mu * (1.0 - 2.0 * torch.distributions.Normal(0, 1).cdf(-mu_over_sigma))
        )
        kl_per_dim = -0.5 * (1 + logvar) + np.log(2 * b) + expected_abs_z / b
        kl = kl_per_dim.sum(dim=-1)
        loss = mse + self.beta * kl
        return h_recon, z, mu, logvar, {"mse": mse, "kl": kl, "loss": loss}


# =============================================================================
# Spike-and-Slab VAE -- VI under p(z_i) = (1-pi) delta_0 + pi N(0,1).
# THE main event: a sparse prior with a full posterior. Binary-Concrete gate.
# =============================================================================
class SpikeSlabVAE(nn.Module):
    def __init__(self, n_inst, d_in, d_latent, beta=1.0, prior_pi=0.1, temperature=0.5,
                 free_bits=0.1, device=DEVICE):
        super().__init__()
        self.n_inst = n_inst
        self.d_in = d_in
        self.d_latent = d_latent
        self.beta = beta
        self.prior_pi = prior_pi
        self.temperature = temperature
        self.free_bits = free_bits

        self.W_enc_gate = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_latent))))
        # Gate bias +2 => sigmoid(2)~0.88: gates start ON so the encoder learns to
        # use latents before the KL pressures the unused ones off (avoids collapse).
        self.b_enc_gate = nn.Parameter(torch.full((n_inst, d_latent), 2.0))
        self.W_enc_mu = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_latent))))
        self.b_enc_mu = nn.Parameter(torch.zeros(n_inst, d_latent))
        self.W_enc_logvar = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_in, d_latent))))
        self.b_enc_logvar = nn.Parameter(torch.full((n_inst, d_latent), -2.0))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((n_inst, d_latent, d_in))))
        self.b_dec = nn.Parameter(torch.zeros(n_inst, d_in))
        self.to(device)

    def encode(self, h):
        h_cent = h - self.b_dec
        gate_logits = einops.einsum(h_cent, self.W_enc_gate,
                                    "batch inst d_in, inst d_in d_lat -> batch inst d_lat") + self.b_enc_gate
        mu = einops.einsum(h_cent, self.W_enc_mu,
                           "batch inst d_in, inst d_in d_lat -> batch inst d_lat") + self.b_enc_mu
        logvar = einops.einsum(h_cent, self.W_enc_logvar,
                               "batch inst d_in, inst d_in d_lat -> batch inst d_lat") + self.b_enc_logvar
        return gate_logits, mu, logvar

    def sample_gate(self, gate_logits):
        """Binary-Concrete (Gumbel-sigmoid) relaxation at train; hard threshold at eval."""
        if self.training:
            u1 = torch.rand_like(gate_logits).clamp(1e-6, 1 - 1e-6)
            u2 = torch.rand_like(gate_logits).clamp(1e-6, 1 - 1e-6)
            g1 = -torch.log(-torch.log(u1))
            g2 = -torch.log(-torch.log(u2))
            return torch.sigmoid((gate_logits + g1 - g2) / self.temperature)
        return (gate_logits > 0).float()

    def decode(self, z):
        return einops.einsum(z, self.W_dec,
                             "batch inst d_lat, inst d_lat d_in -> batch inst d_in") + self.b_dec

    def forward(self, h):
        gate_logits, mu, logvar = self.encode(h)
        logvar = logvar.clamp(-10, 2)
        gate = self.sample_gate(gate_logits)
        std = torch.exp(0.5 * logvar)
        z_continuous = mu + std * torch.randn_like(std)
        z = gate * z_continuous
        h_recon = self.decode(z)

        mse = (h_recon - h).pow(2).mean(dim=-1)

        # (a) gate KL: Bernoulli(q_pi) || Bernoulli(prior_pi)
        q_pi = torch.sigmoid(gate_logits).clamp(1e-6, 1 - 1e-6)
        prior_pi = torch.tensor(self.prior_pi, device=h.device)
        kl_gate_pl = (q_pi * (torch.log(q_pi) - torch.log(prior_pi))
                      + (1 - q_pi) * (torch.log(1 - q_pi) - torch.log(1 - prior_pi)))
        # (b) slab KL: N(mu,sigma^2) || N(0,1), gated by q_pi
        kl_cont_pl = q_pi * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))

        # Free bits per latent: each latent gets free_bits nats free, so the
        # cheapest "turn everything off" solution is no longer optimal.
        kl_per_latent = torch.clamp(kl_gate_pl + kl_cont_pl - self.free_bits, min=0.0)
        kl = kl_per_latent.sum(dim=-1)
        kl_gate = kl_gate_pl.sum(dim=-1)
        loss = mse + self.beta * kl
        return h_recon, z, gate, mu, logvar, {"mse": mse, "kl": kl, "kl_gate": kl_gate, "loss": loss}
