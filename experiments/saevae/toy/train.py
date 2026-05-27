"""Training helpers for the toy SAE / VAE models.

Kept minimal and faithful to the notebooks: Adam, full-batch resampling of fresh
synthetic activations each step, optional KL warmup for the VAEs (prevents
posterior collapse by letting the encoder learn to use z before the KL bites).
"""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from .models import ToyModel, ToyModelConfig


def train_toy_model(cfg: ToyModelConfig, feature_probability=0.025, importance=1.0,
                    steps=5000, batch_size=1024, lr=1e-3, device=None, tie_instances=True,
                    progress=True):
    """Train the toy "LLM" and (optionally) tie all instances to instance 0.

    Tying makes every instance the same base model, so an n_inst sweep compares
    inference methods on an identical generative process.
    """
    kwargs = {} if device is None else {"device": device}
    model = ToyModel(cfg, feature_probability=feature_probability,
                     importance=importance, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iterator = tqdm(range(steps), desc="Training toy model") if progress else range(steps)
    for _ in iterator:
        optimizer.zero_grad()
        batch = model.generate_batch(batch_size)
        loss = model.calculate_loss(model(batch), batch)
        loss.backward()
        optimizer.step()
    if tie_instances and cfg.n_inst > 1:
        with torch.no_grad():
            model.W.data[1:] = model.W.data[0]
            model.b_final.data[1:] = model.b_final.data[0]
    return model


def train_sae(model, sae, steps=5000, batch_size=1024, lr=1e-3, progress=True):
    """Train an SAE on the toy model's hidden activations. Returns loss history."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    losses = []
    desc = f"SAE (lam={sae.sparsity_coeff:.3f})"
    iterator = tqdm(range(steps), desc=desc) if progress else range(steps)
    for step in iterator:
        optimizer.zero_grad()
        h, _ = model.get_hidden_activations(batch_size)
        _, _, loss_dict = sae(h)
        loss = loss_dict["loss"].mean()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            losses.append(loss.item())
    return losses


def train_vae(model, vae, steps=5000, batch_size=1024, lr=1e-3, kl_warmup=1000, progress=True):
    """Train a VAE (Gaussian / Laplace / SpikeSlab). Linear KL warmup over ``kl_warmup`` steps.

    The model's ``beta`` is restored to its original value on exit.
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    original_beta = vae.beta
    losses = []
    desc = f"VAE (beta={original_beta:.2f})"
    iterator = tqdm(range(steps), desc=desc) if progress else range(steps)
    for step in iterator:
        vae.beta = original_beta * min(1.0, step / kl_warmup) if kl_warmup > 0 else original_beta
        optimizer.zero_grad()
        h, _ = model.get_hidden_activations(batch_size)
        loss_dict = vae(h)[-1]
        loss = loss_dict["loss"].mean()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            losses.append(loss.item())
    vae.beta = original_beta
    return losses
