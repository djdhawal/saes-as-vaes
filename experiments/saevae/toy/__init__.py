"""Toy-model lab for the SAE-as-VAE thesis.

The toy superposition model has *known* ground truth, making it the rigorous
home for the "VAE beats SAE" findings (calibration, ambiguity, OOD) that can't
be checked on GPT-2 activations -- and a cheap, fully-local (CPU, seconds)
setting for the Pareto curve, prior-sensitivity, and amortization-gap findings.
"""

from .models import (
    DEVICE,
    GaussianVAE,
    LaplaceVAE,
    SpikeSlabVAE,
    ToyModel,
    ToyModelConfig,
    ToySAE,
)
from .train import train_sae, train_toy_model, train_vae
from .metrics import activations_for_l0, align_features, evaluate

__all__ = [
    "DEVICE",
    "ToyModel",
    "ToyModelConfig",
    "ToySAE",
    "GaussianVAE",
    "LaplaceVAE",
    "SpikeSlabVAE",
    "train_toy_model",
    "train_sae",
    "train_vae",
    "align_features",
    "evaluate",
    "activations_for_l0",
]
