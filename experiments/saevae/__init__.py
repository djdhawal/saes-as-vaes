"""saevae — SAE-vs-VAE comparison on cached LLM activations.

Foundation modules only at this stage. Model classes (SAE, GaussianVAE,
SpikeSlabVAE) will be re-exported here as they're implemented.
"""

from .base import LinearAutoencoder
from .config import TrainConfig
from .normalize import ActivationNormalizer
from .sae import SAE
from .train import train
from . import data, losses, metrics, seeding

__all__ = [
    "LinearAutoencoder",
    "TrainConfig",
    "ActivationNormalizer",
    "SAE",
    "train",
    "data",
    "losses",
    "metrics",
    "seeding",
]
