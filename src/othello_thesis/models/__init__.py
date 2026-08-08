"""Canonical Transformer, Mamba, and JEPA predictor components."""

from .jepa_backbone import build_jepa_encoder, normalize_encoder_architecture
from .mamba import MambaARConfig, OthelloMambaAR
from .predictor import LinearPredictor, build_predictor
from .transformer import GPTConfig, OthelloGPT

# Thesis-facing aliases keep the architecture explicit without changing the
# underlying classes, state-dict schema, or initialization behaviour.
TransformerConfig = GPTConfig
OthelloTransformerAR = OthelloGPT

__all__ = [
    "GPTConfig",
    "LinearPredictor",
    "MambaARConfig",
    "OthelloGPT",
    "OthelloMambaAR",
    "OthelloTransformerAR",
    "TransformerConfig",
    "build_jepa_encoder",
    "build_predictor",
    "normalize_encoder_architecture",
]
