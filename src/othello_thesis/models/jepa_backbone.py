"""Backbone factory for JEPA encoders.

The JEPA objectives operate on a small encoder surface: token embeddings,
position embeddings, dropout, a stack of causal blocks, and a final layer norm.
Both ``OthelloGPT`` and ``OthelloMambaAR`` expose that surface, so the objective
logic can stay architecture-agnostic.
"""

from __future__ import annotations

import torch.nn as nn

from othello_thesis.models.mamba import MambaARConfig, OthelloMambaAR
from othello_thesis.models.transformer import GPTConfig, OthelloGPT


def normalize_encoder_architecture(value: str | None) -> str:
    """Normalize encoder architecture aliases used by YAML configs."""
    normalized = (value or "transformer").lower().replace("-", "_")
    aliases = {
        "transformer": "transformer",
        "gpt": "transformer",
        "othello_gpt": "transformer",
        "mamba": "mamba",
        "mamba_ar": "mamba",
        "ssm": "mamba",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown encoder_architecture={value!r}; expected one of: {valid}."
        )
    return aliases[normalized]


def build_jepa_encoder(
    *,
    encoder_architecture: str = "transformer",
    board_size: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    dropout: float,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    mamba_backend: str = "mamba_ssm",
) -> nn.Module:
    """Build the context/target encoder used inside a JEPA objective."""
    architecture = normalize_encoder_architecture(encoder_architecture)
    if architecture == "transformer":
        return OthelloGPT(
            GPTConfig(
                board_size=board_size,
                n_layers=n_layers,
                n_heads=n_heads,
                d_model=d_model,
                dropout=dropout,
            )
        )
    if architecture == "mamba":
        return OthelloMambaAR(
            MambaARConfig(
                board_size=board_size,
                n_layers=n_layers,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                mlp_hidden_mult=0,
                mamba_backend=mamba_backend,
            )
        )
    raise AssertionError(f"Unhandled encoder architecture: {architecture}")
