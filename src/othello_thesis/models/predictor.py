"""Predictor heads for Othello JEPA pretraining."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from othello_thesis.models.transformer import TransformerBlock


# ============================================================================
# Initialisation
# ============================================================================

def _init_projection(module: nn.Module) -> None:
    """Initialise linear projection layers used in simple predictor heads."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _init_transformer_weights(module: nn.Module) -> None:
    """Initialise transformer predictor weights in the OthelloGPT style."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


# ============================================================================
# Predictor heads
# ============================================================================

class IdentityPredictor(nn.Module):
    """Parameter-free predictor for direct context-to-target matching.

    This is intentionally restricted to a single target position. It keeps the
    common JEPA predictor interface while applying no learned transformation.
    """

    def __init__(self, prediction_horizon: int = 1):
        super().__init__()
        if prediction_horizon != 1:
            raise ValueError("IdentityPredictor requires prediction_horizon=1.")
        self.prediction_horizon = prediction_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _activation(name: str) -> nn.Module:
    """Build an activation module by config name."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown predictor activation: {name!r}.")


class LinearPredictor(nn.Module):
    """Single linear projection with optional K-step output."""

    def __init__(self, d_model: int, prediction_horizon: int = 1):
        super().__init__()
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be >= 1.")
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.proj = nn.Linear(d_model, prediction_horizon * d_model)
        self.apply(_init_projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embeddings, preserving leading dimensions."""
        y = self.proj(x)
        if self.prediction_horizon == 1:
            return y
        return y.view(*x.shape[:-1], self.prediction_horizon, self.d_model)


class MLPPredictor(nn.Module):
    """Configurable MLP predictor with optional K-step output."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        *,
        n_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        prediction_horizon: int = 1,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("MLP predictor requires n_layers >= 2.")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be >= 1.")
        if hidden_dim is None:
            hidden_dim = d_model
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon

        layers: list[nn.Module] = [nn.Linear(d_model, hidden_dim), _activation(activation)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), _activation(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, prediction_horizon * d_model))
        self.net = nn.Sequential(*layers)
        self.apply(_init_projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embeddings, preserving leading dimensions."""
        y = self.net(x)
        if self.prediction_horizon == 1:
            return y
        return y.view(*x.shape[:-1], self.prediction_horizon, self.d_model)


class TransformerPredictor(nn.Module):
    """Causal transformer predictor that emits K future embedding predictions.

    The module consumes the context encoder's hidden sequence ``(B, T, d)`` and
    appends ``prediction_horizon`` learned query tokens. The query outputs are
    trained to match the EMA target encoder embeddings for the next K moves.
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_context_length: int,
        prediction_horizon: int = 4,
    ):
        super().__init__()
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be >= 1.")
        if max_context_length < 1:
            raise ValueError("max_context_length must be >= 1.")

        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.max_context_length = max_context_length
        self.max_sequence_length = max_context_length + prediction_horizon

        cfg = SimpleNamespace(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            dropout=dropout,
            block_size=self.max_sequence_length,
        )
        self.query_tokens = nn.Parameter(torch.empty(1, prediction_horizon, d_model))
        self.pos_emb = nn.Embedding(self.max_sequence_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        self.apply(_init_transformer_weights)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                std = 0.02 / math.sqrt(2 * n_layers)
                nn.init.normal_(param, mean=0.0, std=std)

    def forward(self, context_hidden: torch.Tensor) -> torch.Tensor:
        """Predict the next K target embeddings from context hidden states."""
        if context_hidden.ndim != 3:
            raise ValueError(
                f"TransformerPredictor expects (B, T, d), got {context_hidden.shape}."
            )
        batch_size, context_length, d_model = context_hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}.")
        if context_length > self.max_context_length:
            raise ValueError(
                f"Context length {context_length} exceeds max_context_length "
                f"{self.max_context_length}."
            )

        queries = self.query_tokens.expand(batch_size, -1, -1)
        x = torch.cat([context_hidden, queries], dim=1)
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device)
        x = self.drop(x + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x[:, -self.prediction_horizon :, :]


class MLPMultiPosPredictor(nn.Module):
    """MLP predictor that emits one embedding per absolute target position.

    The predictor concatenates a context summary with a separate learned
    predictor-position embedding, then applies the same MLP independently to
    each requested future position. The positional embeddings are intentionally
    not shared with the encoder's ``wpe``.

    This architecture is introduced for JEPA v4 alongside the disjoint-future
    view. It is an implementation adaptation for K-position prediction and is
    a confound relative to v2's single predictor architecture.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden_mult: int = 4,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_mult < 1:
            raise ValueError("hidden_mult must be >= 1.")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        self.d_model = d_model
        self.block_size = block_size
        self.pred_pos_embed = nn.Embedding(block_size, d_model)
        hidden_dim = hidden_mult * d_model
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.apply(_init_projection)

    def forward(
        self,
        c_summary: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future embeddings for absolute target positions.

        Args:
            c_summary: Context summary with shape ``(B, d)``.
            target_positions: Absolute target positions with shape ``(B, K)``.

        Returns:
            Predicted embeddings with shape ``(B, K, d)``.
        """
        if c_summary.ndim != 2:
            raise ValueError(f"Expected c_summary with shape (B, d), got {c_summary.shape}.")
        if target_positions.ndim != 2:
            raise ValueError(
                f"Expected target_positions with shape (B, K), got {target_positions.shape}."
            )
        if target_positions.size(0) != c_summary.size(0):
            raise ValueError("target_positions batch size must match c_summary.")
        if target_positions.numel() > 0:
            min_pos = int(target_positions.min().item())
            max_pos = int(target_positions.max().item())
            if min_pos < 0 or max_pos >= self.block_size:
                raise ValueError(
                    f"target_positions must be in [0, {self.block_size}), "
                    f"got min={min_pos} max={max_pos}."
                )

        pos_emb = self.pred_pos_embed(target_positions)
        c_expanded = c_summary.unsqueeze(1).expand(-1, target_positions.size(1), -1)
        x = torch.cat([c_expanded, pos_emb], dim=-1)
        return self.mlp(x)


def build_predictor(
    kind: str,
    d_model: int,
    hidden_dim: int | None = None,
    *,
    hidden_mult: int = 4,
    n_layers: int = 4,
    n_heads: int = 8,
    dropout: float = 0.1,
    activation: str = "relu",
    max_context_length: int | None = None,
    prediction_horizon: int = 1,
) -> nn.Module:
    """Build a predictor head.

    Args:
        kind: Predictor type. Supported values are ``"identity"``,
            ``"linear"``, ``"mlp"``, ``"transformer"``, and
            ``"mlp_multi_pos"``.
        d_model: Input and output embedding dimension.
        hidden_dim: Optional hidden size for the MLP predictor.
        hidden_mult: Hidden multiplier for ``mlp_multi_pos``.
        n_layers: Number of layers for the MLP or transformer predictor.
        n_heads: Number of heads for the transformer predictor.
        dropout: Dropout probability for the MLP or transformer predictor.
        activation: Activation used by the MLP predictor.
        max_context_length: Maximum context length for transformer prediction.
        prediction_horizon: Number of future target embeddings to predict.

    Returns:
        Predictor module mapping encoder embeddings to predicted embeddings.
    """
    normalized = kind.lower()
    if normalized in {"identity", "none", "no_predictor"}:
        return IdentityPredictor(prediction_horizon)
    if normalized == "linear":
        return LinearPredictor(d_model, prediction_horizon)
    if normalized == "mlp":
        return MLPPredictor(
            d_model,
            hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            prediction_horizon=prediction_horizon,
        )
    if normalized == "transformer":
        if max_context_length is None:
            raise ValueError("max_context_length is required for transformer predictor.")
        return TransformerPredictor(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_context_length=max_context_length,
            prediction_horizon=prediction_horizon,
        )
    if normalized in {"mlp_multi_pos", "mlp-multi-pos"}:
        if max_context_length is None:
            raise ValueError("max_context_length is required for mlp_multi_pos predictor.")
        return MLPMultiPosPredictor(
            d_model=d_model,
            hidden_mult=hidden_mult,
            block_size=max_context_length,
            dropout=dropout,
        )
    raise ValueError(
        "Unknown predictor kind: "
        f"{kind!r}. Expected 'identity', 'linear', 'mlp', 'transformer', "
        "or 'mlp_multi_pos'."
    )


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(8, 64)
    identity = build_predictor("identity", d_model=64, prediction_horizon=1)
    identity_out = identity(x)
    assert identity_out is x
    assert sum(p.numel() for p in identity.parameters()) == 0

    for predictor_kind in ("linear", "mlp"):
        predictor = build_predictor(predictor_kind, d_model=64)
        out = predictor(x)
        assert out.shape == x.shape, (predictor_kind, out.shape, x.shape)

        predictor_k = build_predictor(
            predictor_kind,
            d_model=64,
            n_layers=3,
            hidden_dim=128,
            prediction_horizon=4,
        )
        out_k = predictor_k(x)
        assert out_k.shape == (8, 4, 64), (predictor_kind, out_k.shape)

    x_seq = torch.randn(8, 6, 64)
    transformer = build_predictor(
        "transformer",
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_context_length=16,
        prediction_horizon=4,
    )
    out_seq = transformer(x_seq)
    assert out_seq.shape == (8, 4, 64), out_seq.shape

    multi_pos = build_predictor(
        "mlp_multi_pos",
        d_model=64,
        hidden_mult=4,
        dropout=0.0,
        max_context_length=16,
    )
    target_positions = torch.arange(3, 7).unsqueeze(0).expand(8, -1)
    out_multi = multi_pos(x, target_positions)
    assert out_multi.shape == (8, 4, 64), out_multi.shape

    print("OK")
