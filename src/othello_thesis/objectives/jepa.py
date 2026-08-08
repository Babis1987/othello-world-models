"""Final thesis JEPA objective: hard-disjoint all-position InfoNCE.

The historical v1-v9 exploration lives under ``JEPA_Experimentation``.  This
active module implements only the configuration that produced the main
Transformer-JEPA and Mamba-JEPA cells: a live context encoder, a separately
constructed frozen EMA target encoder, a linear predictor, one-step
hard-disjoint targets, and InfoNCE.

``variant='jepa_v1'`` is retained solely as the historical topology tag stored
in the six canonical checkpoints.  It must not be confused with the thesis
experiment label: the active objective is the final v5-style contrastive path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_thesis.models.jepa_backbone import build_jepa_encoder
from othello_thesis.models.predictor import build_predictor
from othello_thesis.models.transformer import GPTConfig


def normalize_jepa_variant(variant: str) -> str:
    """Accept only the topology tag used by the final six checkpoints."""
    if variant not in {"v1", "jepa_v1"}:
        raise ValueError(
            "The main experiment supports only variant='v1' "
            "(serialized internally as 'jepa_v1'); historical variants are "
            "available under JEPA_Experimentation."
        )
    return "jepa_v1"


def normalize_jepa_loss_type(loss_type: str, variant: str = "jepa_v1") -> str:
    """Accept only the final InfoNCE loss."""
    normalize_jepa_variant(variant)
    if loss_type != "infonce":
        raise ValueError(
            "The main experiment supports only loss_type='infonce'; "
            "historical losses are available under JEPA_Experimentation."
        )
    return "infonce"


def normalize_jepa_view_mode(view_mode: str) -> str:
    """Accept only length-one hard-disjoint future targets."""
    if view_mode != "hard_disjoint_future":
        raise ValueError(
            "The main experiment supports only "
            "view_mode='hard_disjoint_future'; historical views are available "
            "under JEPA_Experimentation."
        )
    return "hard_disjoint_future"


@dataclass
class JEPAConfig:
    """Serialized model contract for the final JEPA objective.

    Field order is intentionally unchanged from the source checkpoints.  The
    VICReg/Smooth-L1 scalar slots remain serialization-only metadata because
    they are present in canonical ``jepa_config`` dictionaries; the active
    objective never dispatches those historical losses.
    """

    variant: str = "jepa_v1"
    loss_type: str = "infonce"
    view_mode: str = "hard_disjoint_future"

    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    encoder_architecture: str = "transformer"
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mamba_backend: str = "mamba_ssm"

    predictor_type: str = "linear"
    predictor_hidden_dim: int | None = None
    predictor_hidden_mult: int = 4
    predictor_n_layers: int = 4
    predictor_n_heads: int = 8
    predictor_dropout: float | None = None
    predictor_activation: str = "relu"

    # Serialization-only slots retained for canonical checkpoint schema.
    vicreg_lambda: float = 25.0
    vicreg_mu: float = 25.0
    vicreg_nu: float = 1.0
    variance_threshold: float = 1.0
    variance_eps: float = 1e-4

    prediction_horizon: int = 1
    ema_momentum: float = 0.996
    smooth_l1_beta: float = 1.0
    use_ema_target: bool | None = True
    contrastive_temperature: float = 0.1


def infonce_jepa_loss(
    z_pred: torch.Tensor,
    z_tgt: torch.Tensor,
    *,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Compute the canonical in-batch InfoNCE objective."""
    if z_pred.shape != z_tgt.shape:
        raise ValueError(
            "z_pred and z_tgt must have the same shape: "
            f"{z_pred.shape} != {z_tgt.shape}"
        )
    if z_pred.ndim == 3:
        z_pred = z_pred.reshape(-1, z_pred.size(-1))
        z_tgt = z_tgt.reshape(-1, z_tgt.size(-1))
    elif z_pred.ndim != 2:
        raise ValueError(
            "Expected embeddings with shape (B, d_model) or "
            f"(B, K, d_model), got {z_pred.shape}"
        )
    if z_pred.size(0) < 2:
        raise ValueError("InfoNCE requires at least two examples in the batch.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    z_pred = z_pred.float()
    z_tgt = z_tgt.float()
    pred_n = F.normalize(z_pred, dim=-1)
    tgt_n = F.normalize(z_tgt, dim=-1)
    logits = pred_n @ tgt_n.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        sims = pred_n @ tgt_n.T
        diag = sims.diagonal()
        off_mask = ~torch.eye(
            sims.size(0),
            dtype=torch.bool,
            device=sims.device,
        )
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return {
        "loss": loss,
        "infonce_loss": loss.detach(),
        "positive_accuracy": accuracy,
        "positive_sim": diag.mean(),
        "negative_sim": sims[off_mask].mean(),
        "z_pred_std": z_pred.std(dim=0, unbiased=False).mean().detach(),
        "z_tgt_std": z_tgt.std(dim=0, unbiased=False).mean().detach(),
    }


def compute_collapse_stats(
    c_summary: torch.Tensor,
    z_targets: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the source-compatible collapse-monitoring scalars."""
    c_summary = c_summary.detach()
    z_targets = z_targets.detach()
    predictions = predictions.detach()
    batch_size = c_summary.size(0)
    device = c_summary.device

    z_std = z_targets.std(dim=0, unbiased=False).mean()
    c_std = c_summary.std(dim=0, unbiased=False).mean()
    p_std = predictions.std(dim=0, unbiased=False).mean()

    if batch_size < 2:
        cos_sim_offdiag = torch.tensor(float("nan"), device=device)
    else:
        z0 = F.normalize(z_targets[:, 0, :], dim=-1)
        sim_matrix = z0 @ z0.T
        mask = ~torch.eye(
            batch_size,
            dtype=torch.bool,
            device=sim_matrix.device,
        )
        cos_sim_offdiag = sim_matrix[mask].mean()

    return {
        "z_std": z_std.detach(),
        "c_std": c_std.detach(),
        "p_std": p_std.detach(),
        "cos_sim_offdiag": cos_sim_offdiag.detach(),
    }


class OthelloJEPA(nn.Module):
    """Final all-position hard-disjoint contrastive JEPA model."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        cfg.variant = normalize_jepa_variant(cfg.variant)
        cfg.loss_type = normalize_jepa_loss_type(cfg.loss_type, cfg.variant)
        cfg.view_mode = normalize_jepa_view_mode(cfg.view_mode)
        if cfg.predictor_type != "linear":
            raise ValueError("The final JEPA objective requires predictor_type='linear'.")
        if cfg.prediction_horizon != 1:
            raise ValueError("The final JEPA objective requires prediction_horizon=1.")
        if cfg.use_ema_target is not True:
            raise ValueError("The final JEPA objective requires use_ema_target=True.")
        if not (0.0 <= cfg.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if cfg.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive.")

        self.jepa_config = cfg
        self.context_encoder = build_jepa_encoder(
            encoder_architecture=cfg.encoder_architecture,
            board_size=cfg.board_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            mamba_backend=cfg.mamba_backend,
        )
        encoder_config = self.context_encoder.config

        # Preserve source RNG order exactly: construct a fresh target encoder,
        # copy/freeze it, and only then construct the predictor.
        predictor_dropout = (
            cfg.dropout
            if cfg.predictor_dropout is None
            else cfg.predictor_dropout
        )
        self.target_encoder = build_jepa_encoder(
            encoder_architecture=cfg.encoder_architecture,
            board_size=cfg.board_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            mamba_backend=cfg.mamba_backend,
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self._freeze_target_encoder()

        self.predictor = build_predictor(
            cfg.predictor_type,
            cfg.d_model,
            cfg.predictor_hidden_dim,
            hidden_mult=cfg.predictor_hidden_mult,
            n_layers=cfg.predictor_n_layers,
            n_heads=cfg.predictor_n_heads,
            dropout=predictor_dropout,
            activation=cfg.predictor_activation,
            max_context_length=encoder_config.block_size,
            prediction_horizon=cfg.prediction_horizon,
        )

    @property
    def config(self) -> GPTConfig:
        """Expose the context encoder config for data loaders."""
        return self.context_encoder.config

    def train(self, mode: bool = True) -> OthelloJEPA:
        """Keep the frozen EMA target encoder in evaluation mode."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _freeze_target_encoder(self) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """Apply the source-compatible EMA update in parameter order."""
        m = self.jepa_config.ema_momentum if momentum is None else momentum
        for context_param, target_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data.mul_(m).add_(
                context_param.data,
                alpha=1.0 - m,
            )

    def encode_hidden(
        self,
        encoder: nn.Module,
        idx: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward through an encoder path without its native LM head."""
        if idx.ndim != 2:
            raise ValueError(f"Expected idx with shape (B, T), got {idx.shape}")
        batch_size, seq_len = idx.size()
        if seq_len < 1:
            raise ValueError("JEPA encoder input must contain at least one token.")
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input length {seq_len} exceeds block_size {self.config.block_size}."
            )

        if positions is None:
            pos = torch.arange(0, seq_len, device=idx.device)
        else:
            if positions.ndim not in {1, 2}:
                raise ValueError(
                    "Expected positions with shape (T,) or (B, T), "
                    f"got {positions.shape}."
                )
            expected_shape = (
                (seq_len,)
                if positions.ndim == 1
                else (batch_size, seq_len)
            )
            if tuple(positions.shape) != expected_shape:
                raise ValueError(
                    f"Expected positions shape {expected_shape}, "
                    f"got {tuple(positions.shape)}."
                )
            if (
                int(positions.min().item()) < 0
                or int(positions.max().item()) >= self.config.block_size
            ):
                raise ValueError(
                    "Position ids must be within the encoder block size: "
                    f"min={int(positions.min().item())}, "
                    f"max={int(positions.max().item())}, "
                    f"block_size={self.config.block_size}."
                )
            pos = positions.to(device=idx.device, dtype=torch.long)
        tok_emb = encoder.wte(idx)
        pos_emb = encoder.wpe(pos)
        x = encoder.drop(tok_emb + pos_emb)
        for block in encoder.blocks:
            x = block(x)
        return encoder.ln_f(x)

    def _predict_embeddings(
        self,
        context_hidden: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the canonical linear predictor to the last context state."""
        del target_positions
        z_pred = self.predictor(context_hidden[:, -1, :])
        if z_pred.ndim == 2:
            z_pred = z_pred.unsqueeze(1)
        return z_pred

    def _target_embeddings(
        self,
        x_target: torch.Tensor,
        horizon: int,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode hard-disjoint target moves with the frozen target encoder."""
        if horizon != 1:
            raise ValueError("The final JEPA objective supports horizon=1 only.")
        with torch.no_grad():
            target_hidden = self.encode_hidden(
                self.target_encoder,
                x_target,
                positions,
            )
        return target_hidden[:, -horizon:, :]

    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute hard-disjoint one-step InfoNCE for one prefix boundary."""
        if x_target.size(1) != 1:
            raise ValueError(
                "The final JEPA target must contain exactly one move; "
                f"got length {x_target.size(1)}."
            )
        context_hidden = self.encode_hidden(self.context_encoder, x_context)
        predictor_positions = target_positions
        if predictor_positions is None:
            predictor_positions = torch.full(
                (x_context.size(0), 1),
                x_context.size(1),
                device=x_context.device,
                dtype=torch.long,
            )
        if predictor_positions.ndim != 2 or predictor_positions.size(1) != 1:
            raise ValueError(
                "target_positions must have shape (B, 1), "
                f"got {predictor_positions.shape}."
            )

        z_pred = self._predict_embeddings(context_hidden, predictor_positions)
        z_tgt = self._target_embeddings(
            x_target,
            1,
            predictor_positions,
        )
        out = infonce_jepa_loss(
            z_pred,
            z_tgt,
            temperature=self.jepa_config.contrastive_temperature,
        )
        out.update(
            compute_collapse_stats(
                context_hidden[:, -1, :],
                z_tgt,
                z_pred,
            )
        )
        out["predictions"] = z_pred.detach()
        out["z_targets"] = z_tgt.detach()
        out["ema_momentum"] = torch.tensor(
            self.jepa_config.ema_momentum,
            device=x_context.device,
        )
        return out
