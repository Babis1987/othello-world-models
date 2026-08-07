"""Contrastive variant of JEPA for discrete strategic sequences.

This module is intentionally separate from :mod:`othello_research.objectives.jepa`.
The predictive JEPA variants minimize a point-to-point embedding distance,
whereas this objective builds an in-batch similarity matrix and optimizes a
multi-positive InfoNCE loss. Keeping that logic separate makes the v5
experiment explicit and reduces regression risk for v1-v4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.models.gpt import GPTConfig, OthelloGPT
from othello_research.models.predictor import build_predictor


@dataclass
class JEPAContrastiveConfig:
    """Configuration for contrastive JEPA.

    The architecture fields follow the same board-size conventions as
    ``JEPAConfig``. v5 deliberately supports only K=1 contrastive prediction;
    multi-step contrastive prediction is a separate future experiment.
    """

    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    view_mode: str = "disjoint_future"
    prediction_horizon: int = 1
    variant: str = "jepa_v5_contrastive"
    loss_type: str = "infonce"

    predictor_type: str = "mlp"
    predictor_hidden_mult: int = 4
    predictor_dropout: float = 0.0
    predictor_n_layers: int = 2
    predictor_activation: str = "gelu"

    use_ema_target: bool = True
    ema_momentum: float = 0.996

    contrastive_temperature: float = 0.1
    contrastive_use_prefix_mask: bool = True


def _normalize_view_mode(view_mode: str) -> str:
    normalized = view_mode.lower().replace("-", "_")
    aliases = {
        "disjoint": "disjoint_future",
        "disjoint_future": "disjoint_future",
        "future": "disjoint_future",
        "hard_disjoint": "hard_disjoint_future",
        "hard_disjoint_future": "hard_disjoint_future",
        "target_only_future": "hard_disjoint_future",
    }
    if normalized not in aliases:
        raise ValueError(
            "Contrastive JEPA supports disjoint_future and hard_disjoint_future views, "
            f"got {view_mode!r}."
        )
    return aliases[normalized]


class OthelloJEPAContrastive(nn.Module):
    """Contrastive JEPA: InfoNCE between predicted and target embeddings."""

    def __init__(self, config: JEPAContrastiveConfig):
        super().__init__()
        config.view_mode = _normalize_view_mode(config.view_mode)
        if config.prediction_horizon != 1:
            raise ValueError("v5 contrastive JEPA supports prediction_horizon=1 only.")
        if not config.use_ema_target:
            raise ValueError("Contrastive JEPA requires an EMA target encoder.")
        if not (0.0 <= config.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if config.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be > 0.")

        self.jepa_config = config
        gpt_config = self._build_gpt_config(config)
        self.context_encoder = OthelloGPT(gpt_config)
        self.target_encoder = OthelloGPT(gpt_config)
        self._copy_to_ema()
        self._freeze_target_encoder()

        predictor_kind = config.predictor_type.lower().replace("-", "_")
        if predictor_kind != "mlp":
            raise ValueError("Contrastive JEPA v5 requires predictor_type='mlp'.")

        self.predictor = build_predictor(
            predictor_kind,
            d_model=config.d_model,
            hidden_dim=config.predictor_hidden_mult * config.d_model,
            hidden_mult=config.predictor_hidden_mult,
            n_layers=config.predictor_n_layers,
            dropout=config.predictor_dropout,
            activation=config.predictor_activation,
            max_context_length=gpt_config.block_size,
            prediction_horizon=1,
        )

    @staticmethod
    def _build_gpt_config(c: JEPAContrastiveConfig) -> GPTConfig:
        return GPTConfig(
            board_size=c.board_size,
            n_layers=c.n_layers,
            n_heads=c.n_heads,
            d_model=c.d_model,
            dropout=c.dropout,
        )

    @property
    def config(self) -> GPTConfig:
        """Expose the context encoder config for loaders and evaluators."""
        return self.context_encoder.config

    def train(self, mode: bool = True) -> "OthelloJEPAContrastive":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _freeze_target_encoder(self) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

    @torch.no_grad()
    def _copy_to_ema(self) -> None:
        for p_t, p_s in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.copy_(p_s.data)

    @torch.no_grad()
    def update_targets(self, momentum: float | None = None) -> None:
        """EMA update of the target encoder."""
        m = self.jepa_config.ema_momentum if momentum is None else momentum
        for p_t, p_s in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """Compatibility alias used by the shared trainer."""
        self.update_targets(momentum)

    def encode_hidden(self, encoder: OthelloGPT, idx: torch.Tensor) -> torch.Tensor:
        """Forward through an OthelloGPT encoder path without the lm_head."""
        if idx.ndim != 2:
            raise ValueError(f"Expected idx with shape (B, T), got {idx.shape}")
        _, seq_len = idx.size()
        if seq_len < 1:
            raise ValueError("JEPA encoder input must contain at least one token.")
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input length {seq_len} exceeds block_size {self.config.block_size}."
            )

        pos = torch.arange(0, seq_len, device=idx.device)
        x = encoder.drop(encoder.wte(idx) + encoder.wpe(pos))
        for block in encoder.blocks:
            x = block(x)
        return encoder.ln_f(x)

    def encode_last_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.encode_hidden(self.context_encoder, idx)[:, -1, :]

    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        """Run contrastive JEPA forward pass."""
        if target_positions is None:
            raise ValueError("target_positions is required for contrastive JEPA.")
        if target_positions.ndim != 2:
            raise ValueError(
                f"Expected target_positions with shape (B, K), got {target_positions.shape}."
            )
        batch_size = x_context.size(0)
        horizon = target_positions.size(1)
        if horizon != 1:
            raise ValueError("v5 contrastive JEPA supports K=1 only; multi-step is deferred.")
        if batch_size < 2:
            raise ValueError("Contrastive JEPA requires batch size >= 2 for in-batch negatives.")
        if target_positions.size(0) != batch_size:
            raise ValueError("target_positions batch size must match x_context.")
        if int(target_positions.max().item()) >= x_target.size(1):
            raise ValueError("target_positions must index valid positions in x_target.")

        c_states = self.encode_hidden(self.context_encoder, x_context)
        c_summary = c_states[:, -1, :]

        with torch.no_grad():
            z_states = self.encode_hidden(self.target_encoder, x_target)
        idx = target_positions.unsqueeze(-1).expand(-1, -1, z_states.size(-1))
        z_targets = torch.gather(z_states, dim=1, index=idx).squeeze(1).detach()

        predictions = self.predictor(c_summary)
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)

        pred_norm = F.normalize(predictions.float(), dim=-1)
        z_norm = F.normalize(z_targets.float(), dim=-1)
        sim_matrix = (pred_norm @ z_norm.T) / self.jepa_config.contrastive_temperature

        if self.jepa_config.contrastive_use_prefix_mask:
            positive_mask = self._compute_prefix_mask(x_context)
        else:
            positive_mask = torch.eye(batch_size, dtype=torch.bool, device=x_context.device)

        loss = self._multi_positive_infonce(sim_matrix, positive_mask)

        with torch.no_grad():
            diagonal = torch.arange(batch_size, device=sim_matrix.device)
            top1_idx = sim_matrix.argmax(dim=-1)
            diag_accuracy = (top1_idx == diagonal).float().mean()
            positive_accuracy = positive_mask[diagonal, top1_idx].float().mean()
            n_positives_mean = positive_mask.float().sum(dim=-1).mean()
            z_std = z_targets.std(dim=0, unbiased=False).mean()
            c_std = c_summary.std(dim=0, unbiased=False).mean()
            p_std = predictions.std(dim=0, unbiased=False).mean()
            pred_raw_norm_mean = predictions.float().norm(dim=-1).mean()
            target_raw_norm_mean = z_targets.float().norm(dim=-1).mean()
            logit_std = sim_matrix.std(unbiased=False)
            logit_max = sim_matrix.max()
            logit_min = sim_matrix.min()

            if batch_size < 2:
                cos_sim_offdiag = torch.tensor(float("nan"), device=x_context.device)
            else:
                sim_z = z_norm @ z_norm.T
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_z.device)
                cos_sim_offdiag = sim_z[mask].mean()

        diagnostics_tensors = {
            "positive_accuracy": positive_accuracy.detach(),
            "diag_accuracy": diag_accuracy.detach(),
            # Deprecated alias: contrastive_accuracy now means positive_accuracy.
            # Use positive_accuracy and diag_accuracy explicitly in new logs.
            "contrastive_accuracy": positive_accuracy.detach(),
            "n_positives_mean": n_positives_mean.detach(),
            "z_std": z_std.detach(),
            "c_std": c_std.detach(),
            "p_std": p_std.detach(),
            "cos_sim_offdiag": cos_sim_offdiag.detach(),
            "pred_raw_norm_mean": pred_raw_norm_mean.detach(),
            "target_raw_norm_mean": target_raw_norm_mean.detach(),
            "logit_std": logit_std.detach(),
            "logit_max": logit_max.detach(),
            "logit_min": logit_min.detach(),
        }
        return {
            "loss": loss,
            **diagnostics_tensors,
            "diagnostics": {
                key: float(value.detach().cpu())
                for key, value in diagnostics_tensors.items()
            },
        }

    def _compute_prefix_mask(self, x_context: torch.Tensor) -> torch.Tensor:
        """Return ``(B, B)`` mask where rows share the same context prefix."""
        a = x_context.unsqueeze(0)
        b = x_context.unsqueeze(1)
        return (a == b).all(dim=-1)

    def _multi_positive_infonce(
        self,
        sim_matrix: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Numerically stable multi-positive InfoNCE."""
        if sim_matrix.ndim != 2 or sim_matrix.size(0) != sim_matrix.size(1):
            raise ValueError(f"Expected square similarity matrix, got {sim_matrix.shape}.")
        if positive_mask.shape != sim_matrix.shape:
            raise ValueError("positive_mask must have the same shape as sim_matrix.")
        if not torch.all(positive_mask.any(dim=-1)):
            raise ValueError("Every row must have at least one positive.")

        sim_stable = sim_matrix - sim_matrix.max(dim=-1, keepdim=True).values.detach()
        exp_sim = sim_stable.exp()
        numerator = (exp_sim * positive_mask.float()).sum(dim=-1).clamp_min(1e-12)
        denominator = exp_sim.sum(dim=-1).clamp_min(1e-12)
        return (-torch.log(numerator / denominator)).mean()


def _smoke() -> None:
    torch.manual_seed(42)
    cfg = JEPAContrastiveConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_hidden_mult=2,
        prediction_horizon=1,
    )
    model = OthelloJEPAContrastive(cfg)
    x = torch.randint(0, model.config.vocab_size, (4, 9))
    target_positions = torch.full((4, 1), 8, dtype=torch.long)
    out = model(x[:, :8], x[:, :9], target_positions)
    assert out["loss"].dim() == 0
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    print(
        "contrastive smoke OK",
        "loss=",
        float(out["loss"].detach()),
        "positive_accuracy=",
        float(out["positive_accuracy"]),
        "diag_accuracy=",
        float(out["diag_accuracy"]),
        "n_positives_mean=",
        float(out["n_positives_mean"]),
    )


if __name__ == "__main__":
    _smoke()
