"""Multi-action latent rollout JEPA objective (v9).

The context encoder observes a prefix ``x[:t]``. The target encoder is an EMA
copy that observes the nested prefixes ``x[:t+k]``. A shared one-step
action-conditioned predictor is recursively applied with future action tokens:

    z_{t+1} = P(z_t, a_{t+1})
    z_{t+2} = P(z_{t+1}, a_{t+2})
    ...

v9A uses weighted Smooth L1 regression to the EMA target states. v9B uses
weighted per-step in-batch InfoNCE. No board labels, legal-move labels, AR
cross-entropy, or reconstruction targets are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.models.gpt import GPTConfig, OthelloGPT
from othello_research.objectives.jepa_action_conditioned import (
    ActionConditionedMLP,
    _off_diagonal_cosine,
)


@dataclass
class JEPAMultiActionConfig:
    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    predictor_type: str = "action_conditioned_mlp"
    predictor_hidden_mult: int = 4
    predictor_n_layers: int = 3
    predictor_activation: str = "gelu"
    predictor_dropout: float = 0.0
    action_dim: int | None = None

    use_ema_target: bool = True
    ema_momentum: float = 0.996
    prediction_horizon: int = 4
    action_horizon: int = 4
    view_mode: str = "nested"

    rollout_loss_mode: str = "smooth_l1"
    loss_type: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    temperature: float = 0.1
    horizon_weight_gamma: float = 0.8
    normalize_latents_for_loss: bool | None = None
    lambda_var: float = 0.0
    lambda_cov: float = 0.0

    variant: str = "v9"


def _normalize_mode(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    aliases = {
        "auto": "smooth_l1",
        "smooth_l1": "smooth_l1",
        "smoothl1": "smooth_l1",
        "huber": "smooth_l1",
        "infonce": "infonce",
        "info_nce": "infonce",
        "contrastive": "infonce",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown v9 rollout_loss_mode {value!r}. Valid values: {valid}")
    return aliases[normalized]


def _variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.float().var(dim=0, unbiased=False) + eps)
    return F.relu(1.0 - std).mean()


def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z.float()
    if z.size(0) < 2:
        return z.new_zeros(())
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (z.size(0) - 1)
    d_model = cov.size(0)
    off_diag = cov.flatten()[:-1].view(d_model - 1, d_model + 1)[:, 1:].flatten()
    return off_diag.square().sum() / d_model


class OthelloJEPAMultiActionRollout(nn.Module):
    """Pure JEPA multi-step latent rollout conditioned on future action tokens."""

    def __init__(self, cfg: JEPAMultiActionConfig) -> None:
        super().__init__()
        cfg.rollout_loss_mode = _normalize_mode(cfg.rollout_loss_mode)
        cfg.loss_type = _normalize_mode(cfg.loss_type)
        if cfg.loss_type != cfg.rollout_loss_mode:
            cfg.loss_type = cfg.rollout_loss_mode
        if not cfg.use_ema_target:
            raise ValueError("v9 requires use_ema_target=True")
        if cfg.view_mode != "nested":
            raise ValueError("v9 requires view_mode='nested'")
        if cfg.prediction_horizon != cfg.action_horizon:
            raise ValueError("v9 requires prediction_horizon == action_horizon")
        if cfg.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        if not (0.0 <= cfg.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1)")
        if cfg.temperature <= 0:
            raise ValueError("temperature must be positive")
        if cfg.horizon_weight_gamma <= 0:
            raise ValueError("horizon_weight_gamma must be positive")
        if cfg.predictor_type.lower().replace("-", "_") not in {"action_conditioned_mlp", "mlp"}:
            raise ValueError("v9 predictor_type must be action_conditioned_mlp or mlp")

        self.cfg = cfg
        self.jepa_config = cfg
        gpt_cfg = GPTConfig(
            board_size=cfg.board_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.context_encoder = OthelloGPT(gpt_cfg)
        self.target_encoder = OthelloGPT(gpt_cfg)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

        action_dim = cfg.d_model if cfg.action_dim is None else cfg.action_dim
        hidden_dim = max(1, cfg.predictor_hidden_mult) * cfg.d_model
        self.action_embedding = nn.Embedding(gpt_cfg.vocab_size, action_dim)
        self.predictor = ActionConditionedMLP(
            cfg.d_model + action_dim,
            cfg.d_model,
            hidden_dim=hidden_dim,
            n_layers=cfg.predictor_n_layers,
            activation=cfg.predictor_activation,
            dropout=cfg.predictor_dropout,
        )

    @property
    def config(self) -> GPTConfig:
        return self.context_encoder.config

    @property
    def normalize_latents_for_loss(self) -> bool:
        if self.cfg.normalize_latents_for_loss is not None:
            return bool(self.cfg.normalize_latents_for_loss)
        return self.cfg.rollout_loss_mode == "infonce"

    def train(self, mode: bool = True) -> "OthelloJEPAMultiActionRollout":
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        m = self.cfg.ema_momentum if momentum is None else momentum
        for context, target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target.data.mul_(m).add_(context.data, alpha=1.0 - m)

    @torch.no_grad()
    def update_targets(self, momentum: float | None = None) -> None:
        self.update_target_encoder(momentum)

    def encode_hidden(self, encoder: OthelloGPT, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2 or idx.size(1) < 1:
            raise ValueError(f"Expected non-empty (B, T) input, got {tuple(idx.shape)}")
        if idx.size(1) > self.config.block_size:
            raise ValueError(
                f"Input length {idx.size(1)} exceeds block_size {self.config.block_size}"
            )
        pos = torch.arange(idx.size(1), device=idx.device)
        hidden = encoder.drop(encoder.wte(idx) + encoder.wpe(pos))
        for block in encoder.blocks:
            hidden = block(hidden)
        return encoder.ln_f(hidden)

    def encode_last_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.encode_hidden(self.context_encoder, idx)[:, -1, :]

    def predict_step(self, latent: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        action_latent = self.action_embedding(action_tokens)
        return self.predictor(torch.cat([latent, action_latent], dim=-1))

    def predict(self, context_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.predict_step(context_latent, actions)

    def _target_latents(
        self,
        x_target: torch.Tensor,
        target_positions: torch.Tensor | None,
        horizon: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_states = self.encode_hidden(self.target_encoder, x_target)
        if target_positions is None:
            return target_states[:, -horizon:, :].detach()
        if target_positions.ndim != 2 or target_positions.size(1) != horizon:
            raise ValueError(
                f"target_positions must have shape (B, {horizon}), "
                f"got {tuple(target_positions.shape)}"
            )
        if target_positions.size(0) != x_target.size(0):
            raise ValueError("target_positions batch size must match x_target")
        if int(target_positions.min().item()) < 0 or int(target_positions.max().item()) >= x_target.size(1):
            raise ValueError("target_positions must index valid positions in x_target")
        gather_idx = target_positions.unsqueeze(-1).expand(-1, -1, target_states.size(-1))
        return torch.gather(target_states, dim=1, index=gather_idx).detach()

    def _smooth_l1_step_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize_latents_for_loss:
            prediction = F.normalize(prediction, dim=-1)
            target = F.normalize(target, dim=-1)
        return F.smooth_l1_loss(
            prediction,
            target,
            beta=self.cfg.smooth_l1_beta,
        )

    def _infonce_step_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_norm = F.normalize(prediction.float(), dim=-1)
        target_norm = F.normalize(target.float(), dim=-1)
        logits = (pred_norm @ target_norm.T) / self.cfg.temperature
        labels = torch.arange(prediction.size(0), device=prediction.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            top1 = logits.argmax(dim=-1)
            diag_accuracy = (top1 == labels).float().mean()
            positive_sim = (pred_norm * target_norm).sum(dim=-1).mean()
            if prediction.size(0) < 2:
                negative_sim = prediction.new_tensor(float("nan"))
            else:
                sim = pred_norm @ target_norm.T
                mask = ~torch.eye(prediction.size(0), dtype=torch.bool, device=prediction.device)
                negative_sim = sim[mask].mean()
        return loss, {
            "positive_accuracy": diag_accuracy,
            "diag_accuracy": diag_accuracy,
            "positive_sim": positive_sim,
            "negative_sim": negative_sim,
        }

    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        target_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x_context.ndim != 2 or x_target.ndim != 2:
            raise ValueError("x_context and x_target must have shape (B, T)")
        if x_target.size(0) != x_context.size(0):
            raise ValueError("context and target batch sizes differ")
        horizon = self.cfg.action_horizon
        expected_target_len = x_context.size(1) + horizon
        if x_target.size(1) != expected_target_len:
            raise ValueError(
                "v9 target must contain exactly the context plus action_horizon "
                f"future actions: got {x_target.size(1)}, expected {expected_target_len}"
            )

        context_states = self.encode_hidden(self.context_encoder, x_context)
        state = context_states[:, -1, :]
        targets = self._target_latents(x_target, target_positions, horizon)
        future_actions = x_target[:, x_context.size(1): expected_target_len]

        weights = torch.tensor(
            [self.cfg.horizon_weight_gamma ** k for k in range(horizon)],
            device=x_context.device,
            dtype=state.dtype,
        )
        weight_sum = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        predictions: list[torch.Tensor] = []
        step_losses: list[torch.Tensor] = []
        weighted_losses: list[torch.Tensor] = []
        step_cosines: list[torch.Tensor] = []
        info_metrics: dict[str, list[torch.Tensor]] = {
            "positive_accuracy": [],
            "diag_accuracy": [],
            "positive_sim": [],
            "negative_sim": [],
        }

        for step_idx in range(horizon):
            state = self.predict_step(state, future_actions[:, step_idx])
            target = targets[:, step_idx, :]
            predictions.append(state)
            if self.cfg.rollout_loss_mode == "infonce":
                step_loss, metrics = self._infonce_step_loss(state, target)
                for key, value in metrics.items():
                    info_metrics[key].append(value.detach())
            else:
                step_loss = self._smooth_l1_step_loss(state, target)
            step_losses.append(step_loss)
            weighted_losses.append(weights[step_idx] * step_loss)
            with torch.no_grad():
                step_cosines.append(
                    F.cosine_similarity(state.float(), target.float(), dim=-1).mean()
                )

        pred_stack = torch.stack(predictions, dim=1)
        rollout_loss = torch.stack(weighted_losses).sum() / weight_sum
        variance_loss = pred_stack.new_zeros(())
        covariance_loss = pred_stack.new_zeros(())
        if self.cfg.rollout_loss_mode == "smooth_l1":
            flat_predictions = pred_stack.reshape(-1, pred_stack.size(-1))
            if self.cfg.lambda_var:
                variance_loss = _variance_loss(flat_predictions)
            if self.cfg.lambda_cov:
                covariance_loss = _covariance_loss(flat_predictions)
        total_loss = (
            rollout_loss
            + self.cfg.lambda_var * variance_loss
            + self.cfg.lambda_cov * covariance_loss
        )

        with torch.no_grad():
            flat_targets = targets.reshape(-1, targets.size(-1))
            flat_predictions = pred_stack.reshape(-1, pred_stack.size(-1))
            c_std = context_states[:, -1, :].std(dim=0, unbiased=False).mean().detach()
            z_std = flat_targets.std(dim=0, unbiased=False).mean().detach()
            p_std = flat_predictions.std(dim=0, unbiased=False).mean().detach()
            cos_sim_offdiag = _off_diagonal_cosine(flat_targets)
            nan = state.new_tensor(float("nan"))
            if self.cfg.rollout_loss_mode == "infonce":
                positive_accuracy = torch.stack(info_metrics["positive_accuracy"]).mean()
                diag_accuracy = torch.stack(info_metrics["diag_accuracy"]).mean()
                positive_sim = torch.stack(info_metrics["positive_sim"]).mean()
                negative_sim = torch.stack(info_metrics["negative_sim"]).mean()
            else:
                positive_accuracy = nan
                diag_accuracy = nan
                positive_sim = nan
                negative_sim = nan

        diagnostics_tensors: dict[str, torch.Tensor] = {
            "main_loss": total_loss.detach(),
            "rollout_loss": rollout_loss.detach(),
            "base_loss": rollout_loss.detach(),
            "weighted_rollout_loss": rollout_loss.detach(),
            "smooth_l1_loss": rollout_loss.detach()
            if self.cfg.rollout_loss_mode == "smooth_l1" else pred_stack.new_tensor(float("nan")),
            "infonce_loss": rollout_loss.detach()
            if self.cfg.rollout_loss_mode == "infonce" else pred_stack.new_tensor(float("nan")),
            "variance_loss": variance_loss.detach(),
            "covariance_loss": covariance_loss.detach(),
            "positive_accuracy": positive_accuracy.detach(),
            "diag_accuracy": diag_accuracy.detach(),
            "contrastive_accuracy": positive_accuracy.detach(),
            "positive_sim": positive_sim.detach(),
            "negative_sim": negative_sim.detach(),
            "c_std": c_std.detach(),
            "z_std": z_std.detach(),
            "p_std": p_std.detach(),
            "cos_sim_offdiag": cos_sim_offdiag.detach(),
            "ema_momentum": pred_stack.new_tensor(self.cfg.ema_momentum),
            "temperature": pred_stack.new_tensor(self.cfg.temperature),
            "horizon_weight_gamma": pred_stack.new_tensor(self.cfg.horizon_weight_gamma),
            "weight_sum": weight_sum.detach(),
        }
        for step_idx, (step_loss, weighted_loss, cosine) in enumerate(
            zip(step_losses, weighted_losses, step_cosines),
            start=1,
        ):
            diagnostics_tensors[f"step_{step_idx}_loss"] = step_loss.detach()
            diagnostics_tensors[f"step_{step_idx}_weighted_loss"] = (
                weighted_loss / weight_sum
            ).detach()
            diagnostics_tensors[f"step_{step_idx}_cosine"] = cosine.detach()
            if self.cfg.rollout_loss_mode == "infonce":
                diagnostics_tensors[f"step_{step_idx}_positive_accuracy"] = (
                    info_metrics["positive_accuracy"][step_idx - 1].detach()
                )
                diagnostics_tensors[f"step_{step_idx}_positive_sim"] = (
                    info_metrics["positive_sim"][step_idx - 1].detach()
                )
                diagnostics_tensors[f"step_{step_idx}_negative_sim"] = (
                    info_metrics["negative_sim"][step_idx - 1].detach()
                )

        return {
            "loss": total_loss,
            **diagnostics_tensors,
            "context_latent": context_states[:, -1, :].detach(),
            "predictions": pred_stack.detach(),
            "z_targets": targets.detach(),
            "diagnostics": {
                key: float(value.detach().cpu())
                for key, value in diagnostics_tensors.items()
            },
        }


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = JEPAMultiActionConfig(
        board_size=8,
        n_layers=1,
        n_heads=4,
        d_model=32,
        dropout=0.0,
        action_dim=16,
        predictor_hidden_mult=2,
        action_horizon=3,
        prediction_horizon=3,
    )
    model = OthelloJEPAMultiActionRollout(cfg)
    x = torch.randint(0, model.config.vocab_size, (4, 9))
    out = model(x[:, :5], x[:, :8])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


if __name__ == "__main__":
    _smoke()
