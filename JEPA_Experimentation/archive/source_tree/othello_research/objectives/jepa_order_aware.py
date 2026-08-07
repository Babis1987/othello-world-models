"""Order-aware action-conditioned JEPA objective (v8)."""

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
    _uniform_group_shape,
)


@dataclass
class JEPAOrderAwareConfig:
    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    predictor_type: str = "action_conditioned_mlp"
    predictor_hidden_mult: int = 4
    predictor_n_layers: int = 4
    predictor_activation: str = "gelu"
    predictor_dropout: float = 0.0
    action_dim: int | None = None

    use_ema_target: bool = True
    ema_momentum: float = 0.996
    prediction_horizon: int = 1
    view_mode: str = "order_aware"

    temperature: float = 0.1
    learnable_temperature: bool = False
    lambda_ce: float = 0.0

    variant: str = "v8"
    loss_type: str = "order_aware_infonce"


def order_aware_infonce_loss(
    predictions: torch.Tensor,
    positive_targets: torch.Tensor,
    hard_negative_targets: torch.Tensor,
    hard_negative_mask: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: torch.Tensor | float,
    *,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped InfoNCE with per-anchor same-set hard-negative candidates."""
    if predictions.ndim != 2 or positive_targets.shape != predictions.shape:
        raise ValueError("predictions and positive_targets must have shape (B, d)")
    if hard_negative_targets.ndim != 3:
        raise ValueError("hard_negative_targets must have shape (B, H, d)")
    if hard_negative_targets.size(0) != predictions.size(0):
        raise ValueError("hard-negative batch size mismatch")
    if hard_negative_targets.size(2) != predictions.size(1):
        raise ValueError("hard-negative latent dimension mismatch")
    if hard_negative_mask.shape != hard_negative_targets.shape[:2]:
        raise ValueError("hard_negative_mask must have shape (B, H)")
    if group_shape is None:
        group_shape = _uniform_group_shape(group_ids)
    groups, members = group_shape
    if groups * members != predictions.size(0):
        raise ValueError("group_shape does not match batch size")

    pred = F.normalize(predictions.float(), dim=-1).reshape(groups, members, -1)
    positive = F.normalize(positive_targets.float(), dim=-1).reshape(
        groups, members, -1
    )
    tau = torch.as_tensor(temperature, device=pred.device, dtype=pred.dtype)
    tau = tau.clamp_min(torch.finfo(pred.dtype).eps)
    standard_logits = torch.bmm(pred, positive.transpose(1, 2)) / tau

    hard_count = hard_negative_targets.size(1)
    if hard_count:
        hard = F.normalize(hard_negative_targets.float(), dim=-1).reshape(
            groups, members, hard_count, -1
        )
        hard_logits = torch.einsum("gmd,gmhd->gmh", pred, hard) / tau
        hard_mask = hard_negative_mask.reshape(groups, members, hard_count)
        hard_logits = hard_logits.masked_fill(~hard_mask, float("-inf"))
        logits = torch.cat([standard_logits, hard_logits], dim=-1)
    else:
        logits = standard_logits

    labels = torch.arange(members, device=pred.device).expand(groups, members)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.reshape(-1))
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    mean_hard_negatives = hard_negative_mask.float().sum(dim=-1).mean()
    return loss, accuracy, mean_hard_negatives


class OthelloJEPAOrderAware(nn.Module):
    """Joint context encoder and EMA target trained with order-aware views."""

    def __init__(self, cfg: JEPAOrderAwareConfig) -> None:
        super().__init__()
        if not cfg.use_ema_target:
            raise ValueError("jepa_order_aware requires use_ema_target=True")
        if cfg.prediction_horizon != 1:
            raise ValueError("v8 currently supports prediction_horizon=1 only")
        if cfg.view_mode != "order_aware":
            raise ValueError("v8 view_mode must be order_aware")
        if cfg.predictor_type not in {"action_conditioned_mlp", "mlp"}:
            raise ValueError("v8 currently supports an action-conditioned MLP predictor")
        if cfg.temperature <= 0:
            raise ValueError("temperature must be positive")
        if cfg.lambda_ce < 0:
            raise ValueError("lambda_ce must be non-negative")

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
        predictor_kwargs = {
            "hidden_dim": hidden_dim,
            "n_layers": cfg.predictor_n_layers,
            "activation": cfg.predictor_activation,
            "dropout": cfg.predictor_dropout,
        }
        self.action_embedding = nn.Embedding(gpt_cfg.vocab_size, action_dim)
        self.predictor = ActionConditionedMLP(
            cfg.d_model + action_dim, cfg.d_model, **predictor_kwargs
        )
        self.action_only_predictor = ActionConditionedMLP(
            action_dim, cfg.d_model, **predictor_kwargs
        )
        self.next_action_head = nn.Linear(cfg.d_model, gpt_cfg.vocab_size)

        if cfg.learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(cfg.temperature)))
        else:
            self.register_buffer(
                "_fixed_temperature",
                torch.tensor(float(cfg.temperature)),
                persistent=True,
            )

    @property
    def config(self) -> GPTConfig:
        return self.context_encoder.config

    def train(self, mode: bool = True) -> "OthelloJEPAOrderAware":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def temperature(self) -> torch.Tensor:
        if hasattr(self, "log_temperature"):
            return self.log_temperature.exp().clamp(1e-4, 100.0)
        return self._fixed_temperature

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        value = self.cfg.ema_momentum if momentum is None else momentum
        context_parameters = list(self.context_encoder.parameters())
        target_parameters = list(self.target_encoder.parameters())
        torch._foreach_mul_(target_parameters, value)
        torch._foreach_add_(
            target_parameters,
            context_parameters,
            alpha=1.0 - value,
        )

    @torch.no_grad()
    def update_targets(self, momentum: float | None = None) -> None:
        self.update_target_encoder(momentum)

    def encode_hidden(self, encoder: OthelloGPT, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2 or idx.size(1) < 1:
            raise ValueError(f"Expected non-empty (B, T) input, got {idx.shape}")
        if idx.size(1) > self.config.block_size:
            raise ValueError("Input exceeds encoder block size")
        positions = torch.arange(idx.size(1), device=idx.device)
        hidden = encoder.drop(encoder.wte(idx) + encoder.wpe(positions))
        for block in encoder.blocks:
            hidden = block(hidden)
        return encoder.ln_f(hidden)

    def encode_last_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.encode_hidden(self.context_encoder, idx)[:, -1, :]

    def predict(self, context_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_latent = self.action_embedding(actions)
        return self.predictor(torch.cat([context_latent, action_latent], dim=-1))

    def _loss(
        self,
        predictions: torch.Tensor,
        positive_targets: torch.Tensor,
        hard_negative_targets: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        group_ids: torch.Tensor,
        group_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return order_aware_infonce_loss(
            predictions,
            positive_targets,
            hard_negative_targets,
            hard_negative_mask,
            group_ids,
            self.temperature(),
            group_shape=group_shape,
        )

    def forward(
        self,
        x_context: torch.Tensor,
        x_positive_target: torch.Tensor,
        x_hard_negative_targets: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        group_ids: torch.Tensor,
        positive_fallback_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x_context.ndim != 2 or x_positive_target.ndim != 2:
            raise ValueError("context and positive target must have shape (B, T)")
        if x_positive_target.size(1) != x_context.size(1) + 1:
            raise ValueError("Positive target must be a partner prefix plus one action")
        if x_hard_negative_targets.ndim != 3:
            raise ValueError("Hard-negative targets must have shape (B, H, T+1)")
        if x_hard_negative_targets.shape[:1] != x_context.shape[:1]:
            raise ValueError("Hard-negative batch size mismatch")
        if x_hard_negative_targets.size(-1) != x_positive_target.size(1):
            raise ValueError("Hard-negative sequence length mismatch")

        actions = x_positive_target[:, -1]
        context = self.encode_last_hidden(x_context)
        action_latent = self.action_embedding(actions)
        predictions = self.predictor(torch.cat([context, action_latent], dim=-1))
        with torch.no_grad():
            batch_size, hard_count, target_length = x_hard_negative_targets.shape
            flat_hard_mask = hard_negative_mask.reshape(-1)
            flat_hard_sequences = x_hard_negative_targets.reshape(
                batch_size * hard_count,
                target_length,
            )
            target_sequences = torch.cat(
                [x_positive_target, flat_hard_sequences[flat_hard_mask]],
                dim=0,
            )

            target_latents = self.encode_hidden(
                self.target_encoder,
                target_sequences,
            )[:, -1, :]
            positive_targets = target_latents[:batch_size]
            if hard_count:
                flat_hard_targets = positive_targets.new_zeros(
                    batch_size * hard_count,
                    positive_targets.size(-1),
                )
                flat_hard_targets[flat_hard_mask] = target_latents[batch_size:]
                hard_targets = flat_hard_targets.reshape(
                    batch_size,
                    hard_count,
                    -1,
                )
            else:
                hard_targets = positive_targets.new_empty(
                    batch_size,
                    0,
                    positive_targets.size(-1),
                )

        group_shape = _uniform_group_shape(group_ids)
        base_loss, full_metric, mean_hard_negatives = self._loss(
            predictions,
            positive_targets,
            hard_targets,
            hard_negative_mask,
            group_ids,
            group_shape,
        )

        ce_loss = predictions.new_zeros(())
        if self.cfg.lambda_ce > 0:
            ce_loss = F.cross_entropy(self.next_action_head(context), actions)
        main_loss = base_loss + self.cfg.lambda_ce * ce_loss

        action_only_predictions = self.action_only_predictor(action_latent.detach())
        action_only_loss, action_only_metric, _ = self._loss(
            action_only_predictions,
            positive_targets,
            hard_targets,
            hard_negative_mask,
            group_ids,
            group_shape,
        )
        total_loss = main_loss + action_only_loss

        with torch.no_grad():
            groups, members = group_shape
            shuffled_context = context.reshape(groups, members, -1).roll(
                shifts=1, dims=1
            ).reshape_as(context)
            shuffled_predictions = self.predictor(
                torch.cat([shuffled_context, action_latent], dim=-1)
            )
            _, shuffled_metric, _ = self._loss(
                shuffled_predictions,
                positive_targets,
                hard_targets,
                hard_negative_mask,
                group_ids,
                group_shape,
            )
            fallback_rate = (
                positive_fallback_mask.float().mean()
                if positive_fallback_mask is not None
                else predictions.new_tensor(float("nan"))
            )
            c_std = context.std(dim=0, unbiased=False).mean()
            z_std = positive_targets.std(dim=0, unbiased=False).mean()
            p_std = predictions.std(dim=0, unbiased=False).mean()

        return {
            "loss": total_loss,
            "main_loss": main_loss.detach(),
            "infonce_loss": base_loss.detach(),
            "hybrid_ce_loss": ce_loss.detach(),
            "action_only_loss": action_only_loss.detach(),
            "full_metric": full_metric.detach(),
            "action_only_metric": action_only_metric.detach(),
            "context_utility": (full_metric - action_only_metric).detach(),
            "shuffled_context_metric": shuffled_metric.detach(),
            "positive_accuracy": full_metric.detach(),
            "action_only_accuracy": action_only_metric.detach(),
            "positive_fallback_rate": fallback_rate.detach(),
            "mean_hard_negatives": mean_hard_negatives.detach(),
            "c_std": c_std.detach(),
            "z_std": z_std.detach(),
            "p_std": p_std.detach(),
            "cos_sim_offdiag": _off_diagonal_cosine(positive_targets).detach(),
            "ema_momentum": predictions.new_tensor(self.cfg.ema_momentum),
            "temperature": self.temperature().detach(),
            "context_latent": context,
            "predictions": predictions,
            "z_targets": positive_targets,
        }
