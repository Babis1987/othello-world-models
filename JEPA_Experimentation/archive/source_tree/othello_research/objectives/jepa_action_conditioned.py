"""Action-conditioned one-step JEPA objective (v7).

The context encoder observes moves ``[0:t]``. A separate action embedding
represents move ``t+1`` and is concatenated with the final context state. The
predictor maps that pair to the EMA target encoder's final state after
observing the nested sequence ``[0:t+1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.models.gpt import GPTConfig, OthelloGPT


@dataclass
class JEPAActionConditionedConfig:
    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    predictor_type: str = "action_conditioned_mlp"
    predictor_hidden_mult: int = 4
    predictor_n_layers: int = 2
    predictor_activation: str = "gelu"
    predictor_dropout: float = 0.0
    action_dim: int | None = None

    use_ema_target: bool = True
    ema_momentum: float = 0.996
    prediction_horizon: int = 1
    view_mode: str = "nested"

    loss_mode: str = "smooth_l1"
    hybrid_base: str = "grouped_infonce"
    smooth_l1_beta: float = 1.0
    normalize_embeddings: bool | None = None
    action_temperature: float = 0.1
    action_temperature_learnable: bool = False
    lambda_ce: float = 0.05

    variant: str = "v7"
    loss_type: str = "smooth_l1"


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown predictor activation: {name!r}")


class ActionConditionedMLP(nn.Module):
    """Configurable low-capacity MLP used by the full and action-only paths."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        n_layers: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("predictor_n_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), _activation(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def grouped_infonce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: torch.Tensor | float,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE where each sample competes only against its group's targets.

    The action-grouped sampler emits contiguous, uniformly sized groups. Using
    that layout directly avoids a Python loop and repeated CUDA synchronisation
    for every group.
    """
    if predictions.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError("predictions and targets must both have shape (B, d)")
    if group_ids.ndim != 1 or group_ids.size(0) != predictions.size(0):
        raise ValueError("group_ids must have shape (B,)")

    if group_shape is None:
        group_shape = _uniform_group_shape(group_ids)
    groups, members = group_shape
    if groups * members != predictions.size(0):
        raise ValueError("group_shape does not match the batch size")

    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    tau = torch.as_tensor(temperature, device=predictions.device, dtype=predictions.dtype)
    tau = tau.clamp_min(torch.finfo(predictions.dtype).eps)

    grouped_predictions = predictions.reshape(groups, members, -1)
    grouped_targets = targets.reshape(groups, members, -1)
    logits = torch.bmm(grouped_predictions, grouped_targets.transpose(1, 2)) / tau
    labels = torch.arange(members, device=predictions.device).expand(groups, members)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.reshape(-1))
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, accuracy


def _uniform_group_shape(group_ids: torch.Tensor) -> tuple[int, int]:
    """Validate the sampler's contiguous, uniform group layout once."""
    if group_ids.numel() < 2:
        raise ValueError("Grouped InfoNCE requires at least two samples")
    unique = torch.unique_consecutive(group_ids)
    groups = unique.numel()
    if groups < 1 or group_ids.numel() % groups:
        raise ValueError("Grouped InfoNCE groups must have equal sizes")
    members = group_ids.numel() // groups
    if members < 2:
        raise ValueError("Every grouped InfoNCE group must contain at least two samples")
    expected = torch.arange(groups, device=group_ids.device).repeat_interleave(members)
    if not torch.equal(group_ids, expected):
        raise ValueError("group_ids must be contiguous, uniformly sized, and zero-based")
    return groups, members


def _off_diagonal_cosine(x: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2:
        return x.new_tensor(float("nan"))
    normalized = F.normalize(x, dim=-1)
    batch_size = x.size(0)
    off_diagonal_sum = normalized.sum(dim=0).square().sum() - batch_size
    return off_diagonal_sum / (batch_size * (batch_size - 1))


class OthelloJEPAActionConditioned(nn.Module):
    """Jointly trained context encoder with EMA nested target and action input."""

    def __init__(self, cfg: JEPAActionConditionedConfig) -> None:
        super().__init__()
        if not cfg.use_ema_target:
            raise ValueError("jepa_action_conditioned requires use_ema_target=True")
        if cfg.prediction_horizon != 1:
            raise ValueError("v7 currently supports prediction_horizon=1 only")
        if cfg.view_mode != "nested":
            raise ValueError("v7 target view must be nested")
        if cfg.predictor_type not in {"action_conditioned_mlp", "mlp"}:
            raise ValueError("v7 predictor_type must be action_conditioned_mlp or mlp")
        if not (0.0 <= cfg.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1)")
        if cfg.loss_mode not in {"smooth_l1", "grouped_infonce", "hybrid"}:
            raise ValueError(f"Unknown v7 loss_mode: {cfg.loss_mode!r}")
        if cfg.hybrid_base not in {"smooth_l1", "grouped_infonce"}:
            raise ValueError(f"Unknown hybrid_base: {cfg.hybrid_base!r}")
        if cfg.action_temperature <= 0:
            raise ValueError("action_temperature must be positive")

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
        self.action_only_predictor = ActionConditionedMLP(
            action_dim,
            cfg.d_model,
            hidden_dim=hidden_dim,
            n_layers=cfg.predictor_n_layers,
            activation=cfg.predictor_activation,
            dropout=cfg.predictor_dropout,
        )
        self.next_action_head = nn.Linear(cfg.d_model, gpt_cfg.vocab_size)

        if cfg.action_temperature_learnable:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(cfg.action_temperature)))
        else:
            self.register_buffer(
                "_fixed_temperature",
                torch.tensor(float(cfg.action_temperature)),
                persistent=True,
            )

    @property
    def config(self) -> GPTConfig:
        return self.context_encoder.config

    @property
    def normalize_embeddings(self) -> bool:
        if self.cfg.normalize_embeddings is not None:
            return bool(self.cfg.normalize_embeddings)
        base = self.cfg.hybrid_base if self.cfg.loss_mode == "hybrid" else self.cfg.loss_mode
        return base == "grouped_infonce"

    def train(self, mode: bool = True) -> "OthelloJEPAActionConditioned":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def temperature(self) -> torch.Tensor:
        if hasattr(self, "log_temperature"):
            return self.log_temperature.exp().clamp(1e-4, 100.0)
        return self._fixed_temperature

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
            raise ValueError(f"Expected non-empty (B, T) input, got {idx.shape}")
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

    def predict(self, context_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_latent = self.action_embedding(actions)
        return self.predictor(torch.cat([context_latent, action_latent], dim=-1))

    def _base_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        mode: str,
        group_ids: torch.Tensor | None,
        group_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_embeddings and mode != "grouped_infonce":
            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)
        if mode == "smooth_l1":
            loss = F.smooth_l1_loss(
                predictions,
                targets,
                beta=self.cfg.smooth_l1_beta,
            )
            return loss, -loss.detach()
        if mode == "grouped_infonce":
            if group_ids is None:
                raise ValueError("group_ids are required for grouped_infonce")
            return grouped_infonce_loss(
                predictions,
                targets,
                group_ids,
                self.temperature(),
                group_shape=group_shape,
            )
        raise ValueError(f"Unsupported base loss mode: {mode!r}")

    def forward(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
        target_positions: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del target_positions
        if x_context.ndim != 2 or x_target.ndim != 2:
            raise ValueError("x_context and x_target must have shape (B, T)")
        if x_target.size(1) != x_context.size(1) + 1:
            raise ValueError(
                "v7 requires target to contain exactly the context plus one next action"
            )
        if x_target.size(0) != x_context.size(0):
            raise ValueError("context and target batch sizes differ")

        actions = x_target[:, -1]
        context = self.encode_last_hidden(x_context)
        action_latent = self.action_embedding(actions)
        predictions = self.predictor(torch.cat([context, action_latent], dim=-1))
        with torch.no_grad():
            targets = self.encode_hidden(self.target_encoder, x_target)[:, -1, :]

        base_mode = self.cfg.hybrid_base if self.cfg.loss_mode == "hybrid" else self.cfg.loss_mode
        group_shape = (
            _uniform_group_shape(group_ids)
            if base_mode == "grouped_infonce" and group_ids is not None
            else None
        )
        base_loss, full_metric = self._base_loss(
            predictions,
            targets,
            mode=base_mode,
            group_ids=group_ids,
            group_shape=group_shape,
        )

        ce_loss = predictions.new_zeros(())
        if self.cfg.loss_mode == "hybrid":
            ce_loss = F.cross_entropy(self.next_action_head(context), actions)
        main_loss = base_loss + self.cfg.lambda_ce * ce_loss

        # Detaching the shared action embedding keeps the control from changing
        # the representation used by the main action-conditioned predictor.
        action_only_predictions = self.action_only_predictor(action_latent.detach())
        action_only_loss, action_only_metric = self._base_loss(
            action_only_predictions,
            targets,
            mode=base_mode,
            group_ids=group_ids,
            group_shape=group_shape,
        )
        total_loss = main_loss + action_only_loss

        with torch.no_grad():
            permutation = torch.randperm(context.size(0), device=context.device)
            shuffled_predictions = self.predictor(
                torch.cat([context[permutation], action_latent], dim=-1)
            )
            shuffled_loss, shuffled_metric = self._base_loss(
                shuffled_predictions,
                targets,
                mode=base_mode,
                group_ids=group_ids,
                group_shape=group_shape,
            )
            del shuffled_loss
            context_utility = full_metric - action_only_metric
            c_std = context.std(dim=0, unbiased=False).mean()
            z_std = targets.std(dim=0, unbiased=False).mean()
            p_std = predictions.std(dim=0, unbiased=False).mean()
            cos_sim_offdiag = _off_diagonal_cosine(targets)

        zero = predictions.new_zeros(())
        not_applicable = predictions.new_tensor(float("nan"))
        infonce_loss = base_loss.detach() if base_mode == "grouped_infonce" else zero
        smooth_l1_loss = base_loss.detach() if base_mode == "smooth_l1" else zero
        positive_accuracy = (
            full_metric.detach() if base_mode == "grouped_infonce" else not_applicable
        )
        action_only_accuracy = (
            action_only_metric.detach()
            if base_mode == "grouped_infonce"
            else not_applicable
        )
        return {
            "loss": total_loss,
            "main_loss": main_loss.detach(),
            "base_loss": base_loss.detach(),
            "smooth_l1_loss": smooth_l1_loss,
            "infonce_loss": infonce_loss,
            "hybrid_ce_loss": ce_loss.detach(),
            "action_only_loss": action_only_loss.detach(),
            "full_metric": full_metric.detach(),
            "action_only_metric": action_only_metric.detach(),
            "context_utility": context_utility.detach(),
            "shuffled_context_metric": shuffled_metric.detach(),
            "positive_accuracy": positive_accuracy,
            "action_only_accuracy": action_only_accuracy,
            "c_std": c_std.detach(),
            "z_std": z_std.detach(),
            "p_std": p_std.detach(),
            "cos_sim_offdiag": cos_sim_offdiag.detach(),
            "ema_momentum": predictions.new_tensor(self.cfg.ema_momentum),
            "temperature": self.temperature().detach(),
            "context_latent": context,
            "predictions": predictions,
            "z_targets": targets,
        }
