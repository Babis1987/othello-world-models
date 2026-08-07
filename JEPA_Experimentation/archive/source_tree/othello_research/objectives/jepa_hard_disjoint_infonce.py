"""Hard-disjoint action InfoNCE JEPA.

This is the contrastive counterpart of the v6 hard-disjoint action setup:

    - context encoder sees the prefix ``m_1..m_t`` and receives gradients;
    - EMA target encoder sees only the single next-action token ``m_{t+1}``;
    - batches are prefix-grouped, so positives are samples with the same exact
      prefix and negatives are samples with different prefixes;
    - loss is multi-positive InfoNCE, not the v6 distance-margin objective.

For target-only action embeddings, a different-prefix negative can share the
same next-action token as an anchor's positive set. Pushing against that target
would contradict the pull term. We therefore mirror the v6 per-pair phantom
replacement: only for the offending anchor/negative logit, the target embedding
is replaced by the target encoder output for a token unused in the current
batch. The sample remains a normal positive for its own prefix group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.models.gpt import GPTConfig
from othello_research.models.jepa_backbone import build_jepa_encoder
from othello_research.models.predictor import build_predictor
from othello_research.objectives.jepa_hard_disjoint_action import select_phantom_token


@dataclass
class JEPAHardDisjointInfoNCEConfig:
    """Config for prefix-grouped hard-disjoint InfoNCE JEPA."""

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

    predictor_type: str = "mlp"
    predictor_hidden_mult: int = 4
    predictor_n_layers: int = 2
    predictor_activation: str = "gelu"
    predictor_dropout: float = 0.0

    use_ema_target: bool = True
    ema_momentum: float = 0.996

    contrastive_temperature: float = 0.1
    dedup_negative_actions: bool = True

    variant: str = "jepa_v5_hard_disjoint_infonce"
    loss_type: str = "infonce"
    view_mode: str = "hard_disjoint_action"


def _prefix_positive_mask(prefix_ids: torch.Tensor) -> torch.Tensor:
    """Return ``(B, B)`` same-prefix positive mask."""
    return prefix_ids.unsqueeze(0) == prefix_ids.unsqueeze(1)


def _action_collision_mask(
    prefix_ids: torch.Tensor,
    next_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive, negative, and same-action collision masks.

    ``collision[i, j]`` is true when sample ``j`` is from a different prefix
    than anchor ``i`` but its action token appears among anchor ``i``'s
    positives. Those logits are phantom-replaced before InfoNCE.
    """
    same_prefix = _prefix_positive_mask(prefix_ids)
    same_action = next_actions.unsqueeze(0) == next_actions.unsqueeze(1)
    collision = (same_prefix.float() @ same_action.float()) > 0.0
    negative = ~same_prefix
    return same_prefix, negative, collision & negative


def hard_disjoint_infonce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prefix_ids: torch.Tensor,
    next_actions: torch.Tensor,
    z_phantom: torch.Tensor | None,
    *,
    temperature: float,
    dedup_negative_actions: bool,
) -> dict[str, torch.Tensor]:
    """Compute multi-positive InfoNCE with optional per-pair phantom logits."""
    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions (B, d), got {predictions.shape}")
    if targets.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError(
            f"Expected targets with same shape as predictions: "
            f"{targets.shape} != {predictions.shape}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    batch_size = predictions.size(0)
    device = predictions.device
    positive_mask, negative_mask, collision_mask = _action_collision_mask(
        prefix_ids,
        next_actions,
    )

    pred_norm = F.normalize(predictions.float(), dim=-1)
    target_norm = F.normalize(targets.float(), dim=-1)
    logits = (pred_norm @ target_norm.T) / temperature

    if dedup_negative_actions and collision_mask.any():
        if z_phantom is not None:
            phantom_norm = F.normalize(z_phantom.float(), dim=-1)
            phantom_logits = (pred_norm @ phantom_norm) / temperature
            logits = torch.where(
                collision_mask,
                phantom_logits.unsqueeze(1).expand_as(logits),
                logits,
            )
        else:
            # Extremely rare when every legal action token appears in the
            # batch. Dropping these denominator terms is preferable to pushing
            # against a target that is also a positive action for this anchor.
            logits = logits.masked_fill(collision_mask, float("-inf"))

    finite_logits = logits[torch.isfinite(logits)]
    sim_max = logits.max(dim=-1, keepdim=True).values.detach()
    sim_stable = logits - sim_max
    exp_logits = sim_stable.exp()
    numerator = (exp_logits * positive_mask.float()).sum(dim=-1).clamp_min(1e-12)
    denominator = exp_logits.sum(dim=-1).clamp_min(1e-12)
    loss = (-torch.log(numerator / denominator)).mean()

    with torch.no_grad():
        diagonal = torch.arange(batch_size, device=device)
        top1_idx = logits.argmax(dim=-1)
        diag_accuracy = (top1_idx == diagonal).float().mean()
        positive_accuracy = positive_mask[diagonal, top1_idx].float().mean()
        n_positives_mean = positive_mask.float().sum(dim=-1).mean()
        n_negatives_mean = negative_mask.float().sum(dim=-1).mean()
        n_collisions = collision_mask.float().sum()
        n_negatives_raw = negative_mask.float().sum().clamp_min(1.0)
        dedup_fraction = n_collisions / n_negatives_raw
        z_std = targets.std(dim=0, unbiased=False).mean()
        p_std = predictions.std(dim=0, unbiased=False).mean()
        pred_raw_norm_mean = predictions.float().norm(dim=-1).mean()
        target_raw_norm_mean = targets.float().norm(dim=-1).mean()
        logit_std = finite_logits.std(unbiased=False) if finite_logits.numel() else torch.tensor(float("nan"), device=device)
        logit_max = finite_logits.max() if finite_logits.numel() else torch.tensor(float("nan"), device=device)
        logit_min = finite_logits.min() if finite_logits.numel() else torch.tensor(float("nan"), device=device)
        if batch_size < 2:
            cos_sim_offdiag = torch.tensor(float("nan"), device=device)
        else:
            sim_z = target_norm @ target_norm.T
            off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            cos_sim_offdiag = sim_z[off_diag].mean()

    return {
        "loss": loss,
        "infonce_loss": loss.detach(),
        "positive_accuracy": positive_accuracy.detach(),
        "diag_accuracy": diag_accuracy.detach(),
        # Deprecated alias kept for log compatibility. It means positive top-1.
        "contrastive_accuracy": positive_accuracy.detach(),
        "n_positives_mean": n_positives_mean.detach(),
        "positives_per_anchor": n_positives_mean.detach(),
        "negatives_per_anchor": n_negatives_mean.detach(),
        "dedup_fraction": dedup_fraction.detach(),
        "z_std": z_std.detach(),
        "p_std": p_std.detach(),
        "cos_sim_offdiag": cos_sim_offdiag.detach(),
        "pred_raw_norm_mean": pred_raw_norm_mean.detach(),
        "target_raw_norm_mean": target_raw_norm_mean.detach(),
        "logit_std": logit_std.detach(),
        "logit_max": logit_max.detach(),
        "logit_min": logit_min.detach(),
    }


class OthelloJEPAHardDisjointInfoNCE(nn.Module):
    """Prefix-grouped hard-disjoint JEPA with multi-positive InfoNCE."""

    def __init__(self, cfg: JEPAHardDisjointInfoNCEConfig):
        super().__init__()
        if not cfg.use_ema_target:
            raise ValueError("hard-disjoint InfoNCE requires use_ema_target=True.")
        if not (0.0 <= cfg.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if cfg.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be > 0.")
        predictor_type = cfg.predictor_type.lower().replace("-", "_")
        if predictor_type != "mlp":
            raise ValueError("hard-disjoint InfoNCE currently requires predictor_type='mlp'.")

        self.cfg = cfg
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
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

        self.predictor = build_predictor(
            "mlp",
            d_model=cfg.d_model,
            hidden_dim=cfg.predictor_hidden_mult * cfg.d_model,
            hidden_mult=cfg.predictor_hidden_mult,
            n_layers=cfg.predictor_n_layers,
            dropout=cfg.predictor_dropout,
            activation=cfg.predictor_activation,
            max_context_length=encoder_config.block_size,
            prediction_horizon=1,
        )
        self._vocab_actions = cfg.board_size * cfg.board_size - 4

    @property
    def config(self) -> GPTConfig:
        return self.context_encoder.config

    def train(self, mode: bool = True) -> "OthelloJEPAHardDisjointInfoNCE":
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        m = self.cfg.ema_momentum if momentum is None else momentum
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_param.data.mul_(m).add_(ctx_param.data, alpha=1.0 - m)

    @torch.no_grad()
    def update_targets(self, momentum: float | None = None) -> None:
        self.update_target_encoder(momentum)

    def encode_hidden(self, encoder: nn.Module, idx: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
    def _encode_target_action(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if action_tokens.ndim != 1:
            raise ValueError(f"Expected action_tokens (B,), got {action_tokens.shape}")
        return self.encode_hidden(self.target_encoder, action_tokens.unsqueeze(1))[:, 0, :]

    def forward(
        self,
        x_context: torch.Tensor,
        next_actions: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        if next_actions.size(0) != x_context.size(0):
            raise ValueError("next_actions and x_context batch size mismatch")
        if prefix_ids.size(0) != x_context.size(0):
            raise ValueError("prefix_ids and x_context batch size mismatch")
        if x_context.size(0) < 2:
            raise ValueError("hard-disjoint InfoNCE requires batch size >= 2.")

        c_summary = self.encode_last_hidden(x_context)
        predictions = self.predictor(c_summary)
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)

        z_target = self._encode_target_action(next_actions).detach()
        phantom_token = select_phantom_token(next_actions, self._vocab_actions)
        if phantom_token is None:
            z_phantom = None
        else:
            phantom = torch.full(
                (1,),
                phantom_token,
                dtype=next_actions.dtype,
                device=next_actions.device,
            )
            z_phantom = self._encode_target_action(phantom)[0].detach()

        out = hard_disjoint_infonce_loss(
            predictions,
            z_target,
            prefix_ids,
            next_actions,
            z_phantom,
            temperature=self.cfg.contrastive_temperature,
            dedup_negative_actions=self.cfg.dedup_negative_actions,
        )
        with torch.no_grad():
            out["c_std"] = c_summary.std(dim=0, unbiased=False).mean().detach()
            out["ema_momentum"] = torch.tensor(self.cfg.ema_momentum, device=x_context.device)
            diagnostics_keys = [
                "positive_accuracy",
                "diag_accuracy",
                "contrastive_accuracy",
                "n_positives_mean",
                "positives_per_anchor",
                "negatives_per_anchor",
                "dedup_fraction",
                "z_std",
                "c_std",
                "p_std",
                "cos_sim_offdiag",
                "pred_raw_norm_mean",
                "target_raw_norm_mean",
                "logit_std",
                "logit_max",
                "logit_min",
            ]
            out["diagnostics"] = {
                key: float(out[key].detach().cpu())
                for key in diagnostics_keys
                if key in out
            }
        return out


def _smoke() -> None:
    torch.manual_seed(42)
    cfg = JEPAHardDisjointInfoNCEConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
    )
    model = OthelloJEPAHardDisjointInfoNCE(cfg)
    x_context = torch.randint(0, model.config.vocab_size, (8, 4))
    prefix_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    x_context[:4] = x_context[0]
    x_context[4:] = x_context[4]
    next_actions = torch.randint(0, model._vocab_actions, (8,))
    next_actions[4] = next_actions[0]
    out = model(x_context, next_actions, prefix_ids)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    print(
        "hard-disjoint InfoNCE smoke OK",
        "loss=", float(out["loss"].detach()),
        "pos_acc=", float(out["positive_accuracy"]),
        "dedup=", float(out["dedup_fraction"]),
    )


if __name__ == "__main__":
    _smoke()
