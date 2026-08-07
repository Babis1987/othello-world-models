"""Transposition-family multi-positive InfoNCE objective (Stage 1).

Model = the standard OthelloGPT encoder + a small projection head. There is
deliberately NO EMA target, NO VICReg, NO predictor, and NO cross-entropy on
moves anywhere: the only training signal is within-family discrimination of
board color configurations. The loss touches only final hidden states, so a
different causal backbone (e.g. Mamba) can be swapped in later without
changing the loss or the sampler.

Loss variant (pinned): sum-of-positives inside the log,

    L_i = -log( sum_{p in P_i} exp(s_ip / tau) / sum_{c in C_i} exp(s_ic / tau) )
        = logsumexp_{C_i}(s/tau) - logsumexp_{P_i}(s/tau)

where C_i = sampled members of the anchor's family (excluding the anchor) and
P_i = same-class members among them. For an encoder whose embeddings carry no
information beyond the shared move-set, logits inside a family are uniform
and L_i sits exactly at the chance plateau log(|C_i| / |P_i|). The headline
metric is ``excess = loss - plateau``: excess < 0 is only reachable by
encoding the color configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.models.gpt import GPTConfig, OthelloGPT


@dataclass
class FamilyInfoNCEConfig:
    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    proj_hidden: int = 512
    proj_dim: int = 128
    temperature: float = 0.1

    variant: str = "family_infonce_v1"


def family_multi_positive_infonce(
    z: torch.Tensor,
    family_ids: torch.Tensor,
    class_ids: torch.Tensor,
    positive_eligible: torch.Tensor,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Compute the within-family multi-positive InfoNCE and its chance plateau.

    Args:
        z: (B, d) L2-normalized projections.
        family_ids / class_ids: (B,) integer group labels.
        positive_eligible: (B,) bool; rows failing the common-prefix leakage
            filter may not serve as positive TARGETS (they remain candidates).
        temperature: softmax temperature tau.
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape (B, d), got {tuple(z.shape)}")
    batch_size = z.size(0)
    sim = (z.float() @ z.float().t()) / float(temperature)
    eye = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    same_family = family_ids.unsqueeze(0) == family_ids.unsqueeze(1)
    cand_mask = same_family & ~eye
    same_class = (class_ids.unsqueeze(0) == class_ids.unsqueeze(1)) & cand_mask
    pos_mask = same_class & positive_eligible.unsqueeze(0)

    n_pos = pos_mask.sum(dim=1)
    n_cand = cand_mask.sum(dim=1)
    n_true_neg = (cand_mask & ~same_class).sum(dim=1)
    anchors = (n_pos >= 1) & (n_true_neg >= 1)
    if not bool(anchors.any()):
        raise ValueError(
            "Batch contains no valid anchors (need >=1 positive and >=1 "
            "negative within the sampled family members)."
        )

    neg_inf = torch.finfo(sim.dtype).min
    lse_cand = torch.logsumexp(sim.masked_fill(~cand_mask, neg_inf), dim=1)
    lse_pos = torch.logsumexp(sim.masked_fill(~pos_mask, neg_inf), dim=1)
    loss_per_anchor = lse_cand - lse_pos
    plateau_per_anchor = torch.log(
        n_cand.float() / n_pos.clamp(min=1).float()
    )

    loss = loss_per_anchor[anchors].mean()
    plateau = plateau_per_anchor[anchors].mean().detach()
    return {
        "loss": loss,
        "plateau": plateau,
        "excess": (loss.detach() - plateau),
        "n_anchors": anchors.sum().detach(),
        "loss_per_anchor": loss_per_anchor.detach(),
        "plateau_per_anchor": plateau_per_anchor.detach(),
        "anchor_mask": anchors,
    }


class OthelloFamilyInfoNCE(nn.Module):
    """OthelloGPT encoder + projection head trained with family InfoNCE."""

    def __init__(self, cfg: FamilyInfoNCEConfig) -> None:
        super().__init__()
        if cfg.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.cfg = cfg
        gpt_cfg = GPTConfig(
            board_size=cfg.board_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.context_encoder = OthelloGPT(gpt_cfg)
        self.projection = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.proj_hidden),
            nn.GELU(),
            nn.Linear(cfg.proj_hidden, cfg.proj_dim),
        )

    @property
    def config(self) -> GPTConfig:
        return self.context_encoder.config

    def encode_hidden(self, encoder: OthelloGPT, idx: torch.Tensor) -> torch.Tensor:
        """Forward an OthelloGPT encoder without the lm_head (repo convention)."""
        if idx.ndim != 2 or idx.size(1) < 1:
            raise ValueError(f"Expected non-empty (B, T) input, got {idx.shape}")
        if idx.size(1) > self.config.block_size:
            raise ValueError("Input exceeds encoder block size")
        positions = torch.arange(idx.size(1), device=idx.device)
        hidden = encoder.drop(encoder.wte(idx) + encoder.wpe(positions))
        for block in encoder.blocks:
            hidden = block(hidden)
        return encoder.ln_f(hidden)

    def encode_states(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Hidden state at position t-1 for each row (causal ⇒ pad-safe)."""
        hidden = self.encode_hidden(self.context_encoder, tokens)
        index = (lengths - 1).clamp(min=0)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), index]

    def project(self, states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(states), dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        family_ids: torch.Tensor,
        class_ids: torch.Tensor,
        positive_eligible: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        states = self.encode_states(tokens, lengths)
        z = self.project(states)
        out = family_multi_positive_infonce(
            z, family_ids, class_ids, positive_eligible, self.cfg.temperature
        )
        out["emb_std"] = states.float().std(dim=0, unbiased=False).mean().detach()
        out["z"] = z.detach()
        return out
