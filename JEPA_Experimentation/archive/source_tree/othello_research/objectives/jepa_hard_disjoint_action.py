"""Hard-disjoint action-anchored JEPA (v6).

Design summary:
    - Context branch: causal Transformer over the move history m_1..m_t,
      identical architecture to the OthelloGPT encoder used in v1-v5. The
      context summary c_t is the hidden state at the final token position.
    - Predictor: small multi-mode MLP that maps c_t to ``num_modes`` predicted
      latent vectors {p_1, ..., p_M}.
    - Target branch: same Transformer architecture as context, frozen and
      EMA-updated from the context encoder. Critically, the target encoder
      sees ONLY the single next-action token m_{t+1} (sequence length 1).
      The output is a single target embedding z per sample. The view is
      therefore hard-disjoint: target receives zero context history.
    - Loss: distance-margin (NOT InfoNCE). For each anchor i with prefix P_i:
        positives_i = {j : prefix_j == P_i}      (includes i itself)
        negatives_i = {j : prefix_j != P_i}
      Pull (per positive j): min over modes of ||p_m - z_j||^2.
      Push (per negative j): hinge(margin - min_m ||p_m - z_j||)^2.

      Per-pair phantom replacement (intentional): when next_action_j equals
      one of anchor i's positive actions but the prefixes differ, the target
      encoder produces the same z for j and for that positive, so the push
      term would contradict the pull. The loss therefore swaps z_j for a
      phantom embedding (target encoder applied to a token unused in the
      current batch) ONLY in the push term for this (i, j) pair. The
      anchor's self-positive uses the real target unchanged.

Keeping this module standalone from ``objectives/jepa.py`` and
``objectives/jepa_contrastive.py`` ensures the v1-v5 paths are not perturbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.models.gpt import GPTConfig
from othello_research.models.jepa_backbone import build_jepa_encoder


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class JEPAHardDisjointActionConfig:
    """Configuration for hard-disjoint action-anchored JEPA (v6).

    The defaults mirror the v6 YAML so unit smoke tests can be constructed
    purely from the dataclass without loading a config file.
    """

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

    # Predictor head.
    predictor_hidden_mult: int = 4
    predictor_n_layers: int = 2
    predictor_activation: str = "gelu"
    predictor_dropout: float = 0.0

    num_modes: int = 10

    # Target branch.
    use_ema_target: bool = True
    ema_momentum: float = 0.996

    # Loss.
    margin: float = 1.0
    lambda_push: float = 1.0
    normalize: bool = True
    dedup_negative_actions: bool = True

    # Identifiers carried through from train_jepa.py without affecting logic.
    variant: str = "jepa_v6_hard_disjoint_action"
    loss_type: str = "distance_margin"
    view_mode: str = "hard_disjoint_action"


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown predictor activation: {name!r}.")


# ============================================================================
# Multi-mode predictor head
# ============================================================================

class MultiModeMLPPredictor(nn.Module):
    """MLP head emitting ``num_modes`` predicted latent vectors.

    Input: context summary ``c_t`` of shape ``(B, d_model)``.
    Output: predicted modes ``(B, num_modes, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_modes: int,
        hidden_mult: int = 4,
        n_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1.")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        self.d_model = d_model
        self.num_modes = num_modes

        hidden_dim = max(1, hidden_mult) * d_model
        layers: list[nn.Module] = []
        in_dim = d_model
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_modes * d_model))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, c_summary: torch.Tensor) -> torch.Tensor:
        if c_summary.ndim != 2:
            raise ValueError(
                f"MultiModeMLPPredictor expects (B, d_model), got {c_summary.shape}"
            )
        out = self.net(c_summary)
        return out.view(c_summary.size(0), self.num_modes, self.d_model)


# ============================================================================
# Distance-margin loss with per-pair phantom replacement
# ============================================================================

def _pairwise_min_mode_sq_dist(
    p: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return min-over-modes squared distances and the chosen mode indices.

    Args:
        p: predicted modes ``(B, M, d)``.
        z: target embeddings ``(B, d)``.

    Returns:
        ``(min_sq_dist, mode_idx)`` both with shape ``(B, B)`` where entry
        ``[i, j]`` is the minimum (over modes) of ``||p[i, m] - z[j]||^2``
        and the mode index that achieved it.
    """
    # ``p[:, :, None, :] - z[None, None, :, :]`` materializes a (B, M, B, d)
    # tensor which is fine for the modest batch sizes used here (B ~ 256,
    # M = 10, d = 512: ~1.25 GB at float32 — too much). We instead expand
    # only over modes and reduce immediately.
    B, M, d = p.shape
    # ||p - z||^2 = ||p||^2 + ||z||^2 - 2 p . z
    p_sq = p.pow(2).sum(dim=-1)               # (B, M)
    z_sq = z.pow(2).sum(dim=-1)               # (B,)
    pz = torch.einsum("imd,jd->imj", p, z)    # (B, M, B)
    sq = p_sq.unsqueeze(2) + z_sq.unsqueeze(0).unsqueeze(0) - 2.0 * pz
    sq = sq.clamp_min(0.0)
    min_sq, mode_idx = sq.min(dim=1)          # (B, B), (B, B)
    return min_sq, mode_idx


def hard_disjoint_action_loss(
    p: torch.Tensor,
    z: torch.Tensor,
    prefix_ids: torch.Tensor,
    next_actions: torch.Tensor,
    z_phantom: torch.Tensor | None,
    *,
    margin: float,
    lambda_push: float,
    normalize: bool,
    dedup_negative_actions: bool,
) -> dict[str, torch.Tensor]:
    """Compute the v6 pull/push distance-margin loss.

    Args:
        p: predicted modes ``(B, M, d_model)``.
        z: target embeddings ``(B, d_model)`` (from target encoder over the
            single next-action token).
        prefix_ids: integer prefix group ids ``(B,)``. Anchors share a
            positive iff their prefix ids match.
        next_actions: token ids of m_{t+1} ``(B,)``.
        z_phantom: target embedding ``(d_model,)`` produced by the target
            encoder on a token unused in the current batch, or ``None`` if
            the vocabulary is exhausted (rare for B<vocab). When ``None``
            colliding negative pairs are masked out of the push instead.
        margin: hinge margin for the push term.
        lambda_push: weight of the push term in the total loss.
        normalize: L2-normalize ``p`` and ``z`` (and ``z_phantom``) before
            computing distances.
        dedup_negative_actions: if False, no phantom replacement is applied
            and the original z is used in the push for every negative pair.

    Returns:
        Dict with the scalar ``loss`` and detached diagnostic tensors.
    """
    if p.ndim != 3:
        raise ValueError(f"Expected p with shape (B, M, d), got {p.shape}")
    if z.ndim != 2 or z.size(0) != p.size(0):
        raise ValueError(f"Expected z with shape (B, d), got {z.shape}")
    B, M, d = p.shape
    device = p.device

    if normalize:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        if z_phantom is not None:
            z_phantom = F.normalize(z_phantom, dim=-1)

    same_prefix = prefix_ids.unsqueeze(0) == prefix_ids.unsqueeze(1)        # (B, B) [i, j]
    same_action = next_actions.unsqueeze(0) == next_actions.unsqueeze(1)    # (B, B) [k, j]

    # collision[i, j] = exists k such that same_prefix[i, k] and same_action[k, j].
    # Equivalent boolean matmul.
    collision = (same_prefix.float() @ same_action.float()) > 0.0
    # Drop the trivial same-prefix-same-sample contribution from collisions: it
    # is irrelevant because collisions matter only for negative pairs.

    pull_mask = same_prefix                                                 # includes self
    neg_mask = ~same_prefix

    min_sq, _ = _pairwise_min_mode_sq_dist(p, z)                            # (B, B)
    min_dist = min_sq.clamp_min(1e-12).sqrt()                                # used by push

    # Pull: per-pair MSE in latent space.
    pull_count = pull_mask.float().sum().clamp_min(1.0)
    pull_loss = (min_sq * pull_mask.float()).sum() / pull_count

    # Push: hinge over closest-mode distance.
    if dedup_negative_actions and z_phantom is not None:
        # Compute closest-mode distance to the phantom target per anchor.
        # z_phantom is (d,) -> ((1, d)) for broadcasting.
        phantom_sq = ((p - z_phantom.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)  # (B, M)
        phantom_min_sq = phantom_sq.min(dim=1).values                              # (B,)
        phantom_min_dist = phantom_min_sq.clamp_min(1e-12).sqrt()                  # (B,)
        # For colliding pairs we use the phantom distance for anchor i, which is
        # a function of i alone. Broadcast to (B, B) along j.
        phantom_dist_grid = phantom_min_dist.unsqueeze(1).expand(B, B)             # (B, B)
        push_dist = torch.where(collision, phantom_dist_grid, min_dist)
    else:
        push_dist = min_dist
        # If dedup disabled or phantom unavailable, drop colliding negatives
        # from the push entirely to avoid contradictory gradients.
        if dedup_negative_actions and z_phantom is None:
            neg_mask = neg_mask & (~collision)

    push_terms = F.relu(margin - push_dist) ** 2                                   # (B, B)
    push_count = neg_mask.float().sum().clamp_min(1.0)
    push_loss = (push_terms * neg_mask.float()).sum() / push_count

    total = pull_loss + lambda_push * push_loss

    # Diagnostics.
    with torch.no_grad():
        pos_dist_sum = (min_dist * pull_mask.float()).sum()
        mean_pos_dist = pos_dist_sum / pull_count
        neg_dist_sum = (min_dist * neg_mask.float()).sum()
        mean_neg_dist = neg_dist_sum / push_count
        n_collisions = (collision & (~same_prefix)).float().sum()
        n_negatives_raw = (~same_prefix).float().sum().clamp_min(1.0)
        dedup_fraction = n_collisions / n_negatives_raw
        positives_per_anchor = pull_mask.float().sum(dim=1).mean()
        negatives_per_anchor = neg_mask.float().sum(dim=1).mean()
        p_std = p.std(dim=0, unbiased=False).mean()
        z_std = z.std(dim=0, unbiased=False).mean()
        z_norm_off = F.normalize(z, dim=-1)
        cos_offdiag = z_norm_off @ z_norm_off.T
        mask_off = ~torch.eye(B, dtype=torch.bool, device=device)
        cos_sim_offdiag = cos_offdiag[mask_off].mean() if B > 1 else torch.tensor(float("nan"), device=device)

    return {
        "loss": total,
        "pull_loss": pull_loss.detach(),
        "push_loss": push_loss.detach(),
        "mean_pos_dist": mean_pos_dist.detach(),
        "mean_neg_dist": mean_neg_dist.detach(),
        "dedup_fraction": dedup_fraction.detach(),
        "positives_per_anchor": positives_per_anchor.detach(),
        "negatives_per_anchor": negatives_per_anchor.detach(),
        "p_std": p_std.detach(),
        "z_std": z_std.detach(),
        "cos_sim_offdiag": cos_sim_offdiag.detach(),
    }


def _first_unused_token(used: set[int], vocab_actions: int) -> int | None:
    """Return the smallest token in ``[0, vocab_actions)`` absent from ``used``."""
    for tok in range(vocab_actions):
        if tok not in used:
            return tok
    return None


def select_phantom_token(next_actions: torch.Tensor, vocab_actions: int) -> int | None:
    """Return a token id present in ``[0, vocab_actions)`` but unused in the batch."""
    used = set(next_actions.detach().cpu().tolist())
    return _first_unused_token(used, vocab_actions)


# ============================================================================
# v6 module
# ============================================================================

class OthelloJEPAHardDisjointAction(nn.Module):
    """Context + EMA target Transformer with multi-mode predictor and v6 loss.

    The target encoder shares the OthelloGPT architecture with the context
    encoder so EMA copying is parameter-aligned. At inference time the target
    encoder is fed a single-token sequence ``(B, 1)`` containing m_{t+1}; its
    output at position 0 is the target embedding z.
    """

    def __init__(self, cfg: JEPAHardDisjointActionConfig) -> None:
        super().__init__()
        if not cfg.use_ema_target:
            raise ValueError("v6 hard-disjoint requires use_ema_target=True.")
        if not (0.0 <= cfg.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if cfg.num_modes < 1:
            raise ValueError("num_modes must be >= 1.")

        self.cfg = cfg
        # Compatibility alias so downstream tools (linear_head, probes, eval
        # notebooks) that access ``jepa_model.jepa_config`` on the predictive
        # JEPAConfig also work on the v6 module. Both names point to the same
        # dataclass instance, so updating one updates the other.
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

        self.predictor = MultiModeMLPPredictor(
            d_model=cfg.d_model,
            num_modes=cfg.num_modes,
            hidden_mult=cfg.predictor_hidden_mult,
            n_layers=cfg.predictor_n_layers,
            activation=cfg.predictor_activation,
            dropout=cfg.predictor_dropout,
        )

        # vocab_actions = number of real move tokens (pad token sits one past).
        self._vocab_actions = cfg.board_size * cfg.board_size - 4

    @property
    def config(self) -> GPTConfig:
        """Expose the underlying context GPT config (loader compatibility)."""
        return self.context_encoder.config

    def train(self, mode: bool = True) -> "OthelloJEPAHardDisjointAction":
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

    # Compatibility alias used by the shared trainer.
    @torch.no_grad()
    def update_targets(self, momentum: float | None = None) -> None:
        self.update_target_encoder(momentum)

    def encode_hidden(self, encoder: nn.Module, idx: torch.Tensor) -> torch.Tensor:
        """Forward through a JEPA encoder path without the lm_head.

        This matches the interface exposed by the predictive and contrastive
        JEPA objectives so downstream evaluation can treat every objective
        uniformly.
        """
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
        """Return the final-position context encoder hidden state."""
        return self.encode_hidden(self.context_encoder, idx)[:, -1, :]

    def _encode_context(self, x_context: torch.Tensor) -> torch.Tensor:
        """Run the context encoder and return the hidden state at the last token."""
        return self.encode_last_hidden(x_context)

    @torch.no_grad()
    def _encode_target_action(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Run the EMA target encoder on a length-1 sequence per sample.

        Args:
            action_tokens: ``(B,)`` long tensor of next-action token ids.

        Returns:
            Target embeddings ``(B, d_model)``.
        """
        if action_tokens.ndim != 1:
            raise ValueError(f"Expected (B,) action_tokens, got {action_tokens.shape}")
        encoder = self.target_encoder
        x_in = action_tokens.unsqueeze(1)  # (B, 1)
        pos = torch.arange(0, 1, device=action_tokens.device)
        x = encoder.wte(x_in) + encoder.wpe(pos)
        # Target encoder is in eval(); dropout becomes identity but we apply for
        # parity with context.
        x = encoder.drop(x)
        for block in encoder.blocks:
            x = block(x)
        x = encoder.ln_f(x)
        return x[:, 0, :]

    def forward(
        self,
        x_context: torch.Tensor,
        next_actions: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the v6 loss for a prefix-grouped batch.

        Args:
            x_context: ``(B, t)`` token ids of the prefix m_1..m_t (per sample).
            next_actions: ``(B,)`` token ids of m_{t+1}.
            prefix_ids: ``(B,)`` integer prefix group ids; equal entries indicate
                the same prefix.

        Returns:
            Dict with scalar ``loss`` plus detached diagnostics.
        """
        if next_actions.size(0) != x_context.size(0):
            raise ValueError("next_actions and x_context batch size mismatch")
        if prefix_ids.size(0) != x_context.size(0):
            raise ValueError("prefix_ids and x_context batch size mismatch")

        c_summary = self._encode_context(x_context)              # (B, d_model)
        p_modes = self.predictor(c_summary)                       # (B, M, d_model)
        z_target = self._encode_target_action(next_actions)       # (B, d_model)

        phantom_token = select_phantom_token(next_actions, self._vocab_actions)
        if phantom_token is None:
            z_phantom = None
        else:
            phantom_tensor = torch.full(
                (1,), phantom_token, dtype=next_actions.dtype, device=next_actions.device,
            )
            z_phantom = self._encode_target_action(phantom_tensor)[0]  # (d_model,)

        out = hard_disjoint_action_loss(
            p_modes,
            z_target,
            prefix_ids,
            next_actions,
            z_phantom,
            margin=self.cfg.margin,
            lambda_push=self.cfg.lambda_push,
            normalize=self.cfg.normalize,
            dedup_negative_actions=self.cfg.dedup_negative_actions,
        )
        with torch.no_grad():
            # Raw context-summary dispersion. This is intentionally computed
            # outside hard_disjoint_action_loss because c_summary is never
            # normalized by the v6 loss. It lets the shared collapse logger
            # distinguish context collapse from low normalized target std.
            out["c_std"] = c_summary.std(dim=0, unbiased=False).mean().detach()
        out["ema_momentum"] = torch.tensor(self.cfg.ema_momentum, device=x_context.device)
        return out

    def forward_all_positions(
        self,
        x_context: torch.Tensor,
        next_actions: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """All-position v6 loss: distance-margin at every prefix boundary 1..t.

        The prefix-grouped batch is built at a single depth ``t`` (so every row
        is length ``t`` with no padding, and ``next_actions`` is the token at
        absolute position ``t``). Instead of supervising only that final
        boundary, this runs the objective at every boundary ``t' in [1, t]``:

            - The context encoder runs ONCE over ``x_context``; because both the
              Transformer and Mamba encoders are causal, the hidden state at
              position ``t'-1`` depends only on tokens ``<= t'-1`` and is exactly
              the context summary for boundary ``t'``.
            - The predictor and the EMA target encoder are each applied ONCE in a
              flattened batch across all ``B * t`` (summary, target) pairs. The
              target encoder sees each target token as an independent length-one
              sequence, so flattening over positions changes nothing.
            - Prefix group ids are RECOMPUTED at each depth ``t'`` from
              ``x_context[:, :t']``: same-prefix rows are positives. Groups built
              at depth ``t`` stay grouped at every ``t' <= t``, while distinct
              deep groups that share a shallow prefix merge at small ``t'`` and
              naturally supply multi-modal positives (same prefix, different next
              action) for the multi-mode predictor.

        The per-boundary losses are averaged with equal weight (each boundary
        counts the same); ``hard_disjoint_action_loss`` already normalizes
        per-pair within each boundary. The returned dict mirrors ``forward`` so
        the shared trainer's metric plumbing is unchanged.
        """
        if x_context.ndim != 2:
            raise ValueError(f"Expected x_context (B, t), got {tuple(x_context.shape)}")
        if next_actions.size(0) != x_context.size(0):
            raise ValueError("next_actions and x_context batch size mismatch")
        if prefix_ids.size(0) != x_context.size(0):
            raise ValueError("prefix_ids and x_context batch size mismatch")

        B, t = x_context.shape
        device = x_context.device

        # full_x[:, t'] is the target token at boundary t' for t' in [1, t].
        full_x = torch.cat([x_context, next_actions.unsqueeze(1)], dim=1)  # (B, t+1)

        # One causal context pass; hidden index t'-1 is the summary for boundary t'.
        context_hidden = self._encode_context_hidden(x_context)          # (B, t, d)
        d = context_hidden.size(-1)

        # One predictor pass over all B*t context summaries (per-vector MLP).
        p_all = self.predictor(context_hidden.reshape(B * t, d)).view(
            B, t, self.cfg.num_modes, d
        )                                                                # (B, t, M, d)

        # One target pass over all B*t target tokens (each encoded length-one).
        target_tokens = full_x[:, 1:].reshape(-1)                        # (B*t,)
        z_all = self._encode_target_action(target_tokens).view(B, t, d)  # (B, t, d)

        # Per-boundary phantom token (or None if that boundary's targets exhaust
        # the vocab, which is the common case at large B). One CPU sync total.
        targets_by_pos = full_x[:, 1:].t().detach().cpu().tolist()       # (t, B)
        phantom_tokens: list[int | None] = [
            _first_unused_token(set(row), self._vocab_actions) for row in targets_by_pos
        ]
        if any(tok is not None for tok in phantom_tokens):
            phantom_input = torch.tensor(
                [tok if tok is not None else 0 for tok in phantom_tokens],
                dtype=next_actions.dtype,
                device=device,
            )                                                            # (t,)
            z_phantom_all = self._encode_target_action(phantom_input)    # (t, d)
        else:
            z_phantom_all = None

        per_pos: list[dict[str, torch.Tensor]] = []
        for tp in range(1, t + 1):
            pos = tp - 1
            # Regroup by exact prefix at this depth; equal ids => positives.
            pid = torch.unique(x_context[:, :tp], dim=0, return_inverse=True)[1]
            z_phantom = (
                z_phantom_all[pos]
                if (z_phantom_all is not None and phantom_tokens[pos] is not None)
                else None
            )
            per_pos.append(
                hard_disjoint_action_loss(
                    p_all[:, pos],
                    z_all[:, pos],
                    pid,
                    full_x[:, tp],
                    z_phantom,
                    margin=self.cfg.margin,
                    lambda_push=self.cfg.lambda_push,
                    normalize=self.cfg.normalize,
                    dedup_negative_actions=self.cfg.dedup_negative_actions,
                )
            )

        # Equal-weight mean across boundaries. The "loss" key keeps its graph;
        # every other key is already detached by hard_disjoint_action_loss.
        out: dict[str, torch.Tensor] = {}
        for key in per_pos[0]:
            out[key] = torch.stack([entry[key] for entry in per_pos]).mean(dim=0)

        with torch.no_grad():
            # Raw context dispersion per boundary, then averaged. Kept outside the
            # loss (which never normalizes c) so the collapse logger can tell
            # context collapse from low normalized-target std.
            c_std_per_pos = context_hidden.float().std(dim=0, unbiased=False).mean(dim=-1)
            out["c_std"] = c_std_per_pos.mean().detach()
        out["ema_momentum"] = torch.tensor(self.cfg.ema_momentum, device=device)
        return out

    def _encode_context_hidden(self, x_context: torch.Tensor) -> torch.Tensor:
        """Full-sequence context hidden states ``(B, t, d)`` (grad-enabled)."""
        return self.encode_hidden(self.context_encoder, x_context)


# ============================================================================
# Smoke test
# ============================================================================

def _smoke() -> None:
    torch.manual_seed(42)
    cfg = JEPAHardDisjointActionConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        num_modes=4,
    )
    model = OthelloJEPAHardDisjointAction(cfg)
    B = 8
    t = 4
    vocab = model.config.vocab_size
    x_context = torch.randint(0, vocab, (B, t))
    # Build prefix groups: 2 groups, 4 samples each.
    prefix_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    # Force same prefix tokens within a group so the semantic invariant holds
    # for the smoke test even though the loss only uses prefix_ids.
    x_context[:4] = x_context[0]
    x_context[4:] = x_context[4]
    next_actions = torch.randint(0, model._vocab_actions, (B,))
    # Inject a collision: a different-prefix sample shares the same action as a
    # group-0 positive.
    next_actions[4] = next_actions[0]

    out = model(x_context, next_actions, prefix_ids)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"]).all()
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters()), \
        "context_encoder did not receive gradients"
    assert any(p.grad is not None for p in model.predictor.parameters()), \
        "predictor did not receive gradients"
    assert all(p.grad is None for p in model.target_encoder.parameters()), \
        "target_encoder must not receive gradients"
    print(
        "v6 smoke OK",
        "loss=", float(out["loss"].detach()),
        "pull=", float(out["pull_loss"]),
        "push=", float(out["push_loss"]),
        "dedup_frac=", float(out["dedup_fraction"]),
        "pos/anchor=", float(out["positives_per_anchor"]),
        "neg/anchor=", float(out["negatives_per_anchor"]),
    )

    # All-position path: same batch, loss at every boundary 1..t.
    model.zero_grad(set_to_none=True)
    out_all = model.forward_all_positions(x_context, next_actions, prefix_ids)
    assert out_all["loss"].dim() == 0
    assert torch.isfinite(out_all["loss"]).all()
    out_all["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters()), \
        "all-position context_encoder did not receive gradients"
    assert any(p.grad is not None for p in model.predictor.parameters()), \
        "all-position predictor did not receive gradients"
    assert all(p.grad is None for p in model.target_encoder.parameters()), \
        "all-position target_encoder must not receive gradients"
    print(
        "v6 all-position smoke OK",
        "loss=", float(out_all["loss"].detach()),
        "pull=", float(out_all["pull_loss"]),
        "push=", float(out_all["push_loss"]),
        "pos/anchor=", float(out_all["positives_per_anchor"]),
    )

    model.update_target_encoder()


if __name__ == "__main__":
    _smoke()
