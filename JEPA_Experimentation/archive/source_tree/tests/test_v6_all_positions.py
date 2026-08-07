"""Tests for the all-position (every-boundary) v6 hard-disjoint action JEPA.

``OthelloJEPAHardDisjointAction.forward_all_positions`` keeps the prefix-grouped
batch semantics and the distance-margin / multi-mode / phantom-dedup loss of v6,
but evaluates the loss at every prefix boundary ``t' in [1, t]`` from a single
amortized context pass. These tests exercise:

    - gradient flow (context + predictor trained, target frozen),
    - per-depth prefix regrouping (equal to the constructed groups at depth t,
      merged at shallower depths),
    - equivalence of the batched predictor/target passes and equal-weight
      aggregation against a per-position reference,
    - phantom-token correctness,
    - coverage of the declared v6 metric keys.
"""

from __future__ import annotations

import torch

from othello_research.objectives.jepa_hard_disjoint_action import (
    JEPAHardDisjointActionConfig,
    OthelloJEPAHardDisjointAction,
    hard_disjoint_action_loss,
    select_phantom_token,
)
from othello_research.training import train_jepa


def _tiny_model() -> OthelloJEPAHardDisjointAction:
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
    model.eval()  # deterministic (dropout is 0 anyway); target is always eval
    return model


def _controlled_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two constructed groups of 4 that share the first 4 tokens.

    Groups diverge only at the final prefix token (index 3), so at depth t=4
    they are distinct while at every shallower depth they merge into one group.
    An action (10) is shared across groups to exercise the collision path.
    """
    group0 = [1, 2, 3, 5]  # shared prefix [1,2,3], diverges at index 3
    group1 = [1, 2, 3, 9]
    x_context = torch.tensor(
        [group0, group0, group0, group0, group1, group1, group1, group1],
        dtype=torch.long,
    )
    next_actions = torch.tensor([10, 11, 12, 10, 13, 14, 10, 15], dtype=torch.long)
    prefix_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return x_context, next_actions, prefix_ids


def _same_partition(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True iff a and b induce the same equivalence classes (labels may differ)."""
    aa = a.unsqueeze(0) == a.unsqueeze(1)
    bb = b.unsqueeze(0) == b.unsqueeze(1)
    return bool((aa == bb).all())


def test_forward_all_positions_backward_target_frozen() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    x_context, next_actions, prefix_ids = _controlled_batch()

    out = model.forward_all_positions(x_context, next_actions, prefix_ids)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert "c_std" in out and torch.isfinite(out["c_std"])

    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_prefix_regroup_matches_constructed_at_depth_t_and_merges_shallow() -> None:
    x_context, _, prefix_ids = _controlled_batch()
    t = x_context.size(1)

    # At the full constructed depth the recomputed partition equals the sampler's.
    pid_full = torch.unique(x_context[:, :t], dim=0, return_inverse=True)[1]
    assert _same_partition(pid_full, prefix_ids)
    assert int(pid_full.unique().numel()) == 2

    # At shallower depths the two groups share the prefix and merge into one.
    for tp in range(1, t):
        pid = torch.unique(x_context[:, :tp], dim=0, return_inverse=True)[1]
        assert int(pid.unique().numel()) == 1, tp


def test_batched_passes_and_aggregation_match_per_position_reference() -> None:
    torch.manual_seed(1)
    model = _tiny_model()
    x_context, next_actions, prefix_ids = _controlled_batch()
    t = x_context.size(1)
    full_x = torch.cat([x_context, next_actions.unsqueeze(1)], dim=1)

    with torch.no_grad():
        method = model.forward_all_positions(x_context, next_actions, prefix_ids)

        # Per-position reference using single (not batched) predictor/target calls.
        context_hidden = model.encode_hidden(model.context_encoder, x_context)
        losses = []
        for tp in range(1, t + 1):
            c = context_hidden[:, tp - 1, :]
            p = model.predictor(c)
            z = model._encode_target_action(full_x[:, tp])
            pid = torch.unique(x_context[:, :tp], dim=0, return_inverse=True)[1]
            phantom_tok = select_phantom_token(full_x[:, tp], model._vocab_actions)
            z_phantom = (
                model._encode_target_action(
                    torch.tensor([phantom_tok], dtype=next_actions.dtype)
                )[0]
                if phantom_tok is not None
                else None
            )
            losses.append(
                hard_disjoint_action_loss(
                    p, z, pid, full_x[:, tp], z_phantom,
                    margin=model.cfg.margin,
                    lambda_push=model.cfg.lambda_push,
                    normalize=model.cfg.normalize,
                    dedup_negative_actions=model.cfg.dedup_negative_actions,
                )["loss"]
            )
        ref_loss = torch.stack(losses).mean()

    assert torch.allclose(method["loss"], ref_loss, atol=1e-5, rtol=1e-4)


def test_phantom_token_absent_from_boundary_targets() -> None:
    _, _, _ = _controlled_batch()
    model = _tiny_model()
    x_context, next_actions, prefix_ids = _controlled_batch()
    t = x_context.size(1)
    full_x = torch.cat([x_context, next_actions.unsqueeze(1)], dim=1)
    for tp in range(1, t + 1):
        targets = set(full_x[:, tp].tolist())
        tok = select_phantom_token(full_x[:, tp], model._vocab_actions)
        # With this tiny batch the vocab is not exhausted, so a phantom exists
        # and must not coincide with any real target at that boundary.
        assert tok is not None and tok not in targets, tp


def test_all_positions_output_covers_declared_metric_keys() -> None:
    torch.manual_seed(2)
    model = _tiny_model()
    x_context, next_actions, prefix_ids = _controlled_batch()
    out = model.forward_all_positions(x_context, next_actions, prefix_ids)

    cfg = train_jepa.TrainConfig(
        out_dir="out",
        objective_class="jepa_hard_disjoint_action",
        view_mode="hard_disjoint_action",
        position_sampling="all",
    )
    for key in train_jepa.objective_metric_keys(cfg):
        assert key in out, key
        assert torch.isfinite(torch.as_tensor(out[key])), key
