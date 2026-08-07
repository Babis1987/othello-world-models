"""Tests for K-step (horizon>1) all-position supervision (v3/v4 hard-disjoint).

``train_jepa._forward_all_position_kstep`` extends all-position supervision from
a single next token to a K-token hard-disjoint future window at every boundary.
``forward_all_position_v1`` dispatches to it when prediction_horizon>1. These
tests cover both predictor types (plain ``mlp`` broadcast, ``mlp_multi_pos``),
gradient flow, the metric contract, a K=1 equivalence to the horizon-1 path,
config validation, the validation gate, and the real training dispatch.
"""

from __future__ import annotations

import math
import pickle
import random
import tempfile
from pathlib import Path

import pytest
import torch

from othello_research.datasets.move_mapping import build_mappings
from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA
from othello_research.training import train_jepa

ROOT = Path(__file__).resolve().parents[1]
PAD = 8 * 8 - 4


def _model(predictor_type: str, horizon: int) -> OthelloJEPA:
    kw = dict(
        variant="v2",
        loss_type="smooth_l1",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=32,
        dropout=0.0,
        prediction_horizon=horizon,
        use_ema_target=True,
        predictor_type=predictor_type,
    )
    if predictor_type == "mlp":
        kw.update(predictor_hidden_dim=64, predictor_n_layers=2, predictor_dropout=0.0)
    else:
        kw.update(predictor_hidden_mult=2, predictor_dropout=0.0)
    return OthelloJEPA(JEPAConfig(**kw))


_KSTEP_METRIC_KEYS = (
    "smooth_l1_loss", "z_pred_std", "z_tgt_std", "z_std",
    "c_std", "p_std", "cos_sim_offdiag", "positions_per_game",
)


@pytest.mark.parametrize("predictor_type,horizon", [("mlp", 8), ("mlp_multi_pos", 4)])
def test_kstep_forward_backward_target_frozen(predictor_type, horizon) -> None:
    torch.manual_seed(0)
    model = _model(predictor_type, horizon)
    x = torch.randint(0, PAD, (6, 16))
    x[4, 11:] = PAD  # short game with padding

    out, pairs = train_jepa.forward_all_position_v1(
        model, x, pad_token=PAD, target_chunk_size=4096, stats_chunk_size=8
    )
    assert pairs > 0
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())

    for key in _KSTEP_METRIC_KEYS:
        assert key in out, key
        assert torch.isfinite(torch.as_tensor(out[key])), key
    # positions_per_game must not exceed available boundaries (seq_len-1).
    assert 0 < float(out["positions_per_game"]) <= 15


def test_kstep_dispatch_only_triggers_for_horizon_gt_1() -> None:
    # A horizon-1 hard-disjoint model must stay on the original single-target path.
    torch.manual_seed(0)
    model = _model("mlp_multi_pos", 1)
    x = torch.randint(0, PAD, (6, 10))
    out, pairs = train_jepa.forward_all_position_v1(
        model, x, pad_token=PAD, target_chunk_size=65536, stats_chunk_size=64
    )
    assert pairs > 0 and torch.isfinite(out["loss"])


def test_kstep_matches_horizon1_all_position_path() -> None:
    """At K=1 the K-step forward reproduces the horizon-1 all-position loss."""
    torch.manual_seed(1)
    model = _model("mlp_multi_pos", 1)
    model.eval()
    x = torch.randint(0, PAD, (8, 12))  # no padding: every boundary is eligible

    with torch.no_grad():
        old, old_pairs = train_jepa.forward_all_position_v1(
            model, x, pad_token=PAD, target_chunk_size=65536, stats_chunk_size=64
        )
        new, new_pairs = train_jepa._forward_all_position_kstep(
            model, x, pad_token=PAD, horizon=1,
            target_chunk_size=65536, stats_chunk_size=64,
        )
    assert old_pairs == new_pairs
    assert torch.allclose(old["loss"], new["loss"], atol=1e-5, rtol=1e-4), (
        float(old["loss"]), float(new["loss"])
    )


def _resolve(fn: str) -> train_jepa.TrainConfig:
    args, _, _ = train_jepa.parse_configured_args(["--config", str(ROOT / "configs" / fn)])
    cfg, _ = train_jepa.namespace_to_train_config(args)
    train_jepa.validate_train_config(cfg)
    return cfg


def test_kstep_configs_resolve_and_validate() -> None:
    base = _resolve("jepa_v3_mlp_k8_b8_hard_disjoint.yml")
    assert base.view_mode == "hard_disjoint_future"
    assert base.prediction_horizon == 8
    assert base.position_sampling == "random"

    v3 = _resolve("jepa_v3_hd_k8_all_positions.yml")
    assert v3.position_sampling == "all"
    assert v3.prediction_horizon == 8
    assert v3.predictor_type == "mlp"
    assert train_jepa.jepa_view_mode(v3) == "hard_disjoint_future"
    assert train_jepa.jepa_loss_type(v3) == "smooth_l1"

    v4 = _resolve("jepa_v4_multi_pos_k4_hd_all_positions.yml")
    assert v4.position_sampling == "all"
    assert v4.prediction_horizon == 4
    assert v4.predictor_type == "mlp_multi_pos"
    assert train_jepa.jepa_view_mode(v4) == "hard_disjoint_future"


def test_gate_rejects_kstep_allpos_for_unsupported_combos() -> None:
    for bad in (
        dict(loss_type="vicreg", view_mode="hard_disjoint_future"),
        dict(loss_type="smooth_l1", view_mode="nested"),
    ):
        cfg = train_jepa.TrainConfig(
            out_dir="out", objective_class="jepa", variant="v4",
            position_sampling="all", prediction_horizon=4,
            use_ema_target=True, **bad,
        )
        with pytest.raises(ValueError, match="prediction_horizon"):
            train_jepa.validate_train_config(cfg)


def _write_synthetic_chunk(path: Path, n_games: int = 240, board_size: int = 8) -> None:
    rng = random.Random(0)
    raw_to_token, _ = build_mappings(board_size)
    valid_raw = [raw for raw, token in enumerate(raw_to_token) if token != -1]
    seeds = [tuple(rng.sample(valid_raw, k=6)) for _ in range(20)]
    games = [list(rng.choice(seeds)) + rng.sample(valid_raw, k=20) for _ in range(n_games)]
    with open(path, "wb") as handle:
        pickle.dump(games, handle)


def test_train_one_chunk_kstep_allpos_dispatch() -> None:
    """Drive the real training dispatch for a K=4 hard-disjoint all-position run."""
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        chunk_path = Path(tmp) / "synthetic.pickle"
        _write_synthetic_chunk(chunk_path)

        cfg = train_jepa.TrainConfig(
            out_dir=str(Path(tmp) / "out"),
            objective_class="jepa",
            variant="v4",
            view_mode="hard_disjoint_future",
            loss_type="smooth_l1",
            prediction_horizon=4,
            position_sampling="all",
            predictor_type="mlp_multi_pos",
            predictor_hidden_mult=2,
            board_size=8,
            n_layers=2,
            n_heads=4,
            d_model=64,
            dropout=0.0,
            use_ema_target=True,
            batch_size=32,
            learning_rate=3e-4,
            warmup_steps=0,
            grad_clip=1.0,
            precision="fp32",
            seed=42,
        )
        train_jepa.validate_train_config(cfg)
        model, _ = train_jepa.build_objective_model(cfg)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        before = next(model.target_encoder.parameters()).detach().clone()
        result = train_jepa.train_one_chunk(
            model, optimizer, scaler, chunk_path, cfg, model.config, device,
            step=0, total_steps=10,
        )
        avg_loss, metrics, seen_batches = result[0], result[1], result[5]
        assert seen_batches > 0
        assert math.isfinite(avg_loss)
        assert math.isfinite(metrics["smooth_l1_loss"])
        assert metrics["positions_per_game"] > 0
        after = next(model.target_encoder.parameters()).detach()
        assert not torch.allclose(before, after)
