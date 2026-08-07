from __future__ import annotations

import math
from pathlib import Path

import torch

from othello_research.objectives.jepa_contrastive import (
    JEPAContrastiveConfig,
    OthelloJEPAContrastive,
)
from othello_research.training import train_jepa as tjm


def _tiny_model(**overrides) -> OthelloJEPAContrastive:
    cfg = JEPAContrastiveConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_dropout=0.0,
        **overrides,
    )
    return OthelloJEPAContrastive(cfg)


def test_prefix_mask_correctness():
    model = _tiny_model()
    x_context = torch.tensor([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 4],
        [5, 6, 7],
    ])
    mask = model._compute_prefix_mask(x_context)
    expected = torch.tensor([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ])
    assert mask.shape == (4, 4)
    assert torch.equal(mask, expected)


def test_multi_positive_infonce_matches_single_positive_case():
    model = _tiny_model(contrastive_use_prefix_mask=False)
    sim = torch.tensor([
        [3.0, 1.0, 0.0],
        [0.0, 4.0, 1.0],
        [1.0, 0.0, 5.0],
    ])
    mask = torch.eye(3, dtype=torch.bool)
    loss = model._multi_positive_infonce(sim, mask)
    expected = torch.nn.functional.cross_entropy(sim, torch.arange(3))
    assert torch.allclose(loss, expected)


def test_temperature_effect_high_temperature_approaches_log_batch():
    model = _tiny_model()
    base = torch.eye(4) * 5.0
    mask = torch.eye(4, dtype=torch.bool)
    low_temp_loss = model._multi_positive_infonce(base / 0.01, mask)
    high_temp_loss = model._multi_positive_infonce(base / 10.0, mask)
    assert low_temp_loss < high_temp_loss
    assert abs(float(high_temp_loss) - math.log(4)) < 0.5


def test_loss_can_decrease_on_fixed_batch(synthetic_game_batch):
    torch.manual_seed(123)
    model = _tiny_model()
    batch = synthetic_game_batch(B=8, T=20, board_size=8, seed=123)
    t = 8
    target_positions = torch.full((8, 1), t, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        out = model(batch[:, :t], batch[:, : t + 1], target_positions)
        out["loss"].backward()
        optimizer.step()
        model.update_targets()
        losses.append(float(out["loss"].detach()))

    assert losses[-1] <= losses[0] + 1e-4


def test_forward_backward_smoke_exposes_explicit_metrics():
    torch.manual_seed(7)
    model = _tiny_model()
    x = torch.randint(0, model.config.vocab_size, (4, 12))
    target_positions = torch.full((4, 1), 10, dtype=torch.long)

    out = model(x[:, :10], x[:, :11], target_positions)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    for key in (
        "positive_accuracy",
        "diag_accuracy",
        "n_positives_mean",
        "pred_raw_norm_mean",
        "target_raw_norm_mean",
        "logit_std",
        "logit_max",
        "logit_min",
    ):
        assert key in out
        assert key in out["diagnostics"]
        assert torch.isfinite(out[key])

    out["loss"].backward()
    assert any(param.grad is not None for param in model.context_encoder.parameters())
    assert any(param.grad is not None for param in model.predictor.parameters())
    assert all(param.grad is None for param in model.target_encoder.parameters())


def test_diagonal_only_positive_accuracy_matches_diag_accuracy():
    torch.manual_seed(8)
    model = _tiny_model(contrastive_use_prefix_mask=False)
    x = torch.randint(0, model.config.vocab_size, (4, 12))
    target_positions = torch.full((4, 1), 10, dtype=torch.long)

    out = model(x[:, :10], x[:, :11], target_positions)
    assert torch.allclose(out["n_positives_mean"], torch.tensor(1.0))
    assert torch.allclose(out["positive_accuracy"], out["diag_accuracy"])
    assert torch.allclose(out["contrastive_accuracy"], out["positive_accuracy"])


def test_prefix_mask_positive_accuracy_is_at_least_diag_accuracy():
    torch.manual_seed(9)
    model = _tiny_model(contrastive_use_prefix_mask=True)
    x_context = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [7, 8, 9, 10, 11],
    ], dtype=torch.long)
    target = torch.tensor([[12], [13], [14], [15]], dtype=torch.long)
    x_target = torch.cat([x_context, target], dim=1)
    target_positions = torch.full((4, 1), x_context.size(1), dtype=torch.long)

    out = model(x_context, x_target, target_positions)
    assert out["n_positives_mean"] > 1.0
    assert out["positive_accuracy"] >= out["diag_accuracy"]


def test_normalized_logits_are_bounded_by_temperature():
    torch.manual_seed(10)
    temperature = 0.2
    model = _tiny_model(contrastive_temperature=temperature)
    x = torch.randint(0, model.config.vocab_size, (6, 12))
    target_positions = torch.full((6, 1), 10, dtype=torch.long)

    out = model(x[:, :10], x[:, :11], target_positions)
    eps = 1e-4
    assert out["logit_max"] <= (1.0 / temperature) + eps
    assert out["logit_min"] >= (-1.0 / temperature) - eps


def test_contrastive_yaml_dispatches_and_predictive_configs_still_parse():
    root = Path(__file__).parents[1]
    parser = tjm.build_parser()

    data = tjm.load_yaml_config(root / "configs" / "jepa_v5_contrastive.yml")
    args = parser.parse_args([
        *tjm.config_to_cli_args(data, parser),
        "--out_dir", "__unused_config_test__",
        "--n_layers", "2",
        "--n_heads", "4",
        "--d_model", "64",
        "--dropout", "0.0",
        "--predictor_hidden_mult", "2",
        "--batch_size", "4",
        "--precision", "fp32",
        "--drive_sync_dir", "",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    assert cfg.objective_class == "jepa_contrastive"
    assert cfg.variant == "v5"
    assert cfg.loss_type == "infonce"
    model, objective_cfg = tjm.build_objective_model(cfg)
    assert isinstance(model, OthelloJEPAContrastive)
    assert objective_cfg.variant == "jepa_v5_contrastive"

    for config_name in (
        "jepa_v1_vicreg_b8.yml",
        "jepa_v2_ema_b8.yml",
        "jepa_v3_mlp_k8_b8.yml",
        "jepa_v4_multi_pos_mlp_k4_b8.yml",
    ):
        data = tjm.load_yaml_config(root / "configs" / config_name)
        args = parser.parse_args([
            *tjm.config_to_cli_args(data, parser),
            "--out_dir", "__unused_config_test__",
            "--n_layers", "2",
            "--n_heads", "4",
            "--d_model", "64",
            "--dropout", "0.0",
            "--drive_sync_dir", "",
        ])
        cfg, _ = tjm.namespace_to_train_config(args)
        assert cfg.objective_class == "jepa"
