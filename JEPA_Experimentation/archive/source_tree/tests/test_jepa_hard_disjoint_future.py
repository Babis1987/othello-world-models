from pathlib import Path

import torch

from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA
from othello_research.objectives.jepa_contrastive import (
    JEPAContrastiveConfig,
    OthelloJEPAContrastive,
)
from othello_research.training import train_jepa as tjm


def _has_grad(module: torch.nn.Module) -> bool:
    return any(p.grad is not None for p in module.parameters())


def test_trainer_accepts_all_version_variant_aliases():
    parser = tjm.build_parser()
    cases = [
        ("jepa", "jepa_v1", "v1", "jepa_v1"),
        ("jepa", "jepa_v2", "v2", "jepa_v2"),
        ("jepa", "jepa_v3", "v3", "jepa_v2"),
        ("jepa", "jepa_v4", "v4", "jepa_v2"),
        ("jepa_contrastive", "jepa_v5", "v5", "jepa_v5_contrastive"),
        ("jepa_hard_disjoint_action", "jepa_v6", "v6", "jepa_v6_hard_disjoint_action"),
    ]
    for objective, variant, expected_public, expected_internal in cases:
        args = parser.parse_args([
            "--out_dir",
            "unused",
            "--objective_class",
            objective,
            "--variant",
            variant,
        ])
        cfg, _ = tjm.namespace_to_train_config(args)
        assert cfg.variant == expected_public
        assert tjm.build_objective_config(cfg).variant == expected_internal


def test_all_jepa_yamls_use_public_version_labels():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "jepa_v1_vicreg_b8.yml": "v1",
        "jepa_v1_vicreg_b8_hard_disjoint.yml": "v1",
        "jepa_v2_ema_b8.yml": "v2",
        "jepa_v2_ema_b8_hard_disjoint.yml": "v2",
        "jepa_no_predictor_k1_vicreg_b8_hard_disjoint.yml": "v1",
        "jepa_linear_k1_ema_b8_hard_disjoint.yml": "v2",
        "jepa_v3_mlp_k8_b8.yml": "v3",
        "jepa_v4_multi_pos_mlp_k4_b8.yml": "v4",
        "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint.yml": "v4",
        "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint_vicreg.yml": "v4",
        "jepa_v5_contrastive.yml": "v5",
        "jepa_v5_contrastive_hard_disjoint.yml": "v5",
        "jepa_v6_hard_disjoint_action.yml": "v6",
    }
    parser = tjm.build_parser()
    for config_name, expected_variant in expected.items():
        data = tjm.load_yaml_config(root / "configs" / config_name)
        assert data["experiment"]["variant"] == expected_variant
        args = parser.parse_args([*tjm.config_to_cli_args(data, parser)])
        cfg, _ = tjm.namespace_to_train_config(args)
        assert cfg.variant == expected_variant


def test_predictive_jepa_v1_hard_disjoint_future_forward_backward():
    torch.manual_seed(0)
    cfg = JEPAConfig(
        variant="jepa_v1",
        loss_type="vicreg",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_type="linear",
        prediction_horizon=1,
        vicreg_lambda=10.0,
        vicreg_mu=50.0,
        variance_threshold=2.0,
    )
    model = OthelloJEPA(cfg)
    x_context = torch.randint(0, model.config.vocab_size, (4, 6))
    x_target = torch.randint(0, model.config.vocab_size, (4, 1))

    out = model(x_context, x_target)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["predictions"].shape == (4, 1, 64)
    assert out["z_targets"].shape == (4, 1, 64)
    out["loss"].backward()
    assert _has_grad(model.context_encoder)
    assert _has_grad(model.predictor)


def test_predictive_jepa_v2_hard_disjoint_future_forward_backward():
    torch.manual_seed(0)
    cfg = JEPAConfig(
        variant="jepa_v2",
        loss_type="smooth_l1",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_type="transformer",
        predictor_n_layers=2,
        predictor_n_heads=4,
        prediction_horizon=2,
        use_ema_target=True,
    )
    model = OthelloJEPA(cfg)
    x_context = torch.randint(0, model.config.vocab_size, (4, 6))
    x_target = torch.randint(0, model.config.vocab_size, (4, 2))

    out = model(x_context, x_target)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["predictions"].shape == (4, 2, 64)
    assert out["z_targets"].shape == (4, 2, 64)
    out["loss"].backward()
    assert _has_grad(model.context_encoder)
    assert _has_grad(model.predictor)
    assert not _has_grad(model.target_encoder)


def test_predictive_jepa_no_predictor_vicreg_hard_disjoint_future_forward_backward():
    torch.manual_seed(0)
    cfg = JEPAConfig(
        variant="jepa_v1",
        loss_type="vicreg",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_type="identity",
        prediction_horizon=1,
        use_ema_target=False,
    )
    model = OthelloJEPA(cfg)
    x_context = torch.randint(0, model.config.vocab_size, (4, 6))
    x_target = torch.randint(0, model.config.vocab_size, (4, 1))
    target_positions = torch.full((4, 1), 6, dtype=torch.long)

    context_summary = model.encode_hidden(model.context_encoder, x_context)[:, -1:, :]
    out = model(x_context, x_target, target_positions)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["predictions"].shape == (4, 1, 64)
    assert out["z_targets"].shape == (4, 1, 64)
    assert torch.allclose(out["predictions"], context_summary)
    assert sum(p.numel() for p in model.predictor.parameters()) == 0
    assert not hasattr(model, "target_encoder")
    assert {"inv_loss", "var_loss", "cov_loss"} <= set(out)
    out["loss"].backward()
    assert _has_grad(model.context_encoder)


def test_identity_predictor_rejects_multi_position_prediction():
    cfg = JEPAConfig(
        variant="jepa_v2",
        view_mode="hard_disjoint_future",
        predictor_type="identity",
        prediction_horizon=2,
        use_ema_target=True,
    )
    try:
        OthelloJEPA(cfg)
    except ValueError as exc:
        assert "prediction_horizon=1" in str(exc)
    else:
        raise AssertionError("Identity predictor accepted prediction_horizon > 1")


def test_predictive_jepa_v4_hard_disjoint_future_forward_backward():
    torch.manual_seed(0)
    cfg = JEPAConfig(
        variant="jepa_v2",
        loss_type="smooth_l1",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_type="mlp_multi_pos",
        predictor_hidden_mult=2,
        predictor_dropout=0.0,
        prediction_horizon=4,
        use_ema_target=True,
    )
    model = OthelloJEPA(cfg)
    context_length = 6
    x_context = torch.randint(0, model.config.vocab_size, (4, context_length))
    x_target = torch.randint(0, model.config.vocab_size, (4, cfg.prediction_horizon))
    target_positions = torch.arange(
        context_length,
        context_length + cfg.prediction_horizon,
    ).unsqueeze(0).expand(4, -1)

    out = model(x_context, x_target, target_positions)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["predictions"].shape == (4, 4, 64)
    assert out["z_targets"].shape == (4, 4, 64)
    with torch.no_grad():
        expected_targets = model.encode_hidden(
            model.target_encoder,
            x_target,
            target_positions,
        )
        local_position_targets = model.encode_hidden(model.target_encoder, x_target)
    assert torch.allclose(out["z_targets"], expected_targets)
    assert not torch.allclose(expected_targets, local_position_targets)
    out["loss"].backward()
    assert _has_grad(model.context_encoder)
    assert _has_grad(model.predictor)
    assert not _has_grad(model.target_encoder)


def test_predictive_jepa_v4_vicreg_hard_disjoint_future_forward_backward():
    torch.manual_seed(0)
    cfg = JEPAConfig(
        variant="jepa_v2",
        loss_type="vicreg",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_type="mlp_multi_pos",
        predictor_hidden_mult=2,
        predictor_dropout=0.0,
        prediction_horizon=4,
        use_ema_target=False,
    )
    model = OthelloJEPA(cfg)
    context_length = 6
    x_context = torch.randint(0, model.config.vocab_size, (4, context_length))
    x_target = torch.randint(0, model.config.vocab_size, (4, cfg.prediction_horizon))
    target_positions = torch.arange(
        context_length,
        context_length + cfg.prediction_horizon,
    ).unsqueeze(0).expand(4, -1)

    out = model(x_context, x_target, target_positions)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["predictions"].shape == (4, 4, 64)
    assert out["z_targets"].shape == (4, 4, 64)
    assert {"inv_loss", "var_loss", "cov_loss"} <= set(out)
    assert not hasattr(model, "target_encoder")
    out["loss"].backward()
    assert _has_grad(model.context_encoder)
    assert _has_grad(model.predictor)


def test_contrastive_jepa_hard_disjoint_future_forward_backward():
    torch.manual_seed(0)
    cfg = JEPAContrastiveConfig(
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        prediction_horizon=1,
    )
    model = OthelloJEPAContrastive(cfg)
    x_context = torch.randint(0, model.config.vocab_size, (4, 6))
    x_context[1] = x_context[0]
    x_target = torch.randint(0, model.config.vocab_size, (4, 1))
    target_positions = torch.zeros((4, 1), dtype=torch.long)

    out = model(x_context, x_target, target_positions)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["n_positives_mean"] > 1.0
    out["loss"].backward()
    assert _has_grad(model.context_encoder)
    assert _has_grad(model.predictor)
    assert not _has_grad(model.target_encoder)


def test_hard_disjoint_future_configs_load():
    root = Path(__file__).resolve().parents[1]
    parser = tjm.build_parser()
    config_names = [
        "jepa_v1_vicreg_b8_hard_disjoint.yml",
        "jepa_v2_ema_b8_hard_disjoint.yml",
        "jepa_no_predictor_k1_vicreg_b8_hard_disjoint.yml",
        "jepa_linear_k1_ema_b8_hard_disjoint.yml",
        "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint.yml",
        "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint_vicreg.yml",
    ]
    for config_name in config_names:
        data = tjm.load_yaml_config(root / "configs" / config_name)
        args = parser.parse_args([
            *tjm.config_to_cli_args(data, parser),
        ])
        cfg, _ = tjm.namespace_to_train_config(args)
        assert cfg.view_mode == "hard_disjoint_future"
        assert "_hard_disjoint" in cfg.out_dir

    data = tjm.load_yaml_config(
        root / "configs" / "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint.yml"
    )
    args = parser.parse_args([*tjm.config_to_cli_args(data, parser)])
    cfg, _ = tjm.namespace_to_train_config(args)
    assert cfg.variant == "v4"
    assert tjm.build_objective_config(cfg).variant == "jepa_v2"
    x = torch.arange(48, dtype=torch.long).reshape(4, 12)
    x_context, x_target, target_positions = tjm.build_jepa_views(
        cfg,
        x,
        context_length=6,
        horizon=4,
    )
    assert torch.equal(x_context, x[:, :6])
    assert torch.equal(x_target, x[:, 6:10])
    assert torch.equal(target_positions, torch.tensor([[6, 7, 8, 9]]).expand(4, -1))

    data = tjm.load_yaml_config(root / "configs" / "jepa_v5_contrastive_hard_disjoint.yml")
    args = parser.parse_args([*tjm.config_to_cli_args(data, parser)])
    cfg, _ = tjm.namespace_to_train_config(args)
    assert tjm.objective_class(cfg) == "jepa_hard_disjoint_infonce"
    assert cfg.view_mode == "hard_disjoint_action"
    assert cfg.out_dir.endswith("_hard_disjoint")

    data = tjm.load_yaml_config(
        root / "configs" / "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint_vicreg.yml"
    )
    cli_args = tjm.config_to_cli_args(data, parser)
    assert "--no_use_ema_target" in cli_args
    args = parser.parse_args(cli_args)
    cfg, _ = tjm.namespace_to_train_config(args)
    assert cfg.variant == "v4"
    assert cfg.loss_type == "vicreg"
    assert cfg.use_ema_target is False
    model, objective_cfg = tjm.build_objective_model(cfg)
    assert objective_cfg.use_ema_target is False
    assert not hasattr(model, "target_encoder")


def test_v4_hard_disjoint_config_only_changes_target_view():
    root = Path(__file__).resolve().parents[1]
    base = tjm.load_yaml_config(root / "configs" / "jepa_v4_multi_pos_mlp_k4_b8.yml")
    hard = tjm.load_yaml_config(
        root / "configs" / "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint.yml"
    )

    assert base["experiment"]["variant"] == hard["experiment"]["variant"] == "v4"
    assert base["objective"]["view_mode"] == "disjoint_future"
    assert hard["objective"]["view_mode"] == "hard_disjoint_future"

    def experimental_settings(data):
        data = {
            section: dict(values) if isinstance(values, dict) else values
            for section, values in data.items()
            if section != "experiment"
        }
        data["paths"].pop("out_dir")
        data["paths"].pop("drive_sync_dir")
        data["objective"].pop("view_mode")
        return data

    assert experimental_settings(base) == experimental_settings(hard)
