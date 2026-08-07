import pytest
import torch

from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA
from othello_research.objectives.jepa_hard_disjoint_action import (
    JEPAHardDisjointActionConfig,
    OthelloJEPAHardDisjointAction,
)
from othello_research.objectives.jepa_hard_disjoint_infonce import (
    JEPAHardDisjointInfoNCEConfig,
    OthelloJEPAHardDisjointInfoNCE,
)
from othello_research.training import train_jepa
from othello_research.training import train_mamba_jepa as tjm


def test_mamba_jepa_v1_vicreg_forward_backward() -> None:
    torch.manual_seed(0)
    cfg = JEPAConfig(
        variant="v1",
        loss_type="vicreg",
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        encoder_architecture="mamba",
        d_state=8,
        d_conv=3,
        expand=2,
        mamba_backend="torch",
        predictor_type="linear",
        prediction_horizon=1,
    )
    model = OthelloJEPA(cfg)
    x = torch.randint(0, model.config.vocab_size, (6, 7))
    out = model(x[:, :6], x[:, 6:7], target_positions=torch.full((6, 1), 6))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())


def test_mamba_jepa_v5_infonce_forward_backward_target_frozen() -> None:
    torch.manual_seed(1)
    cfg = JEPAHardDisjointInfoNCEConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        encoder_architecture="mamba",
        d_state=8,
        d_conv=3,
        expand=2,
        mamba_backend="torch",
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        predictor_dropout=0.0,
    )
    model = OthelloJEPAHardDisjointInfoNCE(cfg)
    x_context = torch.randint(0, model.config.vocab_size, (8, 5))
    x_context[:4] = x_context[0]
    x_context[4:] = x_context[4]
    prefix_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    next_actions = torch.randint(0, model._vocab_actions, (8,))
    out = model(x_context, next_actions, prefix_ids)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_mamba_jepa_v6_action_margin_forward_backward_target_frozen() -> None:
    torch.manual_seed(2)
    cfg = JEPAHardDisjointActionConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        encoder_architecture="mamba",
        d_state=8,
        d_conv=3,
        expand=2,
        mamba_backend="torch",
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        num_modes=4,
    )
    model = OthelloJEPAHardDisjointAction(cfg)
    x_context = torch.randint(0, model.config.vocab_size, (8, 5))
    x_context[:4] = x_context[0]
    x_context[4:] = x_context[4]
    prefix_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    next_actions = torch.randint(0, model._vocab_actions, (8,))
    out = model(x_context, next_actions, prefix_ids)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_mamba_jepa_yaml_keys_are_accepted() -> None:
    parser = tjm.build_parser()
    for name in [
        "mamba_jepa_v1_hd_vicreg.yml",
        "mamba_jepa_v1_hd_vicreg_ema_b8_all_positions.yml",
        "mamba_jepa_v5_hd_infonce.yml",
        "mamba_jepa_v5_hd_infonce_b8_all_positions.yml",
        "mamba_jepa_v6_hd_action.yml",
    ]:
        data = tjm.load_yaml_config(tjm.Path("configs") / name)
        args = parser.parse_args([*tjm.config_to_cli_args(data, parser)])
        cfg, _ = tjm.namespace_to_train_config(args)
        assert cfg.encoder_architecture == "mamba"
        assert cfg.mamba_backend == "mamba_ssm"


@pytest.mark.parametrize(
    "config_name,expected_loss",
    [
        ("mamba_jepa_v1_hd_vicreg_ema_b8_all_positions.yml", "vicreg"),
        ("mamba_jepa_v5_hd_infonce_b8_all_positions.yml", "infonce"),
    ],
)
def test_mamba_jepa_all_position_ema_candidate_configs(
    config_name: str, expected_loss: str
) -> None:
    args, _, data = tjm.parse_configured_args(
        ["--config", str(tjm.Path("configs") / config_name)]
    )
    cfg, _ = tjm.namespace_to_train_config(args)
    tjm.validate_train_config(cfg)

    assert data["experiment"]["name"].endswith("_allpos_b8")
    assert cfg.objective_class == "jepa"
    assert cfg.encoder_architecture == "mamba"
    assert cfg.board_size == 8
    assert cfg.view_mode == "hard_disjoint_future"
    assert cfg.loss_type == expected_loss
    assert cfg.position_sampling == "all"
    assert cfg.prediction_horizon == 1
    assert cfg.use_ema_target is True
    assert cfg.ema_momentum == pytest.approx(0.996)
    assert cfg.predictor_type == "linear"
    assert cfg.batch_size == 256
    assert cfg.precision == "bf16"
    assert cfg.passes_over_data == 1
    assert cfg.li_split_train_chunks == 200


@pytest.mark.parametrize("loss_type", ["vicreg", "infonce"])
def test_mamba_jepa_all_position_ema_candidates_forward_backward(
    loss_type: str,
) -> None:
    torch.manual_seed(3)
    cfg = JEPAConfig(
        variant="v1",
        loss_type=loss_type,
        view_mode="hard_disjoint_future",
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        encoder_architecture="mamba",
        d_state=8,
        d_conv=3,
        expand=2,
        mamba_backend="torch",
        predictor_type="linear",
        prediction_horizon=1,
        use_ema_target=True,
        ema_momentum=0.996,
        contrastive_temperature=0.1,
    )
    model = OthelloJEPA(cfg)
    pad_token = 8 * 8 - 4
    x = torch.randint(0, pad_token, (6, 12))
    x[-1, 9:] = pad_token

    out, pairs = train_jepa.forward_all_position_v1(
        model,
        x,
        pad_token=pad_token,
        target_chunk_size=65_536,
        stats_chunk_size=64,
    )
    assert pairs == 6 * 11 - 3
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())



def test_train_mamba_jepa_rejects_fp16_precision() -> None:
    parser = tjm.build_parser()
    args = parser.parse_args([
        "--out_dir", "__unused__",
        "--encoder_architecture", "mamba",
        "--precision", "fp16",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    with pytest.raises(ValueError, match="must use bf16 or fp32"):
        tjm.validate_train_config(cfg)
