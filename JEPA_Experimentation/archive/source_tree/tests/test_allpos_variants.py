"""Protocol and correctness tests for the all-position JEPA variant family.

Covers the five 8x8 Transformer all-position ablations added alongside the
selected v1 hard-disjoint VICReg run: v1 HD + EMA, v1 nested, v2 nested,
v2 HD, and v5-style hard-disjoint InfoNCE.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA
from othello_research.training import train_jepa, train_mamba_jepa


ROOT = Path(__file__).resolve().parents[1]

CONFIG_EXPECTATIONS = {
    "jepa_v1_vicreg_b8_hd_ema_all_positions.yml": {
        "run_name": "jepa_vicreg_b8_run_001_hd_ema_allpos_bf16",
        "variant": "v1",
        "loss": "vicreg",
        "view": "hard_disjoint_future",
        "use_ema": True,
    },
    "jepa_v1_vicreg_b8_nested_all_positions.yml": {
        "run_name": "jepa_vicreg_b8_run_001_nested_allpos_bf16",
        "variant": "v1",
        "loss": "vicreg",
        "view": "nested",
        "use_ema": False,
    },
    "jepa_v2_ema_b8_nested_all_positions.yml": {
        "run_name": "jepa_v2_ema_b8_run_001_nested_allpos_bf16",
        "variant": "v2",
        "loss": "smooth_l1",
        "view": "nested",
        "use_ema": True,
    },
    "jepa_v2_ema_b8_hd_all_positions.yml": {
        "run_name": "jepa_v2_ema_b8_run_001_hd_allpos_bf16",
        "variant": "v2",
        "loss": "smooth_l1",
        "view": "hard_disjoint_future",
        "use_ema": True,
    },
    "jepa_v5_infonce_b8_hd_all_positions.yml": {
        "run_name": "jepa_v5_infonce_b8_run_001_hd_allpos_bf16",
        "variant": "v1",
        "loss": "infonce",
        "view": "hard_disjoint_future",
        "use_ema": True,
    },
}


def _resolved(filename: str) -> train_jepa.TrainConfig:
    args, _, _ = train_jepa.parse_configured_args(
        ["--config", str(ROOT / "configs" / filename)]
    )
    cfg, _ = train_jepa.namespace_to_train_config(args)
    train_jepa.validate_train_config(cfg)
    return cfg


def _uses_ema(cfg: train_jepa.TrainConfig) -> bool:
    if cfg.use_ema_target is not None:
        return bool(cfg.use_ema_target)
    return train_jepa.normalize_jepa_variant(cfg.variant) == "jepa_v2"


def test_allpos_variant_configs_resolve_and_validate() -> None:
    for filename, expected in CONFIG_EXPECTATIONS.items():
        cfg = _resolved(filename)
        data = train_jepa.load_yaml_config(ROOT / "configs" / filename)
        assert data["experiment"]["name"] == expected["run_name"], filename
        assert cfg.board_size == 8, filename
        assert cfg.position_sampling == "all", filename
        assert cfg.prediction_horizon == 1, filename
        assert cfg.variant == expected["variant"], filename
        assert train_jepa.jepa_loss_type(cfg) == expected["loss"], filename
        assert train_jepa.jepa_view_mode(cfg) == expected["view"], filename
        assert _uses_ema(cfg) == expected["use_ema"], filename
        assert cfg.predictor_type == "linear", filename
        assert cfg.batch_size == 256, filename
        assert cfg.precision == "bf16", filename


def test_allpos_validator_rejects_shared_encoder_without_anticollapse() -> None:
    for loss in ("smooth_l1", "infonce"):
        cfg = train_jepa.TrainConfig(
            out_dir="out",
            variant="v1",
            loss_type=loss,
            view_mode="hard_disjoint_future",
            prediction_horizon=1,
            position_sampling="all",
            use_ema_target=False,
        )
        with pytest.raises(ValueError, match="EMA target"):
            train_jepa.validate_train_config(cfg)


def test_allpos_validator_rejects_unsupported_view_and_loss() -> None:
    cfg = train_jepa.TrainConfig(
        out_dir="out",
        variant="v1",
        loss_type="mse",
        view_mode="hard_disjoint_future",
        prediction_horizon=1,
        position_sampling="all",
    )
    with pytest.raises(ValueError, match="vicreg, smooth_l1, or"):
        train_jepa.validate_train_config(cfg)
    cfg = train_jepa.TrainConfig(
        out_dir="out",
        variant="v1",
        loss_type="vicreg",
        view_mode="disjoint_future",
        prediction_horizon=1,
        position_sampling="all",
        predictor_type="mlp_multi_pos",
    )
    with pytest.raises(ValueError, match="hard_disjoint_future"):
        train_jepa.validate_train_config(cfg)


def _tiny_model(*, variant: str, loss_type: str, view_mode: str, use_ema: bool | None) -> OthelloJEPA:
    cfg = JEPAConfig(
        variant=variant,
        loss_type=loss_type,
        view_mode=view_mode,
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=32,
        dropout=0.0,
        predictor_type="linear",
        prediction_horizon=1,
        use_ema_target=use_ema,
        contrastive_temperature=0.1,
    )
    return OthelloJEPA(cfg)


VARIANT_MATRIX = [
    # (label, variant, loss, view, use_ema)
    ("v1_hd_ema", "v1", "vicreg", "hard_disjoint_future", True),
    ("v1_nested", "v1", "vicreg", "nested", None),
    ("v2_nested", "v2", "smooth_l1", "nested", None),
    ("v2_hd", "v2", "smooth_l1", "hard_disjoint_future", None),
    ("v5_infonce_hd", "v1", "infonce", "hard_disjoint_future", True),
]


@pytest.mark.parametrize("label,variant,loss,view,use_ema", VARIANT_MATRIX)
def test_forward_all_position_variant_backward(label, variant, loss, view, use_ema) -> None:
    torch.manual_seed(0)
    model = _tiny_model(variant=variant, loss_type=loss, view_mode=view, use_ema=use_ema)
    pad = 8 * 8 - 4
    x = torch.randint(0, pad, (6, 12))
    x[4, 9:] = pad  # short game with padding

    out, pairs = train_jepa.forward_all_position_v1(
        model, x, pad_token=pad, target_chunk_size=65536, stats_chunk_size=64
    )
    assert pairs > 0, label
    assert torch.isfinite(out["loss"]), (label, out["loss"])
    out["loss"].backward()

    context_grads = [
        p.grad for p in model.context_encoder.parameters() if p.grad is not None
    ]
    assert context_grads, f"{label}: context encoder received no gradients"
    assert all(torch.isfinite(g).all() for g in context_grads), label

    if hasattr(model, "target_encoder"):
        assert all(
            p.grad is None for p in model.target_encoder.parameters()
        ), f"{label}: EMA target encoder must not receive gradients"
        # The EMA update must move the target toward the context encoder.
        before = [p.detach().clone() for p in model.target_encoder.parameters()]
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(1.0)
        model.update_target_encoder(momentum=0.5)
        after = list(model.target_encoder.parameters())
        assert any(
            not torch.equal(b, a.detach()) for b, a in zip(before, after)
        ), f"{label}: EMA update did not change the target encoder"

    if loss == "infonce":
        assert 0.0 <= float(out["positive_accuracy"]) <= 1.0, label
        assert torch.isfinite(out["positive_sim"]), label
        assert torch.isfinite(out["negative_sim"]), label
    if loss == "smooth_l1":
        assert torch.isfinite(out["smooth_l1_loss"]), label


@pytest.mark.parametrize("label,variant,loss,view,use_ema", VARIANT_MATRIX)
def test_allpos_outputs_cover_declared_metric_keys(label, variant, loss, view, use_ema) -> None:
    torch.manual_seed(0)
    model = _tiny_model(variant=variant, loss_type=loss, view_mode=view, use_ema=use_ema)
    pad = 8 * 8 - 4
    x = torch.randint(0, pad, (4, 10))
    out, pairs = train_jepa.forward_all_position_v1(
        model, x, pad_token=pad, target_chunk_size=4096, stats_chunk_size=8
    )
    assert pairs > 0

    cfg = train_jepa.TrainConfig(
        out_dir="out",
        variant=variant,
        loss_type=loss,
        view_mode=view,
        prediction_horizon=1,
        position_sampling="all",
        use_ema_target=use_ema,
    )
    for key in train_jepa.objective_metric_keys(cfg):
        assert key in out, f"{label}: forward output is missing declared metric {key!r}"


def test_nested_shared_target_reuses_context_pass() -> None:
    """With a shared encoder, nested targets must equal the context hiddens."""
    torch.manual_seed(0)
    model = _tiny_model(variant="v1", loss_type="vicreg", view_mode="nested", use_ema=None)
    model.eval()
    pad = 8 * 8 - 4
    x = torch.randint(0, pad, (3, 9))
    with torch.no_grad():
        hidden = model.encode_hidden(model.context_encoder, x)
        out, pairs = train_jepa.forward_all_position_v1(
            model, x, pad_token=pad, target_chunk_size=4096, stats_chunk_size=8
        )
    assert pairs == 3 * 8
    # The nested-target invariance term at full alignment must be finite and
    # correspond to predicting hidden[t] from hidden[t-1]; verify the target
    # grid indirectly through the reported target std (equal to hidden std).
    assert torch.isfinite(out["z_tgt_std"])
    # z_tgt_std includes the variance eps inside the sqrt while z_std does
    # not, so they agree only up to ~sqrt-eps regularization.
    assert abs(float(out["z_tgt_std"]) - float(out["z_std"])) < 1e-3
    assert float(hidden.std()) > 0.0


def test_v6_allpos_configs_resolve_and_validate() -> None:
    """The v6 all-position configs must parse and clear validation.

    The gate in validate_train_config now admits position_sampling=all for
    objective_class=jepa_hard_disjoint_action; these configs guard that path.
    The transformer config goes through the base parser, the Mamba one through
    the Mamba parser (which accepts the SSM-only keys).
    """
    # Transformer v6 all-position config.
    args, _, _ = train_jepa.parse_configured_args(
        ["--config", str(ROOT / "configs" / "jepa_v6_hd_action_all_positions.yml")]
    )
    cfg, _ = train_jepa.namespace_to_train_config(args)
    train_jepa.validate_train_config(cfg)
    assert train_jepa.objective_class(cfg) == "jepa_hard_disjoint_action"
    assert cfg.position_sampling == "all"
    assert cfg.board_size == 8
    assert cfg.predictor_type == "mlp_modes"
    assert cfg.batch_size == cfg.branch_groups_per_batch * cfg.branch_samples_per_group

    # Mamba v6 all-position config (SSM-only keys need the Mamba parser).
    m_args, _, _ = train_mamba_jepa.parse_configured_args(
        ["--config", str(ROOT / "configs" / "mamba_jepa_v6_hd_action_all_positions.yml")]
    )
    m_cfg, _ = train_mamba_jepa.namespace_to_train_config(m_args)
    train_mamba_jepa.validate_train_config(m_cfg)
    assert train_jepa.objective_class(m_cfg) == "jepa_hard_disjoint_action"
    assert m_cfg.position_sampling == "all"
    assert m_cfg.encoder_architecture == "mamba"
    assert m_cfg.batch_size == m_cfg.branch_groups_per_batch * m_cfg.branch_samples_per_group


def test_v6_allpos_rejected_for_infonce_sibling() -> None:
    """position_sampling=all must still reject the v5-style InfoNCE objective."""
    cfg = train_jepa.TrainConfig(
        out_dir="out",
        objective_class="jepa_hard_disjoint_infonce",
        view_mode="hard_disjoint_action",
        position_sampling="all",
    )
    with pytest.raises(ValueError, match="jepa_hard_disjoint_action"):
        train_jepa.validate_train_config(cfg)


def test_smoke_evaluation_path_supports_variants() -> None:
    """eval_chunks uses the same forward; a no-grad pass must also work."""
    torch.manual_seed(0)
    model = _tiny_model(variant="v2", loss_type="smooth_l1", view_mode="nested", use_ema=None)
    model.eval()
    pad = 8 * 8 - 4
    x = torch.randint(0, pad, (4, 10))
    with torch.no_grad():
        out, pairs = train_jepa.forward_all_position_v1(
            model, x, pad_token=pad, target_chunk_size=4096, stats_chunk_size=8
        )
    assert pairs > 0
    assert torch.isfinite(out["loss"])
