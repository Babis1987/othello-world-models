from __future__ import annotations

import pytest
import torch

from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA
from othello_research.objectives.jepa_contrastive import (
    JEPAContrastiveConfig,
    OthelloJEPAContrastive,
)
from othello_research.training.train_jepa import filter_kwargs_for_dataclass


VARIANTS = {
    "v1": ("jepa", {
        "loss_type": "vicreg",
        "use_ema_target": False,
        "view_mode": "nested",
        "prediction_horizon": 1,
        "predictor_type": "linear",
    }),
    "v2": ("jepa", {
        "loss_type": "smooth_l1",
        "use_ema_target": True,
        "view_mode": "nested",
        "prediction_horizon": 4,
        "predictor_type": "transformer",
    }),
    "v3": ("jepa", {
        "loss_type": "smooth_l1",
        "use_ema_target": True,
        "view_mode": "nested",
        "prediction_horizon": 8,
        "predictor_type": "mlp",
        "predictor_hidden_dim": 64,
        "predictor_n_layers": 2,
        "ema_momentum": 0.99,
    }),
    "v4": ("jepa", {
        "loss_type": "smooth_l1",
        "use_ema_target": True,
        "view_mode": "disjoint_future",
        "prediction_horizon": 4,
        "predictor_type": "mlp_multi_pos",
    }),
    "v5_contrastive": ("jepa_contrastive", {
        "view_mode": "disjoint_future",
        "prediction_horizon": 1,
        "predictor_type": "mlp",
        "contrastive_temperature": 0.1,
        "contrastive_use_prefix_mask": True,
    }),
}


@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
def test_forward_backward_smoke(
    variant_name,
    tiny_config_dict,
    synthetic_game_batch,
    cpu_device,
):
    objective_class, extra = VARIANTS[variant_name]
    config_dict = {**tiny_config_dict, **extra}

    if objective_class == "jepa":
        config = JEPAConfig(**filter_kwargs_for_dataclass(JEPAConfig, config_dict))
        model = OthelloJEPA(config).to(cpu_device)
    else:
        config = JEPAContrastiveConfig(
            **filter_kwargs_for_dataclass(JEPAContrastiveConfig, config_dict)
        )
        model = OthelloJEPAContrastive(config).to(cpu_device)

    batch = synthetic_game_batch(B=4, T=30, board_size=8, seed=42).to(cpu_device)
    t = 10
    K = config.prediction_horizon
    x_context = batch[:, :t]
    x_target = batch[:, : t + K]
    target_positions = torch.arange(t, t + K).unsqueeze(0).expand(4, K).to(cpu_device)

    out = model(x_context, x_target, target_positions)
    assert isinstance(out["loss"], torch.Tensor)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    if objective_class == "jepa_contrastive":
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

    out["loss"].backward()

    assert any(
        param.grad is not None for param in model.context_encoder.parameters()
    ), f"Missing context gradients for {variant_name}"
    assert any(
        param.grad is not None for param in model.predictor.parameters()
    ), f"Missing predictor gradients for {variant_name}"

    if hasattr(model, "target_encoder"):
        for name, param in model.target_encoder.named_parameters():
            assert param.grad is None, f"Unexpected gradient on target.{name}"
