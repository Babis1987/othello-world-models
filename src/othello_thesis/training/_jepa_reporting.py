"""Metrics and text presentation for the final JEPA objective."""

from __future__ import annotations

import torch

from othello_thesis.training._jepa_config import (
    TrainConfig,
    jepa_loss_type,
    jepa_view_mode,
    objective_class,
)
from othello_thesis.training._jepa_runtime import unwrap_model


def _assert_canonical(cfg: TrainConfig) -> None:
    objective_class(cfg)
    if jepa_loss_type(cfg) != "infonce":
        raise ValueError("The final JEPA reporter supports InfoNCE only.")
    if jepa_view_mode(cfg) != "hard_disjoint_future":
        raise ValueError("The final JEPA reporter supports hard-disjoint views only.")
    if cfg.position_sampling != "all":
        raise ValueError("The final JEPA reporter supports all-position runs only.")


def objective_metric_keys(cfg: TrainConfig) -> tuple[str, ...]:
    """Return the source-ordered metric columns for canonical runs."""
    _assert_canonical(cfg)
    return (
        "infonce_loss",
        "positive_accuracy",
        "positive_sim",
        "negative_sim",
        "z_pred_std",
        "z_tgt_std",
        "z_std",
        "c_std",
        "p_std",
        "cos_sim_offdiag",
        "positions_per_game",
    )


def format_objective_metrics(cfg: TrainConfig, metrics: dict[str, float]) -> str:
    """Format the exact canonical progress fragment."""
    _assert_canonical(cfg)
    position_text = (
        f" pos/game={metrics.get('positions_per_game', float('nan')):.1f}"
    )
    return (
        f"infonce={metrics.get('infonce_loss', float('nan')):.4f} "
        f"pos_acc={metrics.get('positive_accuracy', float('nan')):.3f} "
        f"pos_sim={metrics.get('positive_sim', float('nan')):.3f} "
        f"neg_sim={metrics.get('negative_sim', float('nan')):.3f} "
        f"pred_std={metrics.get('z_pred_std', float('nan')):.3f} "
        f"tgt_std={metrics.get('z_tgt_std', float('nan')):.3f}"
    ) + format_collapse_metrics(metrics) + position_text


def format_objective_header(cfg: TrainConfig, model: torch.nn.Module) -> str:
    """Format the exact one-line source header for canonical runs."""
    _assert_canonical(cfg)
    ema_value = (
        cfg.ema_momentum
        if hasattr(unwrap_model(model), "target_encoder")
        else "n/a"
    )
    return (
        f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
        f"view_mode={cfg.view_mode} loss={cfg.loss_type} "
        f"predictor={cfg.predictor_type} horizon={cfg.prediction_horizon} "
        f"position_sampling={cfg.position_sampling} "
        f"ema={ema_value} precision={cfg.precision}"
    )


def format_collapse_metrics(metrics: dict[str, float]) -> str:
    """Format the exact optional collapse-stat suffix."""
    if not {"z_std", "c_std", "p_std", "cos_sim_offdiag"} & set(metrics):
        return ""
    return (
        f" z_std={metrics.get('z_std', float('nan')):.3f}"
        f" c_std={metrics.get('c_std', float('nan')):.3f}"
        f" p_std={metrics.get('p_std', float('nan')):.3f}"
        f" cos={metrics.get('cos_sim_offdiag', float('nan')):.3f}"
    )
