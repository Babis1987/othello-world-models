"""Internal implementation module extracted from the canonical JEPA trainer."""

from __future__ import annotations

import torch

from othello_thesis.training._jepa_config import (
    TrainConfig,
    jepa_loss_type,
    jepa_view_mode,
    normalize_multiaction_loss_mode,
    objective_class,
)
from othello_thesis.training._jepa_runtime import unwrap_model

def objective_metric_keys(cfg: TrainConfig) -> tuple[str, ...]:
    """Return model output metric keys to average for this objective."""
    if objective_class(cfg) == "jepa_multiaction":
        step_keys: list[str] = []
        for step_idx in range(1, cfg.action_horizon + 1):
            step_keys.extend([
                f"step_{step_idx}_loss",
                f"step_{step_idx}_weighted_loss",
                f"step_{step_idx}_cosine",
            ])
            if normalize_multiaction_loss_mode(cfg.rollout_loss_mode) == "infonce":
                step_keys.extend([
                    f"step_{step_idx}_positive_accuracy",
                    f"step_{step_idx}_positive_sim",
                    f"step_{step_idx}_negative_sim",
                ])
        return (
            "main_loss",
            "rollout_loss",
            "smooth_l1_loss",
            "infonce_loss",
            "variance_loss",
            "covariance_loss",
            "positive_accuracy",
            "diag_accuracy",
            "positive_sim",
            "negative_sim",
            "c_std",
            "z_std",
            "p_std",
            "cos_sim_offdiag",
            "ema_momentum",
            "temperature",
            "horizon_weight_gamma",
            "weight_sum",
            *step_keys,
        )
    if objective_class(cfg) == "jepa_order_aware":
        return (
            "main_loss",
            "infonce_loss",
            "hybrid_ce_loss",
            "action_only_loss",
            "full_metric",
            "action_only_metric",
            "context_utility",
            "shuffled_context_metric",
            "positive_accuracy",
            "action_only_accuracy",
            "positive_fallback_rate",
            "mean_hard_negatives",
            "c_std",
            "z_std",
            "p_std",
            "cos_sim_offdiag",
            "ema_momentum",
            "temperature",
        )
    if objective_class(cfg) == "jepa_action_conditioned":
        return (
            "main_loss",
            "base_loss",
            "smooth_l1_loss",
            "infonce_loss",
            "hybrid_ce_loss",
            "action_only_loss",
            "full_metric",
            "action_only_metric",
            "context_utility",
            "shuffled_context_metric",
            "positive_accuracy",
            "action_only_accuracy",
            "c_std",
            "z_std",
            "p_std",
            "cos_sim_offdiag",
            "ema_momentum",
            "temperature",
        )
    if objective_class(cfg) == "jepa_hard_disjoint_action":
        return (
            "pull_loss",
            "push_loss",
            "mean_pos_dist",
            "mean_neg_dist",
            "dedup_fraction",
            "positives_per_anchor",
            "negatives_per_anchor",
            "c_std",
            "p_std",
            "z_std",
            "cos_sim_offdiag",
        )
    if objective_class(cfg) == "jepa_hard_disjoint_infonce":
        return (
            "infonce_loss",
            "positive_accuracy",
            "diag_accuracy",
            "contrastive_accuracy",
            "n_positives_mean",
            "positives_per_anchor",
            "negatives_per_anchor",
            "dedup_fraction",
            "z_std",
            "c_std",
            "p_std",
            "cos_sim_offdiag",
            "pred_raw_norm_mean",
            "target_raw_norm_mean",
            "logit_std",
            "logit_max",
            "logit_min",
        )
    if objective_class(cfg) == "jepa_contrastive":
        return (
            "positive_accuracy",
            "diag_accuracy",
            "contrastive_accuracy",
            "n_positives_mean",
            "z_std",
            "c_std",
            "p_std",
            "cos_sim_offdiag",
            "pred_raw_norm_mean",
            "target_raw_norm_mean",
            "logit_std",
            "logit_max",
            "logit_min",
        )
    loss_type = jepa_loss_type(cfg)
    collapse_keys: tuple[str, ...] = ()
    if jepa_view_mode(cfg) in {"disjoint_future", "hard_disjoint_future"}:
        collapse_keys = ("z_std", "c_std", "p_std", "cos_sim_offdiag")
    if cfg.position_sampling == "all":
        # The stratified all-position forward always reports collapse stats.
        collapse_keys = ("z_std", "c_std", "p_std", "cos_sim_offdiag")
    position_keys = ("positions_per_game",) if cfg.position_sampling == "all" else ()
    if loss_type == "vicreg":
        return (
            "inv_loss", "var_loss", "cov_loss", "z_pred_std", "z_tgt_std",
            *collapse_keys, *position_keys,
        )
    if loss_type == "mse":
        return ("mse_loss", "z_pred_std", "z_tgt_std", "z_pred_norm", "z_tgt_norm", *collapse_keys)
    if loss_type == "infonce":
        return (
            "infonce_loss", "positive_accuracy", "positive_sim", "negative_sim",
            "z_pred_std", "z_tgt_std", *collapse_keys, *position_keys,
        )
    return ("smooth_l1_loss", "z_pred_std", "z_tgt_std", *collapse_keys, *position_keys)

def format_objective_metrics(cfg: TrainConfig, metrics: dict[str, float]) -> str:
    """Format objective-specific metrics for progress logs."""
    if objective_class(cfg) == "jepa_multiaction":
        mode = normalize_multiaction_loss_mode(cfg.rollout_loss_mode)
        if mode == "infonce":
            return (
                f"rollout={metrics.get('rollout_loss', float('nan')):.4f} "
                f"acc={metrics.get('positive_accuracy', float('nan')):.3f} "
                f"pos_sim={metrics.get('positive_sim', float('nan')):.3f} "
                f"neg_sim={metrics.get('negative_sim', float('nan')):.3f}"
            ) + format_collapse_metrics(metrics)
        return (
            f"rollout={metrics.get('rollout_loss', float('nan')):.4f} "
            f"var={metrics.get('variance_loss', float('nan')):.4f} "
            f"cov={metrics.get('covariance_loss', float('nan')):.4f}"
        ) + format_collapse_metrics(metrics)
    if objective_class(cfg) == "jepa_order_aware":
        return (
            f"main={metrics.get('main_loss', float('nan')):.4f} "
            f"utility={metrics.get('context_utility', float('nan')):.3f} "
            f"acc={metrics.get('positive_accuracy', float('nan')):.3f} "
            f"fallback={metrics.get('positive_fallback_rate', float('nan')):.2%} "
            f"hard={metrics.get('mean_hard_negatives', float('nan')):.2f}"
        ) + format_collapse_metrics(metrics)
    if objective_class(cfg) == "jepa_action_conditioned":
        return (
            f"main={metrics.get('main_loss', float('nan')):.4f} "
            f"action_only={metrics.get('action_only_loss', float('nan')):.4f} "
            f"utility={metrics.get('context_utility', float('nan')):.3f} "
            f"shuffled={metrics.get('shuffled_context_metric', float('nan')):.3f} "
            f"acc={metrics.get('positive_accuracy', float('nan')):.3f} "
            f"action_acc={metrics.get('action_only_accuracy', float('nan')):.3f}"
        ) + format_collapse_metrics(metrics)
    if objective_class(cfg) == "jepa_hard_disjoint_action":
        return (
            f"pull={metrics.get('pull_loss', float('nan')):.4f} "
            f"push={metrics.get('push_loss', float('nan')):.4f} "
            f"d_pos={metrics.get('mean_pos_dist', float('nan')):.3f} "
            f"d_neg={metrics.get('mean_neg_dist', float('nan')):.3f} "
            f"dedup={metrics.get('dedup_fraction', float('nan')):.2%} "
            f"c_std={metrics.get('c_std', float('nan')):.3f} "
            f"p_std={metrics.get('p_std', float('nan')):.3f} "
            f"z_std={metrics.get('z_std', float('nan')):.3f}"
        )
    if objective_class(cfg) == "jepa_hard_disjoint_infonce":
        return (
            f"infonce={metrics.get('infonce_loss', float('nan')):.4f} "
            f"pos_acc={metrics.get('positive_accuracy', float('nan')):.3f} "
            f"diag_acc={metrics.get('diag_accuracy', float('nan')):.3f} "
            f"positives={metrics.get('n_positives_mean', float('nan')):.2f} "
            f"dedup={metrics.get('dedup_fraction', float('nan')):.2%} "
            f"logit_std={metrics.get('logit_std', float('nan')):.3f} "
            f"pred_norm={metrics.get('pred_raw_norm_mean', float('nan')):.2f} "
            f"tgt_norm={metrics.get('target_raw_norm_mean', float('nan')):.2f}"
        ) + format_collapse_metrics(metrics)
    if objective_class(cfg) == "jepa_contrastive":
        return (
            f"pos_acc={metrics.get('positive_accuracy', float('nan')):.3f} "
            f"diag_acc={metrics.get('diag_accuracy', float('nan')):.3f} "
            f"positives={metrics.get('n_positives_mean', float('nan')):.2f} "
            f"logit_std={metrics.get('logit_std', float('nan')):.3f} "
            f"pred_norm={metrics.get('pred_raw_norm_mean', float('nan')):.2f} "
            f"tgt_norm={metrics.get('target_raw_norm_mean', float('nan')):.2f}"
        ) + format_collapse_metrics(metrics)
    loss_type = jepa_loss_type(cfg)
    position_text = (
        f" pos/game={metrics.get('positions_per_game', float('nan')):.1f}"
        if cfg.position_sampling == "all" else ""
    )
    if loss_type == "vicreg":
        return (
            f"inv={metrics.get('inv_loss', float('nan')):.4f} "
            f"var={metrics.get('var_loss', float('nan')):.4f} "
            f"cov={metrics.get('cov_loss', float('nan')):.4f} "
            f"pred_std={metrics.get('z_pred_std', float('nan')):.3f} "
            f"tgt_std={metrics.get('z_tgt_std', float('nan')):.3f}"
        ) + format_collapse_metrics(metrics) + position_text
    if loss_type == "mse":
        return (
            f"mse={metrics.get('mse_loss', float('nan')):.4f} "
            f"pred_std={metrics.get('z_pred_std', float('nan')):.3f} "
            f"tgt_std={metrics.get('z_tgt_std', float('nan')):.3f}"
        ) + format_collapse_metrics(metrics) + position_text
    if loss_type == "infonce":
        return (
            f"infonce={metrics.get('infonce_loss', float('nan')):.4f} "
            f"pos_acc={metrics.get('positive_accuracy', float('nan')):.3f} "
            f"pos_sim={metrics.get('positive_sim', float('nan')):.3f} "
            f"neg_sim={metrics.get('negative_sim', float('nan')):.3f} "
            f"pred_std={metrics.get('z_pred_std', float('nan')):.3f} "
            f"tgt_std={metrics.get('z_tgt_std', float('nan')):.3f}"
        ) + format_collapse_metrics(metrics) + position_text
    return (
        f"smooth_l1={metrics.get('smooth_l1_loss', float('nan')):.4f} "
        f"pred_std={metrics.get('z_pred_std', float('nan')):.3f} "
        f"tgt_std={metrics.get('z_tgt_std', float('nan')):.3f}"
    ) + format_collapse_metrics(metrics) + position_text

def format_objective_header(cfg: TrainConfig, model: torch.nn.Module) -> str:
    """Format the one-line objective summary printed at run start."""
    ema_value = cfg.ema_momentum if hasattr(unwrap_model(model), "target_encoder") else "n/a"
    if objective_class(cfg) == "jepa_multiaction":
        return (
            f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
            f"loss={cfg.rollout_loss_mode} view_mode={cfg.view_mode} "
            f"predictor={cfg.predictor_type} action_horizon={cfg.action_horizon} "
            f"gamma={cfg.horizon_weight_gamma} temperature={cfg.temperature} "
            f"lambda_var={cfg.lambda_var} lambda_cov={cfg.lambda_cov} "
            f"ema={ema_value} precision={cfg.precision}"
        )
    if objective_class(cfg) == "jepa_order_aware":
        return (
            f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
            f"loss=order_aware_infonce lambda_ce={cfg.lambda_ce} "
            f"predictor={cfg.predictor_type} horizon=1 "
            f"temperature={cfg.order_temperature} ema={ema_value} precision={cfg.precision} "
            f"order_t=[{cfg.order_t_min},{cfg.order_t_max}] "
            f"groups={cfg.order_groups_per_batch} samples_per_group={cfg.order_samples_per_group} "
            f"hard_negatives={cfg.order_hard_negatives_per_anchor} "
            f"chunk_sample_multiplier={cfg.order_chunk_sample_multiplier}"
        )
    if objective_class(cfg) == "jepa_action_conditioned":
        return (
            f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
            f"loss={cfg.action_loss_mode} hybrid_base={cfg.hybrid_base} "
            f"predictor={cfg.predictor_type} horizon=1 action_dim={cfg.action_dim or cfg.d_model} "
            f"temperature={cfg.action_temperature} ema={ema_value} precision={cfg.precision} "
            f"action_t=[{cfg.action_t_min},{cfg.action_t_max}] "
            f"groups={cfg.action_groups_per_batch} samples_per_group={cfg.action_samples_per_group} "
            f"chunk_sample_multiplier={cfg.action_chunk_sample_multiplier}"
        )
    if objective_class(cfg) == "jepa_hard_disjoint_action":
        return (
            f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
            f"loss={cfg.loss_type} view_mode={cfg.view_mode} "
            f"num_modes={cfg.hard_disjoint_num_modes} margin={cfg.hard_disjoint_margin} "
            f"lambda_push={cfg.hard_disjoint_lambda_push} normalize={cfg.hard_disjoint_normalize} "
            f"ema={ema_value} precision={cfg.precision} "
            f"branch_t=[{cfg.branch_t_min},{cfg.branch_t_max}] "
            f"groups={cfg.branch_groups_per_batch} samples_per_group={cfg.branch_samples_per_group}"
        )
    if objective_class(cfg) == "jepa_hard_disjoint_infonce":
        return (
            f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
            f"loss={cfg.loss_type} view_mode={cfg.view_mode} "
            f"predictor={cfg.predictor_type} temperature={cfg.contrastive_temperature} "
            f"dedup_negative_actions={cfg.hard_disjoint_dedup_negative_actions} "
            f"ema={ema_value} precision={cfg.precision} "
            f"branch_t=[{cfg.branch_t_min},{cfg.branch_t_max}] "
            f"groups={cfg.branch_groups_per_batch} samples_per_group={cfg.branch_samples_per_group}"
        )
    if objective_class(cfg) == "jepa_contrastive":
        return (
            f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
            f"loss={cfg.loss_type} view_mode={cfg.view_mode} "
            f"predictor={cfg.predictor_type} horizon={cfg.prediction_horizon} "
            f"temperature={cfg.contrastive_temperature} "
            f"ema={ema_value} precision={cfg.precision}"
        )
    return (
        f"jepa: objective_class={cfg.objective_class} variant={cfg.variant} "
        f"view_mode={cfg.view_mode} loss={cfg.loss_type} "
        f"predictor={cfg.predictor_type} horizon={cfg.prediction_horizon} "
        f"position_sampling={cfg.position_sampling} "
        f"ema={ema_value} precision={cfg.precision}"
    )

def format_collapse_metrics(metrics: dict[str, float]) -> str:
    """Format optional collapse stats for logs."""
    if not {"z_std", "c_std", "p_std", "cos_sim_offdiag"} & set(metrics):
        return ""
    return (
        f" z_std={metrics.get('z_std', float('nan')):.3f}"
        f" c_std={metrics.get('c_std', float('nan')):.3f}"
        f" p_std={metrics.get('p_std', float('nan')):.3f}"
        f" cos={metrics.get('cos_sim_offdiag', float('nan')):.3f}"
    )
