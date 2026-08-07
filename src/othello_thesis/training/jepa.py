"""Public facade for canonical JEPA training.

Implementation is split by responsibility so notebooks can read the training
composition without navigating a monolith.  The public import path remains
``othello_thesis.training.jepa`` for scripts, notebooks, and checkpoints.
"""

from __future__ import annotations

from othello_thesis.training._jepa_all_positions import (
    _forward_all_position_kstep,
    build_jepa_views,
    build_local_target_positions,
    build_target_positions,
    forward_all_position_v1,
    maybe_warn_collapse,
    output_metric,
    sample_valid_window,
    valid_window_mask,
)
from othello_thesis.training._jepa_config import (
    CONFIG_METADATA_KEYS,
    TrainConfig,
    config_to_cli_args,
    flatten_config_leaves,
    jepa_loss_type,
    jepa_variant,
    jepa_view_mode,
    load_yaml_config,
    normalize_contrastive_loss_type,
    normalize_contrastive_variant,
    normalize_experiment_variant,
    normalize_multiaction_loss_mode,
    objective_class,
    objective_horizon,
    parser_destinations,
    write_config_records,
)
from othello_thesis.training._jepa_cli import (
    build_parser,
    namespace_to_train_config,
    parse_configured_args,
    validate_train_config,
)
from othello_thesis.training._jepa_loop import (
    dispatch_eval_chunks,
    dispatch_train_one_chunk,
    eval_chunks,
    train_one_chunk,
)
from othello_thesis.training._jepa_models import (
    ModelFactory,
    build_model_from_checkpoint,
    build_model_from_objective_config,
    build_objective_config,
    build_objective_model,
    filter_kwargs_for_dataclass,
)
from othello_thesis.training._jepa_reporting import (
    format_collapse_metrics,
    format_objective_header,
    format_objective_metrics,
    objective_metric_keys,
)
from othello_thesis.training._jepa_runner import main, run_training
from othello_thesis.training._jepa_runtime import (
    append_csv,
    autocast_dtype,
    estimate_total_steps,
    fraction_checkpoint_name,
    get_lr,
    init_wandb,
    list_chunks,
    load_model_state,
    make_loader,
    model_state_dict,
    parse_fraction_checkpoints,
    parse_milestones,
    read_manifest_counts,
    resolve_chunks,
    save_checkpoint,
    sync_to_drive,
    unwrap_model,
    update_target_encoder,
    use_autocast,
)
from othello_thesis.training.progress import (
    LineProgress as _LineProgress,
    NoOpBar as _NoOpBar,
    make_chunk_progress,
    make_manual_progress,
    tqdm,
    tqdm_disabled as _tqdm_disabled,
)
from othello_thesis.objectives.jepa import (
    normalize_jepa_loss_type,
    normalize_jepa_variant,
    normalize_jepa_view_mode,
)


__all__ = [
    "ModelFactory",
    "TrainConfig",
    "append_csv",
    "autocast_dtype",
    "build_jepa_views",
    "build_local_target_positions",
    "build_model_from_checkpoint",
    "build_model_from_objective_config",
    "build_objective_config",
    "build_objective_model",
    "build_parser",
    "build_target_positions",
    "config_to_cli_args",
    "dispatch_eval_chunks",
    "dispatch_train_one_chunk",
    "estimate_total_steps",
    "eval_chunks",
    "filter_kwargs_for_dataclass",
    "format_collapse_metrics",
    "format_objective_header",
    "format_objective_metrics",
    "forward_all_position_v1",
    "fraction_checkpoint_name",
    "get_lr",
    "jepa_loss_type",
    "jepa_variant",
    "jepa_view_mode",
    "list_chunks",
    "load_model_state",
    "load_yaml_config",
    "main",
    "make_loader",
    "namespace_to_train_config",
    "normalize_contrastive_loss_type",
    "normalize_contrastive_variant",
    "normalize_experiment_variant",
    "normalize_jepa_loss_type",
    "normalize_jepa_variant",
    "normalize_jepa_view_mode",
    "normalize_multiaction_loss_mode",
    "objective_class",
    "objective_horizon",
    "objective_metric_keys",
    "parse_configured_args",
    "parse_fraction_checkpoints",
    "parse_milestones",
    "read_manifest_counts",
    "resolve_chunks",
    "run_training",
    "sample_valid_window",
    "save_checkpoint",
    "sync_to_drive",
    "train_one_chunk",
    "unwrap_model",
    "update_target_encoder",
    "use_autocast",
    "valid_window_mask",
    "validate_train_config",
    "write_config_records",
]


if __name__ == "__main__":
    main()
