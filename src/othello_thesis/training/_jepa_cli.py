"""CLI and validation for final-v5 all-position contrastive JEPA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from othello_thesis.objectives.jepa import (
    normalize_jepa_loss_type,
    normalize_jepa_variant,
    normalize_jepa_view_mode,
)
from othello_thesis.training._jepa_config import (
    TrainConfig,
    config_to_cli_args,
    load_yaml_config,
    objective_class,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="YAML experiment config. CLI flags override config values.")
    parser.add_argument("--print_resolved_config", "--print-resolved-config",
                        action="store_true",
                        help="Print the config after YAML + CLI resolution and exit.")

    # Paths and split
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Single directory; first N chunks become train, rest val.")
    parser.add_argument("--pair_index_path", type=str, default=None,
                        help="Offline SQLite pair index required by v8 order-aware JEPA.")
    parser.add_argument("--li_split_train_chunks", type=int, default=200)
    parser.add_argument("--objective_class", type=str, default="jepa",
                        choices=[
                            "jepa",
                            "predictive",
                            "jepa_predictive",
                            "jepa_contrastive",
                            "contrastive",
                            "v5",
                            "jepa_hard_disjoint_infonce",
                            "hard_disjoint_infonce",
                            "v5_hard_disjoint",
                            "v5_hard_disjoint_infonce",
                            "jepa_hard_disjoint_action",
                            "hard_disjoint_action",
                            "v6",
                            "jepa_action_conditioned",
                            "action_conditioned",
                            "v7",
                            "jepa_order_aware",
                            "order_aware",
                            "v8",
                            "jepa_multiaction",
                            "jepa_multi_action",
                            "multiaction",
                            "multi_action",
                            "jepa_v9_multiaction",
                            "v9",
                            "v9a",
                            "v9b",
                            "jepa_v9a",
                            "jepa_v9b",
                            "family_infonce",
                            "family",
                            "transposition_family",
                        ],
                        help="Objective implementation selected by config.")
    parser.add_argument("--variant", type=str, default="v2",
                        choices=[
                            "jepa_v1",
                            "jepa_v2",
                            "jepa_v3",
                            "jepa_v4",
                            "jepa_v5",
                            "jepa_v6",
                            "jepa_v7",
                            "jepa_v8",
                            "jepa_v9",
                            "jepa_v5_contrastive",
                            "contrastive_disjoint_k1",
                            "cpc_disjoint_future",
                            "contrastive",
                            "cpc",
                            "v1",
                            "v2",
                            "v3",
                            "v4",
                            "v5",
                            "v5_hard_disjoint",
                            "v5_hard_disjoint_infonce",
                            "v6",
                            "v7",
                            "v8",
                            "v9",
                            "v9a",
                            "v9b",
                            "jepa_v9a",
                            "jepa_v9b",
                            "vicreg",
                            "ema",
                            "ema_smooth_l1",
                            "jepa_v5_hard_disjoint_infonce",
                            "jepa_v6_hard_disjoint_action",
                            "hard_disjoint_action",
                            "jepa_v7_action_conditioned",
                            "jepa_action_conditioned",
                            "action_conditioned",
                            "jepa_v8_order_aware",
                            "jepa_order_aware",
                            "order_aware",
                            "jepa_v9_multiaction",
                            "jepa_multiaction",
                            "jepa_multi_action",
                            "multiaction",
                            "multi_action",
                            "family_infonce",
                            "family",
                            "transposition_family",
                        ],
                        help="Public thesis experiment family. Prefer v1 through v9.")
    parser.add_argument("--loss_type", type=str, default="auto",
                        choices=[
                            "auto",
                            "mse",
                            "l2",
                            "smooth_l1",
                            "smooth-l1",
                            "smoothl1",
                            "huber",
                            "vicreg",
                            "infonce",
                            "info_nce",
                            "contrastive",
                            "distance_margin",
                            "distance-margin",
                            "order_aware_infonce",
                        ],
                        help="Embedding prediction loss. auto = VICReg for v1, Smooth L1 for predictive v2-v4.")

    # Model architecture
    parser.add_argument("--board_size", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # JEPA objective
    parser.add_argument("--predictor_type", type=str, default="transformer",
                        choices=["identity", "linear", "mlp", "transformer", "mlp_multi_pos", "mlp_modes",
                                 "action_conditioned_mlp"])
    parser.add_argument("--predictor_hidden_dim", type=int, default=None,
                        help="Hidden size for MLP predictor. None uses d_model.")
    parser.add_argument("--predictor_hidden_mult", type=int, default=4,
                        help="Hidden multiplier for mlp_multi_pos predictor.")
    parser.add_argument("--predictor_n_layers", type=int, default=4)
    parser.add_argument("--predictor_n_heads", type=int, default=8)
    parser.add_argument("--predictor_dropout", type=float, default=None,
                        help="Predictor dropout. None reuses encoder dropout.")
    parser.add_argument("--predictor_activation", type=str, default="relu",
                        choices=["relu", "gelu", "silu"],
                        help="Activation used by the MLP predictor.")

    # v1 VICReg objective
    parser.add_argument("--vicreg_lambda", type=float, default=25.0,
                        help="VICReg invariance loss weight for jepa_v1.")
    parser.add_argument("--vicreg_mu", type=float, default=25.0,
                        help="VICReg variance loss weight for jepa_v1.")
    parser.add_argument("--vicreg_nu", type=float, default=1.0,
                        help="VICReg covariance loss weight for jepa_v1.")
    parser.add_argument("--variance_threshold", type=float, default=1.0,
                        help="VICReg standard-deviation floor for jepa_v1.")
    parser.add_argument("--variance_eps", type=float, default=1e-4,
                        help="VICReg variance epsilon for jepa_v1.")

    # v2 EMA Smooth L1 objective
    parser.add_argument("--view_mode", type=str, default="nested",
                        choices=["nested", "disjoint_future", "disjoint", "future",
                                 "hard_disjoint", "hard_disjoint_future",
                                 "hard-disjoint", "hard-disjoint-future",
                                 "target_only_future", "target-only-future",
                                 "hard_disjoint_action", "hard-disjoint-action",
                                 "order_aware"],
                        help="JEPA view construction mode.")
    parser.add_argument("--prediction_horizon", type=int, default=4,
                        help="Number of future target embeddings to predict.")
    parser.add_argument("--ema_momentum", type=float, default=0.996,
                        help="EMA momentum for target encoder update.")
    parser.add_argument("--smooth_l1_beta", type=float, default=1.0,
                        help="Beta parameter for torch Smooth L1 loss.")
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use_ema_target", action="store_true", default=None,
                           help="Force a separate EMA target encoder.")
    ema_group.add_argument("--no_use_ema_target", dest="use_ema_target",
                           action="store_false",
                           help="Disable the EMA target encoder and use the shared encoder target path.")
    parser.add_argument("--log_collapse_stats_every_steps", type=int, default=0,
                        help="Print v4 collapse stats every N optimiser steps. 0 = off.")
    parser.add_argument("--collapse_warning_z_std", type=float, default=0.1)
    parser.add_argument("--collapse_warning_cos_sim", type=float, default=0.95)
    parser.add_argument("--collapse_warning_contrastive_accuracy", type=float, default=0.05)

    # v5 contrastive objective
    parser.add_argument("--contrastive_loss", type=str, default="infonce",
                        choices=["infonce"],
                        help="Contrastive loss label for experiment logging.")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--contrastive_use_prefix_mask", action="store_true", default=True)

    # v6 hard-disjoint action objective
    parser.add_argument("--hard_disjoint_num_modes", type=int, default=10,
                        help="Number of predicted modes M for v6.")
    parser.add_argument("--hard_disjoint_margin", type=float, default=1.0,
                        help="Hinge margin in the v6 push term.")
    parser.add_argument("--hard_disjoint_lambda_push", type=float, default=1.0,
                        help="Weight of the v6 push term.")
    parser.add_argument("--hard_disjoint_normalize", action="store_true", default=True,
                        help="L2-normalize predicted modes and targets before distances.")
    parser.add_argument("--no_hard_disjoint_normalize",
                        dest="hard_disjoint_normalize", action="store_false")
    parser.add_argument("--hard_disjoint_dedup_negative_actions", action="store_true", default=True,
                        help="Apply per-pair phantom replacement for negatives whose action collides with a positive.")
    parser.add_argument("--no_hard_disjoint_dedup_negative_actions",
                        dest="hard_disjoint_dedup_negative_actions", action="store_false")
    parser.add_argument("--branch_t_min", type=int, default=4,
                        help="Minimum context length t for v6 prefix-grouped batches.")
    parser.add_argument("--branch_t_max", type=int, default=10,
                        help="Maximum context length t for v6 prefix-grouped batches.")
    parser.add_argument("--branch_groups_per_batch", type=int, default=64,
                        help="Number of prefix groups per batch.")
    parser.add_argument("--branch_samples_per_group", type=int, default=4,
                        help="Number of games per prefix group.")
    parser.add_argument("--no_contrastive_use_prefix_mask",
                        dest="contrastive_use_prefix_mask",
                        action="store_false")

    # v7 action-conditioned objective
    parser.add_argument("--loss_mode", type=str, default=None,
                        choices=["smooth_l1", "grouped_infonce", "hybrid"],
                        help="v7 objective loss mode; preferred YAML field.")
    parser.add_argument("--action_loss_mode", type=str, default="smooth_l1",
                        choices=["smooth_l1", "grouped_infonce", "hybrid"],
                        help="Backward-compatible alias for --loss_mode.")
    parser.add_argument("--hybrid_base", type=str, default="grouped_infonce",
                        choices=["smooth_l1", "grouped_infonce"])
    parser.add_argument("--action_dim", type=int, default=None)
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize_embeddings", action="store_true", default=None)
    normalize_group.add_argument("--no_normalize_embeddings", dest="normalize_embeddings",
                                 action="store_false")
    parser.add_argument("--action_temperature", type=float, default=0.1)
    parser.add_argument("--action_temperature_learnable", action="store_true")
    parser.add_argument("--action_groups_per_batch", type=int, default=8)
    parser.add_argument("--action_samples_per_group", type=int, default=16)
    parser.add_argument("--action_chunk_sample_multiplier", type=float, default=1.0)
    parser.add_argument("--action_t_min", type=int, default=4)
    parser.add_argument("--action_t_max", type=int, default=10)
    parser.add_argument("--lambda_ce", type=float, default=0.05)

    # v8 order-aware objective
    parser.add_argument("--order_temperature", type=float, default=0.1)
    parser.add_argument("--order_temperature_learnable", action="store_true")
    parser.add_argument("--order_groups_per_batch", type=int, default=8)
    parser.add_argument("--order_samples_per_group", type=int, default=16)
    parser.add_argument("--order_hard_negatives_per_anchor", type=int, default=1)
    parser.add_argument("--order_chunk_sample_multiplier", type=float, default=1.0)
    parser.add_argument("--order_pair_chunk_cache_size", type=int, default=4)
    parser.add_argument(
        "--order_min_pair_anchors_per_group",
        type=int,
        default=0,
        help=(
            "Minimum anchors per v8 action group that must have a cross-order "
            "positive or same-set hard negative."
        ),
    )
    parser.add_argument("--order_index_use_constructive_pairs", action="store_true",
                        help="Record that the offline v8 index includes constructive pairs.")
    parser.add_argument("--order_index_use_surface_hard_negatives", action="store_true",
                        help="Record that the offline v8 index includes near-surface hard negatives.")
    parser.add_argument("--order_surface_jaccard_threshold", type=float, default=0.9,
                        help="Occupied-cell Jaccard threshold for v8 surface hard negatives.")
    parser.add_argument("--order_surface_max_pairs_per_anchor", type=int, default=1,
                        help="Maximum same-chunk surface hard negatives mined per anchor.")
    parser.add_argument("--order_index_positions_per_game", type=int, default=1,
                        help="Balanced prefix positions stored per game in the offline v8 index.")
    parser.add_argument("--order_index_sampling_seed", type=int, default=42)
    parser.add_argument("--order_t_min", type=int, default=4)
    parser.add_argument("--order_t_max", type=int, default=40)
    parser.add_argument("--order_utility_early_stop_threshold", type=float, default=-1.0,
                        help="Stop v8 after sustained context utility at/below this value; negative disables.")
    parser.add_argument("--order_utility_early_stop_window", type=int, default=500)
    parser.add_argument("--order_utility_early_stop_min_steps", type=int, default=5000)

    # v9 multi-action rollout objective
    parser.add_argument("--action_horizon", type=int, default=4,
                        help="Number of future actions for v9 latent rollout.")
    parser.add_argument("--rollout_loss_mode", type=str, default="smooth_l1",
                        choices=["smooth_l1", "smooth-l1", "smoothl1", "huber",
                                 "infonce", "info_nce", "contrastive"],
                        help="v9 rollout loss: Smooth L1 for v9A or InfoNCE for v9B.")
    parser.add_argument("--horizon_weight_gamma", type=float, default=0.8,
                        help="Geometric per-step rollout weight gamma for v9.")
    normalize_latents_group = parser.add_mutually_exclusive_group()
    normalize_latents_group.add_argument("--normalize_latents_for_loss",
                                         action="store_true", default=None,
                                         help="Normalize v9 latents before regression loss.")
    normalize_latents_group.add_argument("--no_normalize_latents_for_loss",
                                         dest="normalize_latents_for_loss",
                                         action="store_false")
    parser.add_argument("--lambda_var", type=float, default=0.0,
                        help="v9A variance anti-collapse weight.")
    parser.add_argument("--lambda_cov", type=float, default=0.0,
                        help="v9A covariance anti-collapse weight.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="v9B InfoNCE temperature.")

    parser.add_argument("--probe_interval_steps", type=int, default=0)
    parser.add_argument("--probe_train_games", type=int, default=512)
    parser.add_argument("--probe_val_games", type=int, default=512)
    parser.add_argument("--probe_epochs", type=int, default=1)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--probe_learning_rate", type=float, default=1e-3)
    parser.add_argument("--probe_layers", type=str, default="")

    # Transposition-family InfoNCE (objective_class=family_infonce).
    parser.add_argument("--artifacts_dir", type=str, default="",
                        help="Self-contained family_artifacts directory (family_infonce only).")
    parser.add_argument("--local_stage_dir", type=str, default="",
                        help="Optional local staging copy of artifacts_dir before training.")
    parser.add_argument("--verify_manifest", dest="verify_manifest",
                        action="store_true", default=True)
    parser.add_argument("--no_verify_manifest", dest="verify_manifest",
                        action="store_false")
    parser.add_argument("--proj_hidden", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--families_per_batch", type=int, default=16)
    parser.add_argument("--members_per_family", type=int, default=16)
    parser.add_argument("--max_common_prefix_frac", type=float, default=0.0)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--retrieval_every_steps", type=int, default=500)
    parser.add_argument("--retrieval_anchors", type=int, default=512)
    parser.add_argument("--milestone_steps", type=str,
                        default="5000,10000,15000,20000")
    parser.add_argument("--drive_sync_every_steps", type=int, default=1000)

    # Optimisation
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--passes_over_data", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=0,
                        help="Override schedule denominator. 0 = auto.")
    parser.add_argument("--games_per_chunk", type=int, default=100_000,
                        help="Used when manifest.json is unavailable.")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--window_sample_attempts", type=int, default=8,
                        help="Random t attempts before declaring a batch has no valid K-window rows.")
    parser.add_argument(
        "--position_sampling",
        type=str,
        default="random",
        choices=["random", "all"],
        help="v1 boundary supervision: historical one-random-t or every valid t.",
    )
    parser.add_argument(
        "--all_position_target_chunk_size",
        type=int,
        default=4096,
        help="Maximum independent length-one targets encoded per target sub-batch.",
    )
    parser.add_argument(
        "--all_position_stats_chunk_size",
        type=int,
        default=8,
        help="Number of absolute positions per vectorized VICReg statistics block.",
    )

    # Run control
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_chunks", type=int, default=1,
                        help="How many val chunks to use per eval. 0 = no eval.")
    parser.add_argument("--eval_every_chunks", type=int, default=5,
                        help="Run eval every N training chunks.")
    parser.add_argument("--checkpoint_every_chunks", type=int, default=0,
                        help="Save checkpoint_chunk_NNNNNN.pt every N chunks. 0 = off.")
    parser.add_argument("--intermediate_checkpoint_fractions", type=str, default="",
                        help="Comma-separated fractions of total_steps for extra checkpoints.")
    parser.add_argument("--milestone_games", type=str,
                        default="1000000,5000000,10000000,15000000,20000000")
    parser.add_argument("--max_chunks", type=int, default=0,
                        help="Stop after N total chunks. 0 = no limit.")
    parser.add_argument(
        "--max_batches_per_chunk",
        type=int,
        default=0,
        help="Diagnostic cap on batches consumed from each chunk. 0 = full chunk.",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile_model", action="store_true",
                        help="Use torch.compile. Often slower on T4; try only after first run.")

    # Drive auto-sync
    parser.add_argument("--drive_sync_dir", type=str, default=None,
                        help="If set, copy local out_dir contents to this Drive "
                             "directory after every drive_sync_every_chunks chunks.")
    parser.add_argument("--drive_sync_every_chunks", type=int, default=1,
                        help="How often to sync to Drive (default 1 = every chunk).")

    # WandB
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    return parser

def parse_configured_args(
    argv: list[str],
) -> tuple[argparse.Namespace, Path | None, dict[str, Any]]:
    """Resolve canonical YAML values followed by explicit CLI overrides."""
    launcher = argparse.ArgumentParser(add_help=False)
    launcher.add_argument("--config", type=str, default=None)
    launcher.add_argument(
        "--print_resolved_config",
        "--print-resolved-config",
        action="store_true",
    )
    launcher_args, remaining = launcher.parse_known_args(argv)

    parser = build_parser()
    config_path: Path | None = None
    config_data: dict[str, Any] = {}
    config_cli_args: list[str] = []
    if launcher_args.config is not None:
        config_path = Path(launcher_args.config).expanduser()
        config_data = load_yaml_config(config_path)
        config_cli_args = config_to_cli_args(config_data, parser)

    args = parser.parse_args([*config_cli_args, *remaining])
    args.config = str(config_path) if config_path is not None else None
    args.print_resolved_config = launcher_args.print_resolved_config
    return args, config_path, config_data


def namespace_to_train_config(
    args: argparse.Namespace,
) -> tuple[TrainConfig, bool]:
    """Build the schema-compatible TrainConfig and lock objective identity."""
    values = vars(args).copy()
    print_resolved_config = bool(values.pop("print_resolved_config", False))
    values.pop("config", None)
    candidate = TrainConfig(**values)
    objective_class(candidate)
    normalize_jepa_variant(candidate.variant)
    candidate.loss_type = normalize_jepa_loss_type(
        candidate.loss_type,
        candidate.variant,
    )
    candidate.view_mode = normalize_jepa_view_mode(candidate.view_mode)
    return candidate, print_resolved_config


def validate_train_config(cfg: TrainConfig) -> None:
    """Reject any departure from the final-v5 objective topology."""
    objective_class(cfg)
    if normalize_jepa_variant(cfg.variant) != "jepa_v1":
        raise ValueError("The final JEPA protocol requires variant='v1'.")
    if normalize_jepa_loss_type(cfg.loss_type, cfg.variant) != "infonce":
        raise ValueError("The final JEPA protocol requires loss_type='infonce'.")
    if normalize_jepa_view_mode(cfg.view_mode) != "hard_disjoint_future":
        raise ValueError(
            "The final JEPA protocol requires "
            "view_mode='hard_disjoint_future'."
        )
    if cfg.predictor_type != "linear":
        raise ValueError("The final JEPA protocol requires predictor_type='linear'.")
    if cfg.prediction_horizon != 1:
        raise ValueError("The final JEPA protocol requires prediction_horizon=1.")
    if cfg.position_sampling != "all":
        raise ValueError("The final JEPA protocol requires position_sampling='all'.")
    if cfg.use_ema_target is not True:
        raise ValueError("The final JEPA protocol requires use_ema_target=True.")
    if cfg.contrastive_loss != "infonce":
        raise ValueError("The final JEPA protocol requires contrastive_loss='infonce'.")
    if not cfg.contrastive_use_prefix_mask:
        raise ValueError("The canonical contrastive prefix mask must remain enabled.")
    if not (0.0 <= cfg.ema_momentum < 1.0):
        raise ValueError("ema_momentum must be in [0, 1).")
    if cfg.contrastive_temperature <= 0:
        raise ValueError("contrastive_temperature must be positive.")
    if cfg.board_size not in {8, 12, 16}:
        raise ValueError("board_size must be one of 8, 12, or 16.")
    if cfg.max_batches_per_chunk < 0:
        raise ValueError("max_batches_per_chunk must be non-negative")
    if cfg.all_position_target_chunk_size < 1:
        raise ValueError("all_position_target_chunk_size must be positive")
    if cfg.all_position_stats_chunk_size < 1:
        raise ValueError("all_position_stats_chunk_size must be positive")
    if cfg.variance_eps <= 0:
        raise ValueError("variance_eps must be positive")
