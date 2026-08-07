"""Internal implementation module extracted from the canonical JEPA trainer."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from othello_thesis.objectives.jepa import (
    normalize_jepa_loss_type,
    normalize_jepa_variant,
    normalize_jepa_view_mode,
)

@dataclass
class TrainConfig:
    out_dir: str
    train_dir: str | None = None
    val_dir: str | None = None
    data_dir: str | None = None
    pair_index_path: str | None = None
    li_split_train_chunks: int = 200
    objective_class: str = "jepa"
    # Public experiment label. Keep this as v1..v9; objective configs below
    # translate it to any internal topology name needed for model construction.
    variant: str = "v2"
    loss_type: str = "auto"
    contrastive_loss: str = "infonce"

    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    predictor_type: str = "transformer"
    predictor_hidden_dim: int | None = None
    predictor_hidden_mult: int = 4
    predictor_n_layers: int = 4
    predictor_n_heads: int = 8
    predictor_dropout: float | None = None
    predictor_activation: str = "relu"

    # v1 VICReg objective.
    vicreg_lambda: float = 25.0
    vicreg_mu: float = 25.0
    vicreg_nu: float = 1.0
    variance_threshold: float = 1.0
    variance_eps: float = 1e-4

    # v2 EMA Smooth L1 objective.
    view_mode: str = "nested"
    prediction_horizon: int = 4
    ema_momentum: float = 0.996
    smooth_l1_beta: float = 1.0
    use_ema_target: bool | None = None
    log_collapse_stats_every_steps: int = 0
    collapse_warning_z_std: float = 0.1
    collapse_warning_cos_sim: float = 0.95
    collapse_warning_contrastive_accuracy: float = 0.05

    # v5 contrastive objective.
    contrastive_temperature: float = 0.1
    contrastive_use_prefix_mask: bool = True

    # v6 hard-disjoint action JEPA. All defaults below leave v1-v5 paths
    # untouched because they are only consulted when objective_class is
    # "jepa_hard_disjoint_action".
    hard_disjoint_num_modes: int = 10
    hard_disjoint_margin: float = 1.0
    hard_disjoint_lambda_push: float = 1.0
    hard_disjoint_normalize: bool = True
    hard_disjoint_dedup_negative_actions: bool = True
    branch_t_min: int = 4
    branch_t_max: int = 10
    branch_groups_per_batch: int = 64
    branch_samples_per_group: int = 4

    # v7 action-conditioned JEPA. These fields are consulted only when
    # objective_class is "jepa_action_conditioned".
    loss_mode: str | None = None
    action_loss_mode: str = "smooth_l1"
    hybrid_base: str = "grouped_infonce"
    action_dim: int | None = None
    normalize_embeddings: bool | None = None
    action_temperature: float = 0.1
    action_temperature_learnable: bool = False
    action_groups_per_batch: int = 8
    action_samples_per_group: int = 16
    action_chunk_sample_multiplier: float = 1.0
    action_t_min: int = 4
    action_t_max: int = 10
    lambda_ce: float = 0.05

    # v8 order-aware JEPA. The runtime path reads only the offline pair index;
    # it never invokes the Othello engine while constructing training batches.
    order_temperature: float = 0.1
    order_temperature_learnable: bool = False
    order_groups_per_batch: int = 8
    order_samples_per_group: int = 16
    order_hard_negatives_per_anchor: int = 1
    order_chunk_sample_multiplier: float = 1.0
    order_pair_chunk_cache_size: int = 4
    order_min_pair_anchors_per_group: int = 0
    order_index_use_constructive_pairs: bool = False
    order_index_use_surface_hard_negatives: bool = False
    order_surface_jaccard_threshold: float = 0.9
    order_surface_max_pairs_per_anchor: int = 1
    order_index_positions_per_game: int = 1
    order_index_sampling_seed: int = 42
    order_t_min: int = 4
    order_t_max: int = 40
    order_utility_early_stop_threshold: float = -1.0
    order_utility_early_stop_window: int = 500
    order_utility_early_stop_min_steps: int = 5000

    # v9 multi-action latent rollout JEPA. These fields are used only when
    # objective_class is "jepa_multiaction".
    action_horizon: int = 4
    rollout_loss_mode: str = "smooth_l1"
    horizon_weight_gamma: float = 0.8
    normalize_latents_for_loss: bool | None = None
    lambda_var: float = 0.0
    lambda_cov: float = 0.0
    temperature: float = 0.1

    # Optional fixed-set board probes during v7/v8 training.
    probe_interval_steps: int = 0
    probe_train_games: int = 512
    probe_val_games: int = 512
    probe_epochs: int = 1
    probe_batch_size: int = 64
    probe_learning_rate: float = 1e-3
    probe_layers: str = ""

    # Transposition-family InfoNCE (objective_class=family_infonce). Training
    # for this objective is step-based over prebuilt family_artifacts and is
    # delegated from main() to train_family_infonce; the chunk loop is unused.
    artifacts_dir: str = ""
    local_stage_dir: str = ""
    verify_manifest: bool = True
    proj_hidden: int = 512
    proj_dim: int = 128
    families_per_batch: int = 16
    members_per_family: int = 16
    max_common_prefix_frac: float = 0.0
    log_every_steps: int = 100
    retrieval_every_steps: int = 500
    retrieval_anchors: int = 512
    milestone_steps: str = "5000,10000,15000,20000"
    drive_sync_every_steps: int = 1000

    batch_size: int = 512
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    passes_over_data: int = 1
    total_steps: int = 0
    games_per_chunk: int = 100_000
    precision: str = "bf16"
    window_sample_attempts: int = 8
    # v1 supervision density. "random" preserves the historical protocol
    # (one shared random boundary per batch); "all" uses every valid boundary.
    position_sampling: str = "random"
    all_position_target_chunk_size: int = 4096
    all_position_stats_chunk_size: int = 8

    num_workers: int = 0
    seed: int = 42
    eval_chunks: int = 1
    eval_every_chunks: int = 5
    checkpoint_every_chunks: int = 0
    intermediate_checkpoint_fractions: str = ""
    milestone_games: str = "1000000,5000000,10000000,15000000,20000000"
    max_chunks: int = 0
    # Diagnostic-only cap used by quick smoke runs. Zero preserves the full
    # shard behaviour used by every thesis training configuration.
    max_batches_per_chunk: int = 0
    resume: str | None = None
    compile_model: bool = False

    # Drive auto-sync (for Colab overnight runs).
    drive_sync_dir: str | None = None
    drive_sync_every_chunks: int = 1

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"

CONFIG_METADATA_KEYS = {
    "commit",
    "description",
    "git_commit",
    "github_repo",
    "name",
    "notes",
    "objective_type",
    "pair_index_build_workers",
    "family_build_workers",
    "family_t_min",
    "family_t_max",
    "tags",
    "type",
}

def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML experiment config as a mapping."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a YAML mapping, got {type(data).__name__}")
    return data

def flatten_config_leaves(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config sections to argparse leaf names."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_config_leaves(value, path))
            continue
        leaf = str(key)
        if leaf in flat:
            raise ValueError(
                f"Duplicate config leaf `{leaf}` found at `{path}`. "
                "Use unique argparse-style leaf names."
            )
        flat[leaf] = value
    return flat

def parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    """Return argparse destinations accepted by the training parser."""
    return {
        action.dest
        for action in parser._actions  # argparse has no public iterator.
        if action.dest != "help"
    }

def config_to_cli_args(data: dict[str, Any], parser: argparse.ArgumentParser) -> list[str]:
    """Convert a YAML config to CLI args for the unified trainer."""
    accepted = parser_destinations(parser)
    option_strings = {
        option
        for action in parser._actions  # argparse has no public iterator.
        for option in action.option_strings
    }
    args: list[str] = []
    unknown: list[str] = []

    for key, value in flatten_config_leaves(data).items():
        if key in CONFIG_METADATA_KEYS:
            continue
        if key not in accepted:
            unknown.append(key)
            continue
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                no_flag = f"--no_{key}"
                if no_flag in option_strings:
                    args.append(no_flag)
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(x) for x in value)
        args.extend([flag, str(value)])

    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"Config contains keys that train_jepa.py does not accept: {names}. "
            "Use argparse field names or move notes under metadata keys."
        )
    return args

def write_config_records(
    *,
    source_config: Path,
    source_data: dict[str, Any],
    resolved_args: dict[str, Any],
) -> None:
    """Copy the source config and resolved values into the output directory."""
    out_dir = Path(str(resolved_args["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    source_copy = out_dir / "source_config.yml"
    if source_config.resolve() != source_copy.resolve():
        shutil.copy2(source_config, source_copy)

    record = {
        "source_config": str(source_config),
        "source": source_data,
        "resolved_args": resolved_args,
    }
    with open(out_dir / "resolved_train_config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False)

def jepa_variant(cfg: TrainConfig) -> str:
    """Return the internal predictive JEPA target topology."""
    return normalize_jepa_variant(cfg.variant)

def jepa_loss_type(cfg: TrainConfig) -> str:
    """Return the normalized configured JEPA loss."""
    return normalize_jepa_loss_type(cfg.loss_type, cfg.variant)

def normalize_contrastive_variant(value: str) -> str:
    """Normalize contrastive experiment aliases without predictive JEPA rules."""
    normalized = value.lower().replace("-", "_")
    aliases = {
        "jepa_v5_contrastive": "jepa_v5_contrastive",
        "jepa_v5": "jepa_v5_contrastive",
        "v5": "jepa_v5_contrastive",
        "contrastive": "jepa_v5_contrastive",
        "cpc": "jepa_v5_contrastive",
        "contrastive_disjoint_k1": "jepa_v5_contrastive",
        "cpc_disjoint_future": "jepa_v5_contrastive",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown contrastive variant {value!r}. Valid values: {valid}")
    return aliases[normalized]

def normalize_experiment_variant(value: str, objective_class_name: str) -> str:
    """Normalize public experiment labels to the canonical ``v1``..``v6`` set.

    ``variant`` is the thesis experiment family shown in configs and logs.
    ``objective_class`` selects the implementation module, while objective
    config builders translate the public label to legacy internal topology
    names where checkpoint reconstruction requires them.
    """
    normalized = value.lower().replace("-", "_")
    aliases = {
        "v1": "v1",
        "jepa_v1": "v1",
        "vicreg": "v1",
        "jepa_vicreg": "v1",
        "v2": "v2",
        "jepa_v2": "v2",
        "ema": "v2",
        "ema_smooth_l1": "v2",
        "jepa_ema": "v2",
        "v3": "v3",
        "jepa_v3": "v3",
        "v4": "v4",
        "jepa_v4": "v4",
        "v5": "v5",
        "jepa_v5": "v5",
        "jepa_v5_contrastive": "v5",
        "contrastive": "v5",
        "cpc": "v5",
        "contrastive_disjoint_k1": "v5",
        "cpc_disjoint_future": "v5",
        "v5_hard_disjoint": "v5",
        "v5_hard_disjoint_infonce": "v5",
        "jepa_v5_hard_disjoint_infonce": "v5",
        "v6": "v6",
        "jepa_v6": "v6",
        "hard_disjoint_action": "v6",
        "jepa_v6_hard_disjoint_action": "v6",
        "v7": "v7",
        "jepa_v7": "v7",
        "action_conditioned": "v7",
        "jepa_action_conditioned": "v7",
        "jepa_v7_action_conditioned": "v7",
        "v8": "v8",
        "jepa_v8": "v8",
        "order_aware": "v8",
        "jepa_order_aware": "v8",
        "jepa_v8_order_aware": "v8",
        "v9": "v9",
        "jepa_v9": "v9",
        "multiaction": "v9",
        "multi_action": "v9",
        "jepa_multiaction": "v9",
        "jepa_multi_action": "v9",
        "jepa_v9_multiaction": "v9",
        "v9a": "v9",
        "jepa_v9a": "v9",
        "v9b": "v9",
        "jepa_v9b": "v9",
        "family_infonce": "family_infonce",
        "family": "family_infonce",
        "transposition_family": "family_infonce",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown experiment variant {value!r}. Valid values: {valid}")

    canonical = aliases[normalized]
    allowed = {
        "jepa": {"v1", "v2", "v3", "v4"},
        "jepa_contrastive": {"v5"},
        "jepa_hard_disjoint_infonce": {"v5"},
        "jepa_hard_disjoint_action": {"v6"},
        "jepa_action_conditioned": {"v7"},
        "jepa_order_aware": {"v8"},
        "jepa_multiaction": {"v9"},
        "family_infonce": {"family_infonce"},
    }
    if objective_class_name not in allowed:
        raise ValueError(f"Unknown objective_class for variant validation: {objective_class_name!r}")
    if canonical not in allowed[objective_class_name]:
        expected = ", ".join(sorted(allowed[objective_class_name]))
        raise ValueError(
            f"variant={canonical!r} is incompatible with "
            f"objective_class={objective_class_name!r}; expected one of: {expected}"
        )
    return canonical

def normalize_contrastive_loss_type(value: str) -> str:
    """Normalize contrastive loss aliases."""
    normalized = value.lower().replace("-", "_")
    aliases = {
        "auto": "infonce",
        "infonce": "infonce",
        "info_nce": "infonce",
        "contrastive": "infonce",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown contrastive loss_type {value!r}. Valid values: {valid}")
    return aliases[normalized]

def normalize_multiaction_loss_mode(value: str) -> str:
    """Normalize v9 rollout loss aliases."""
    normalized = value.lower().replace("-", "_")
    aliases = {
        "auto": "smooth_l1",
        "smooth_l1": "smooth_l1",
        "smoothl1": "smooth_l1",
        "huber": "smooth_l1",
        "infonce": "infonce",
        "info_nce": "infonce",
        "contrastive": "infonce",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown v9 rollout_loss_mode {value!r}. Valid values: {valid}")
    return aliases[normalized]

def jepa_view_mode(cfg: TrainConfig) -> str:
    """Return the normalized configured JEPA view mode."""
    return normalize_jepa_view_mode(cfg.view_mode)

def objective_class(cfg: TrainConfig) -> str:
    """Return the normalized objective implementation name."""
    normalized = cfg.objective_class.lower().replace("-", "_")
    aliases = {
        "jepa": "jepa",
        "predictive": "jepa",
        "jepa_predictive": "jepa",
        "jepa_contrastive": "jepa_contrastive",
        "contrastive": "jepa_contrastive",
        "v5": "jepa_contrastive",
        "jepa_hard_disjoint_infonce": "jepa_hard_disjoint_infonce",
        "hard_disjoint_infonce": "jepa_hard_disjoint_infonce",
        "v5_hard_disjoint": "jepa_hard_disjoint_infonce",
        "v5_hard_disjoint_infonce": "jepa_hard_disjoint_infonce",
        "jepa_hard_disjoint_action": "jepa_hard_disjoint_action",
        "hard_disjoint_action": "jepa_hard_disjoint_action",
        "v6": "jepa_hard_disjoint_action",
        "jepa_action_conditioned": "jepa_action_conditioned",
        "action_conditioned": "jepa_action_conditioned",
        "v7": "jepa_action_conditioned",
        "jepa_order_aware": "jepa_order_aware",
        "order_aware": "jepa_order_aware",
        "v8": "jepa_order_aware",
        "jepa_multiaction": "jepa_multiaction",
        "jepa_multi_action": "jepa_multiaction",
        "multiaction": "jepa_multiaction",
        "multi_action": "jepa_multiaction",
        "jepa_v9_multiaction": "jepa_multiaction",
        "v9": "jepa_multiaction",
        "v9a": "jepa_multiaction",
        "v9b": "jepa_multiaction",
        "family_infonce": "family_infonce",
        "family": "family_infonce",
        "transposition_family": "family_infonce",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown objective_class {cfg.objective_class!r}. Valid values: {valid}")
    return aliases[normalized]

def objective_horizon(cfg: TrainConfig) -> int:
    """Return how many future tokens must be valid for this objective."""
    if objective_class(cfg) == "jepa_multiaction":
        return cfg.action_horizon
    return cfg.prediction_horizon
