"""Resolve the locked 2 x 2 x 3 training grid for notebooks.

This module selects paths and constructs dataclass configurations. It never
constructs a model, seeds an RNG, or starts training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from othello_thesis.training.mamba_ar import MambaARTrainConfig
from othello_thesis.training.transformer_ar import TrainConfig as ARTrainConfig


Architecture = Literal["transformer", "mamba"]
Objective = Literal["ar", "jepa"]
RunProfile = Literal["smoke", "full"]
BOARD_SIZES = (8, 12, 16)
DEFAULT_PROJECT_ROOT = Path(
    "/content/drive/Othercomputers/MyLaptop/Master_Thesis_Code_Final"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "/content/drive/MyDrive/Master_Thesis_Artifacts"
)
DEFAULT_LOCAL_ROOT = Path("/content")


@dataclass(frozen=True)
class RunLayout:
    """Canonical data, local-output, and synced-output locations."""

    data_dir: Path
    local_out_dir: Path
    drive_out_dir: Path
    run_name: str
    registry_relative_path: Path


def canonical_config_dir(project_root: Path | None = None) -> Path:
    """Return the checked-in canonical configuration directory."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    return project_root / "Main_Experimental_Setup" / "configs"


def _load_yaml(name: str, project_root: Path | None = None) -> dict:
    path = canonical_config_dir(project_root) / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _validate_identity(
    architecture: str,
    objective: str,
    board_size: int,
) -> None:
    if architecture not in {"transformer", "mamba"}:
        raise ValueError(f"Unknown architecture: {architecture!r}")
    if objective not in {"ar", "jepa"}:
        raise ValueError(f"Unknown objective: {objective!r}")
    if board_size not in BOARD_SIZES:
        raise ValueError(f"board_size must be one of {BOARD_SIZES}, got {board_size}")


def resolve_run_layout(
    architecture: Architecture,
    objective: Objective,
    board_size: int,
    *,
    profile: RunProfile = "full",
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    local_root: Path = DEFAULT_LOCAL_ROOT,
    project_root: Path | None = None,
) -> RunLayout:
    """Resolve the primary path from the canonical evaluation registry."""
    _validate_identity(architecture, objective, board_size)
    if profile not in {"smoke", "full"}:
        raise ValueError(f"profile must be 'smoke' or 'full', got {profile!r}")
    registry = _load_yaml("thesis_evaluation_registry.yml", project_root)
    protocol = _load_yaml("canonical_training_protocol.yml", project_root)
    relative = Path(registry["cases"][architecture][objective][board_size][0])
    data_relative = Path(protocol["data"][board_size])
    suffix = "_smoke" if profile == "smoke" else ""
    run_name = relative.name + suffix
    selected_relative = relative.with_name(run_name)
    return RunLayout(
        data_dir=artifact_root / data_relative,
        local_out_dir=local_root / selected_relative,
        drive_out_dir=artifact_root / selected_relative,
        run_name=run_name,
        registry_relative_path=selected_relative,
    )


def _profile_overrides(profile: RunProfile) -> dict:
    if profile == "full":
        return {}
    return {
        "max_chunks": 1,
        "eval_chunks": 0,
        "milestone_games": "",
        "wandb_mode": "disabled",
    }


def build_ar_train_config(
    architecture: Architecture,
    board_size: int,
    layout: RunLayout,
    *,
    profile: RunProfile = "full",
    seed: int = 42,
    resume: str | None = None,
    project_root: Path | None = None,
) -> ARTrainConfig | MambaARTrainConfig:
    """Build the exact canonical AR dataclass without constructing a model."""
    _validate_identity(architecture, "ar", board_size)
    protocol = _load_yaml("canonical_training_protocol.yml", project_root)
    case = protocol["cases"][architecture]["ar"]
    values = {
        "out_dir": str(layout.local_out_dir),
        "data_dir": str(layout.data_dir),
        "drive_sync_dir": str(layout.drive_out_dir),
        "board_size": board_size,
        "seed": seed,
        "resume": resume,
        "checkpoint_every_chunks": case["boards"][board_size][
            "checkpoint_every_chunks"
        ],
        **case["model"],
        **protocol["ar_common"],
        **_profile_overrides(profile),
    }
    config_type = ARTrainConfig if architecture == "transformer" else MambaARTrainConfig
    return config_type(**values)


def _jepa_config_name(architecture: Architecture, board_size: int) -> str:
    if architecture == "transformer":
        return f"jepa_v5_infonce_b{board_size}_hd_all_positions.yml"
    return f"mamba_jepa_v5_hd_infonce_b{board_size}_all_positions.yml"


def build_jepa_train_config(
    architecture: Architecture,
    board_size: int,
    layout: RunLayout,
    *,
    profile: RunProfile = "full",
    seed: int = 42,
    resume: str | None = None,
    project_root: Path | None = None,
):
    """Resolve a locked JEPA YAML plus explicit runtime path overrides."""
    _validate_identity(architecture, "jepa", board_size)
    if architecture == "transformer":
        from othello_thesis.training import jepa as trainer
    else:
        from othello_thesis.training import mamba_jepa as trainer

    config_path = canonical_config_dir(project_root) / _jepa_config_name(
        architecture,
        board_size,
    )
    arguments = [
        "--config",
        str(config_path),
        "--out_dir",
        str(layout.local_out_dir),
        "--data_dir",
        str(layout.data_dir),
        "--drive_sync_dir",
        str(layout.drive_out_dir),
        "--board_size",
        str(board_size),
        "--seed",
        str(seed),
    ]
    if resume is not None:
        arguments.extend(("--resume", resume))
    if profile == "smoke":
        arguments.extend(
            (
                "--max_chunks",
                "1",
                "--max_batches_per_chunk",
                "1",
                "--eval_chunks",
                "0",
                "--milestone_games",
                "",
                "--intermediate_checkpoint_fractions",
                "",
                "--wandb_mode",
                "disabled",
            )
        )
    args, _, _ = trainer.parse_configured_args(arguments)
    config, _ = trainer.namespace_to_train_config(args)
    trainer.validate_train_config(config)
    return config


def validate_canonical_corpus(data_dir: Path, board_size: int) -> dict:
    """Reject a missing or incomplete canonical training corpus."""
    if board_size not in BOARD_SIZES:
        raise ValueError(f"board_size must be one of {BOARD_SIZES}, got {board_size}")
    manifest_path = data_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "shard_counts" in manifest:
            counts = manifest["shard_counts"]
            if not isinstance(counts, dict) or len(counts) < 201:
                raise ValueError(
                    "Legacy corpus manifest must contain at least 201 shard counts"
                )
            ordered_counts = [counts[name] for name in sorted(counts)]
            training_games = sum(ordered_counts[:200])
            if training_games < 19_900_000:
                raise ValueError(
                    "Legacy corpus has too few games in its first 200 shards: "
                    f"{training_games:,}"
                )
            return {
                **manifest,
                "board_size": board_size,
                "training_games": training_games,
                "legacy": True,
            }
        for key, expected in {
            "complete": True,
            "board_size": board_size,
            "training_games": 20_000_000,
        }.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Corpus manifest {key!r}: expected {expected!r}, "
                    f"got {manifest.get(key)!r}"
                )
        shards = manifest.get("shards", [])
        if len(shards) < 201:
            raise ValueError(f"Expected at least 201 shards, found {len(shards)}")
        return manifest

    chunks = sorted(
        [*data_dir.glob("games_*.pickle"), *data_dir.glob("gen10e5__*.pickle")]
    )
    if len(chunks) < 201:
        raise FileNotFoundError(
            f"No canonical manifest and only {len(chunks)} game chunks in {data_dir}"
        )
    return {"board_size": board_size, "shards": len(chunks), "legacy": True}
