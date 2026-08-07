"""Registry resolution and checkpoint-identity contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from othello_thesis.evaluation.unified import UnifiedEvaluator

ARCHITECTURES = ("transformer", "mamba")
OBJECTIVES = ("ar", "jepa")
BOARD_SIZES = (8, 12, 16)
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "Main_Experimental_Setup"
    / "configs"
    / "thesis_evaluation_registry.yml"
)


class ModelNotReadyError(FileNotFoundError):
    """Raised when a registered experiment has no completed final checkpoint."""


@dataclass(frozen=True)
class EvaluationCase:
    architecture: str
    objective: str
    board_size: int
    data_relative: str
    run_candidates: tuple[str, ...]
    canonical_jepa: Mapping[str, Any]

    @property
    def key(self) -> str:
        return f"{self.architecture}_{self.objective}_b{self.board_size}"

    @property
    def label(self) -> str:
        return (
            f"{self.architecture.title()}-{self.objective.upper()} "
            f"{self.board_size}x{self.board_size}"
        )


@dataclass
class PreparedEvaluation:
    case: EvaluationCase
    artifacts_root: Path
    data_dir: Path
    run_dir: Path
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    evaluator: UnifiedEvaluator
    primary_model: torch.nn.Module
    output_dir: Path
    training_metrics: dict[str, Any] | None


def normalize_architecture(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "transformer": "transformer",
        "gpt": "transformer",
        "mamba": "mamba",
        "ssm": "mamba",
    }
    if normalized not in aliases:
        raise ValueError(
            f"architecture must be one of {ARCHITECTURES}, got {value!r}"
        )
    return aliases[normalized]


def normalize_objective(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "ar": "ar",
        "autoregressive": "ar",
        "jepa": "jepa",
    }
    if normalized not in aliases:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {value!r}")
    return aliases[normalized]


def normalize_board_size(value: int | str) -> int:
    board_size = int(value)
    if board_size not in BOARD_SIZES:
        raise ValueError(f"board_size must be one of {BOARD_SIZES}, got {value!r}")
    return board_size


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "thesis_eval_registry_v1":
        raise ValueError(f"Unsupported registry schema in {registry_path}")
    return payload


def resolve_case(
    architecture: str,
    objective: str,
    board_size: int | str,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> EvaluationCase:
    architecture = normalize_architecture(architecture)
    objective = normalize_objective(objective)
    board_size = normalize_board_size(board_size)
    registry = load_registry(registry_path)
    data_relative = registry["data"][board_size]
    candidates = registry["cases"][architecture][objective][board_size]
    return EvaluationCase(
        architecture=architecture,
        objective=objective,
        board_size=board_size,
        data_relative=str(data_relative),
        run_candidates=tuple(str(item) for item in candidates),
        canonical_jepa=dict(registry["canonical_jepa"]),
    )


def _nested_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _read_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    """Read a lightweight training config without opening a large checkpoint."""
    for name in ("source_config.yml", "resolved_train_config.yml"):
        path = run_dir / name
        if path.is_file():
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                return dict(payload)
    path = run_dir / "train_config.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return dict(payload)
    return None


def _architecture_from_metadata(
    payload: Mapping[str, Any],
    run_dir: Path,
) -> str:
    model = _nested_mapping(payload, "model")
    candidates = (
        model.get("encoder_architecture"),
        model.get("architecture"),
        payload.get("encoder_architecture"),
        payload.get("architecture"),
    )
    for value in candidates:
        if value is None:
            continue
        text = str(value).lower()
        if "mamba" in text:
            return "mamba"
        if text in {"transformer", "gpt"}:
            return "transformer"
    path_text = run_dir.as_posix().lower()
    return "mamba" if "mamba" in path_text else "transformer"


def _metadata_matches_case(
    payload: Mapping[str, Any],
    run_dir: Path,
    case: EvaluationCase,
) -> bool:
    data = _nested_mapping(payload, "data")
    objective = _nested_mapping(payload, "objective")
    board_value = data.get("board_size", payload.get("board_size"))
    if board_value is None or int(board_value) != case.board_size:
        return False
    if _architecture_from_metadata(payload, run_dir) != case.architecture:
        return False
    if case.objective == "ar":
        return not objective and not payload.get("objective_class")
    expected = case.canonical_jepa
    actual = objective or dict(payload)
    return (
        str(actual.get("objective_class", "jepa")).lower() == "jepa"
        and str(actual.get("loss_type", "")).lower() == expected["loss_type"]
        and str(actual.get("view_mode", "")).lower() == expected["view_mode"]
        and int(actual.get("prediction_horizon", -1))
        == int(expected["prediction_horizon"])
        and str(actual.get("position_sampling", "")).lower()
        == expected["position_sampling"]
        and bool(actual.get("use_ema_target", False))
        == bool(expected["use_ema_target"])
        and math.isclose(
            float(actual.get("ema_momentum", -1.0)),
            float(expected["ema_momentum"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def resolve_run_dir(
    case: EvaluationCase,
    artifacts_root: str | Path,
) -> Path:
    artifacts_root = Path(artifacts_root)
    for relative in case.run_candidates:
        run_dir = artifacts_root / relative
        if (run_dir / "final.pt").is_file():
            return run_dir

    runs_root = artifacts_root / "runs"
    discovered: list[Path] = []
    if runs_root.is_dir():
        for final_path in runs_root.rglob("final.pt"):
            run_dir = final_path.parent
            lowered_parts = {part.lower() for part in run_dir.parts}
            if (
                "unified_eval" in lowered_parts
                or "thesis_eval" in lowered_parts
                or "smoke_eval" in lowered_parts
                or "smoke" in run_dir.name.lower()
            ):
                continue
            payload = _read_run_metadata(run_dir)
            if payload and _metadata_matches_case(payload, run_dir, case):
                discovered.append(run_dir)
    unique = sorted(set(discovered), key=lambda path: path.as_posix())
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        formatted = "\n  - ".join(str(path) for path in unique)
        raise RuntimeError(
            f"Multiple completed runs match {case.label}; registry selection is "
            f"required:\n  - {formatted}"
        )

    canonical = artifacts_root / case.run_candidates[0]
    raise ModelNotReadyError(
        f"{case.label} is registered but not trained yet. Expected the exact "
        f"headline checkpoint at:\n  {canonical / 'final.pt'}\n"
        "Training may use another run name; once final.pt and its training "
        "metadata exist, the resolver can discover it automatically."
    )


def resolve_data_dir(case: EvaluationCase, artifacts_root: str | Path) -> Path:
    data_dir = Path(artifacts_root) / case.data_relative
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Corpus for {case.label} is missing: {data_dir}"
        )
    return data_dir


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return dict(
        checkpoint.get("model_config")
        or checkpoint.get("jepa_config")
        or {}
    )


def _train_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return dict(checkpoint.get("train_config") or {})


def _checkpoint_architecture(
    model_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> str:
    values = (
        model_config.get("encoder_architecture"),
        model_config.get("architecture"),
        train_config.get("encoder_architecture"),
        train_config.get("architecture"),
    )
    for value in values:
        if value is None:
            continue
        lowered = str(value).lower()
        if "mamba" in lowered:
            return "mamba"
        if lowered in {"transformer", "gpt"}:
            return "transformer"
    return "transformer"


def validate_checkpoint_identity(
    case: EvaluationCase,
    checkpoint: Mapping[str, Any],
) -> None:
    model_config = _checkpoint_config(checkpoint)
    train_config = _train_config(checkpoint)
    board_size = int(
        model_config.get(
            "board_size",
            train_config.get("board_size", -1),
        )
    )
    if board_size != case.board_size:
        raise ValueError(
            f"Checkpoint board_size={board_size}, expected {case.board_size}"
        )
    architecture = _checkpoint_architecture(model_config, train_config)
    if architecture != case.architecture:
        raise ValueError(
            f"Checkpoint architecture={architecture}, expected {case.architecture}"
        )

    if case.objective == "ar":
        objective_class = str(
            model_config.get(
                "objective_class",
                train_config.get("objective_class", "ar"),
            )
        ).lower()
        if objective_class not in {"", "ar", "autoregressive", "none"}:
            raise ValueError(
                f"Checkpoint objective_class={objective_class!r}, expected AR"
            )
        return

    expected = case.canonical_jepa
    objective_class = str(
        model_config.get(
            "objective_class",
            train_config.get("objective_class", "jepa"),
        )
    ).lower().replace("-", "_")
    checks = {
        "objective_class": (objective_class, expected["objective_class"]),
        "loss_type": (
            str(model_config.get("loss_type", "")).lower(),
            expected["loss_type"],
        ),
        "view_mode": (
            str(model_config.get("view_mode", "")).lower(),
            expected["view_mode"],
        ),
        "prediction_horizon": (
            int(model_config.get("prediction_horizon", -1)),
            int(expected["prediction_horizon"]),
        ),
        "position_sampling": (
            str(train_config.get("position_sampling", "")).lower(),
            expected["position_sampling"],
        ),
        "use_ema_target": (
            bool(model_config.get("use_ema_target", False)),
            bool(expected["use_ema_target"]),
        ),
        "ema_momentum": (
            float(model_config.get("ema_momentum", -1.0)),
            float(expected["ema_momentum"]),
        ),
        "contrastive_temperature": (
            float(model_config.get("contrastive_temperature", -1.0)),
            float(expected["contrastive_temperature"]),
        ),
        "predictor_type": (
            str(model_config.get("predictor_type", "")).lower(),
            expected["predictor_type"],
        ),
    }
    mismatches = {
        name: {"actual": actual, "expected": wanted}
        for name, (actual, wanted) in checks.items()
        if actual != wanted
    }
    if mismatches:
        raise ValueError(
            "Checkpoint is not the canonical v5-style hard-disjoint InfoNCE "
            f"all-position JEPA condition: {mismatches}"
        )
