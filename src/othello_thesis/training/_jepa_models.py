"""Canonical JEPA objective construction.

The final thesis path intentionally supports only the winning all-position
hard-disjoint InfoNCE objective represented by the generic ``OthelloJEPA``
wrapper.  Architecture-specific fields live in ``JEPAConfig``; consequently
Transformer and Mamba construction share this single, auditable path.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Callable

import torch

from othello_thesis.objectives.jepa import (
    JEPAConfig,
    OthelloJEPA,
    normalize_jepa_loss_type,
    normalize_jepa_variant,
)
from othello_thesis.training._jepa_config import TrainConfig, objective_class


ModelFactory = Callable[[JEPAConfig], torch.nn.Module]


def filter_kwargs_for_dataclass(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields accepted by ``cls`` without changing their order."""
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


def _require_canonical_objective(cfg: TrainConfig) -> None:
    resolved = objective_class(cfg)
    if resolved != "jepa":
        raise ValueError(
            "Master_Thesis_Code_Final trains only the canonical "
            "objective_class='jepa' path; "
            f"got {resolved!r}. Historical JEPA variants live under "
            "JEPA_Experimentation."
        )


def build_objective_config(cfg: TrainConfig) -> JEPAConfig:
    """Resolve ``TrainConfig`` into the canonical objective dataclass."""
    _require_canonical_objective(cfg)
    values = asdict(cfg)
    values["variant"] = normalize_jepa_variant(cfg.variant)
    values["loss_type"] = normalize_jepa_loss_type(cfg.loss_type, cfg.variant)
    return JEPAConfig(**filter_kwargs_for_dataclass(JEPAConfig, values))


def build_objective_model(cfg: TrainConfig) -> tuple[torch.nn.Module, JEPAConfig]:
    """Instantiate the canonical JEPA model exactly once from ``cfg``."""
    objective_cfg = build_objective_config(cfg)
    return OthelloJEPA(objective_cfg), objective_cfg


def build_model_from_objective_config(config: JEPAConfig) -> torch.nn.Module:
    """Default factory used by the CLI and programmatic training API."""
    return OthelloJEPA(config)


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> tuple[torch.nn.Module, JEPAConfig]:
    """Instantiate a canonical JEPA model from checkpoint metadata."""
    config_data = ckpt.get("model_config") or ckpt.get("jepa_config")
    if config_data is None:
        raise KeyError("Checkpoint has neither model_config nor jepa_config.")
    config_data = dict(config_data)
    cls_name = config_data.pop("objective_class", None)
    if cls_name is None:
        train_config = ckpt.get("train_config") or {}
        cls_name = train_config.get("objective_class", "jepa")
    cls_name = str(cls_name).lower().replace("-", "_")
    if cls_name != "jepa":
        raise ValueError(
            "Master_Thesis_Code_Final loads only canonical JEPA checkpoints; "
            f"got objective_class={cls_name!r}."
        )
    objective_cfg = JEPAConfig(
        **filter_kwargs_for_dataclass(JEPAConfig, config_data)
    )
    return OthelloJEPA(objective_cfg), objective_cfg
