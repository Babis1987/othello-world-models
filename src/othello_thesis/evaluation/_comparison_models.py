"""Stable data contracts shared by comparison report versions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from othello_thesis.evaluation.thesis import (
    ARCHITECTURES,
    BOARD_SIZES,
    OBJECTIVES,
    EvaluationCase,
    normalize_architecture,
    normalize_board_size,
    normalize_objective,
)


@dataclass(frozen=True)
class ComparisonSelection:
    """Normalized selectors exposed by the comparison notebook."""

    architectures: tuple[str, ...]
    objectives: tuple[str, ...]
    board_sizes: tuple[int, ...]

    @classmethod
    def from_values(
        cls,
        architecture: str = "both",
        objective: str = "both",
        board_size: int | str = 12,
    ) -> "ComparisonSelection":
        architecture_text = str(architecture).strip().lower()
        objective_text = str(objective).strip().lower()
        board_text = str(board_size).strip().lower().replace("x", "")

        if architecture_text in {"both", "all"}:
            architectures = tuple(ARCHITECTURES)
        else:
            architectures = (normalize_architecture(architecture_text),)

        if objective_text in {"both", "all"}:
            objectives = tuple(OBJECTIVES)
        else:
            objectives = (normalize_objective(objective_text),)

        if board_text == "all":
            board_sizes = tuple(BOARD_SIZES)
        else:
            board_sizes = (normalize_board_size(int(board_text)),)

        return cls(
            architectures=architectures,
            objectives=objectives,
            board_sizes=board_sizes,
        )

    @property
    def expected_count(self) -> int:
        return (
            len(self.architectures)
            * len(self.objectives)
            * len(self.board_sizes)
        )

    @property
    def is_single_case(self) -> bool:
        return self.expected_count == 1

    @property
    def slug(self) -> str:
        architecture = (
            self.architectures[0]
            if len(self.architectures) == 1
            else "both_architectures"
        )
        objective = (
            self.objectives[0]
            if len(self.objectives) == 1
            else "both_objectives"
        )
        boards = (
            f"b{self.board_sizes[0]}"
            if len(self.board_sizes) == 1
            else "all_boards"
        )
        return f"{architecture}__{objective}__{boards}"

    @property
    def display(self) -> str:
        architecture = (
            self.architectures[0].title()
            if len(self.architectures) == 1
            else "Transformer + Mamba"
        )
        objective = (
            self.objectives[0].upper()
            if len(self.objectives) == 1
            else "AR + JEPA"
        )
        boards = (
            f"{self.board_sizes[0]}x{self.board_sizes[0]}"
            if len(self.board_sizes) == 1
            else "8x8 + 12x12 + 16x16"
        )
        return f"{architecture} | {objective} | {boards}"


@dataclass(frozen=True)
class ComparisonRun:
    """One validated common-evaluation cell and its normalized metrics."""

    case: EvaluationCase
    results_path: Path
    payload: Mapping[str, Any]
    metrics: Mapping[str, Any]

    @property
    def label(self) -> str:
        return self.case.label

    @property
    def short_label(self) -> str:
        architecture = "T" if self.case.architecture == "transformer" else "M"
        return f"{architecture}-{self.case.objective.upper()} {self.case.board_size}x{self.case.board_size}"

    @property
    def system_key(self) -> tuple[str, str]:
        return self.case.architecture, self.case.objective

    @property
    def head_kind(self) -> str:
        """``native`` for AR, ``frozen`` for JEPA readouts."""
        return str(self.metrics.get("legal_head_kind") or "unknown")


@dataclass(frozen=True)
class ComparisonReport:
    """Artifacts returned to the notebook frontend."""

    report_path: Path
    figure_paths: Mapping[str, Path]
    runs: tuple[ComparisonRun, ...]
    missing: Mapping[str, str]
    selection: ComparisonSelection
    export_paths: Mapping[str, Path]
    warnings: tuple[str, ...] = ()
