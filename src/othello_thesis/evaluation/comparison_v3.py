"""Controlled cross-board reporting for Thesis Common Evaluation results.

Version 3 keeps the validated loading, metric normalization, head provenance,
tables, and exports from :mod:`comparison_v2`, but changes the analytical
unit.  Raw scores are ranked only *within* one board size.  Cross-board claims
are made from the trajectories of controlled within-size contrasts:

* Mamba - Transformer, separately inside AR and JEPA;
* JEPA - AR, separately inside Transformer and Mamba;
* factorial architecture/objective main effects and their interaction.

This prevents a global 8x8-vs-16x16 leaderboard from being mistaken for an
architecture or objective finding.  All effects remain descriptive because
the current grid contains one training seed per cell.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from othello_thesis.evaluation import comparison_v2 as v2
from othello_thesis.evaluation.thesis import (
    ARCHITECTURES,
    BOARD_SIZES,
    DEFAULT_REGISTRY,
    OBJECTIVES,
)
from othello_thesis.evaluation.unified import PROTOCOL_ID


COMPARISON_REPORT_VERSION = "thesis_comparison_v3"

ComparisonSelection = v2.ComparisonSelection
ComparisonRun = v2.ComparisonRun
ComparisonReport = v2.ComparisonReport
metric_glossary_markdown = v2.metric_glossary_markdown


CONTRAST_METRICS = (
    ("legal_preferred_top1", "Legal top-1 / best available head"),
    ("legal_preferred_mass", "Legal probability mass / best available head"),
    ("board_macro_mean", "Mean board-state macro accuracy"),
    ("board_linear_absolute", "Board Linear / absolute"),
    ("board_linear_relative", "Board Linear / relative"),
    ("board_mlp_absolute", "Board MLP / absolute"),
    ("board_mlp_relative", "Board MLP / relative"),
)

CORE_METRICS = CONTRAST_METRICS[:3]

CONTRAST_LABELS = {
    "architecture_ar": "Mamba - Transformer | AR",
    "architecture_jepa": "Mamba - Transformer | JEPA",
    "objective_transformer": "JEPA - AR | Transformer",
    "objective_mamba": "JEPA - AR | Mamba",
    "architecture_main": "Architecture main effect",
    "objective_main": "Objective main effect",
    "interaction": "Interaction (difference-in-differences)",
}

SIMPLE_CONTRASTS = (
    "architecture_ar",
    "architecture_jepa",
    "objective_transformer",
    "objective_mamba",
    "interaction",
)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise ValueError("Cannot average an empty sequence")
    return sum(materialized) / len(materialized)


def _cell_values(
    runs: Sequence[ComparisonRun],
    board_size: int,
    metric: str,
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for run in runs:
        if run.case.board_size != board_size:
            continue
        value = v2._finite_float(run.metrics.get(metric))
        if value is not None:
            values[(run.case.architecture, run.case.objective)] = value
    return values


def _difference(
    values: Mapping[tuple[str, str], float],
    positive: tuple[str, str],
    negative: tuple[str, str],
) -> float | None:
    if positive not in values or negative not in values:
        return None
    return values[positive] - values[negative]


def compute_controlled_contrasts(
    runs: Sequence[ComparisonRun],
) -> dict[int, dict[str, dict[str, float]]]:
    """Return every estimable controlled contrast independently per board.

    Partial notebook selections are supported: selecting only AR can still
    estimate ``architecture_ar``; the interaction appears only when all four
    cells for that board size and metric are complete.
    """

    result: dict[int, dict[str, dict[str, float]]] = {}
    for board_size in sorted({run.case.board_size for run in runs}):
        board_metrics: dict[str, dict[str, float]] = {}
        for metric, _ in CONTRAST_METRICS:
            values = _cell_values(runs, board_size, metric)
            contrasts: dict[str, float] = {}
            candidates = {
                "architecture_ar": _difference(
                    values, ("mamba", "ar"), ("transformer", "ar")
                ),
                "architecture_jepa": _difference(
                    values, ("mamba", "jepa"), ("transformer", "jepa")
                ),
                "objective_transformer": _difference(
                    values, ("transformer", "jepa"), ("transformer", "ar")
                ),
                "objective_mamba": _difference(
                    values, ("mamba", "jepa"), ("mamba", "ar")
                ),
            }
            contrasts.update(
                {
                    key: value
                    for key, value in candidates.items()
                    if value is not None
                }
            )
            architecture_parts = [
                value
                for key in ("architecture_ar", "architecture_jepa")
                if (value := candidates[key]) is not None
            ]
            objective_parts = [
                value
                for key in ("objective_transformer", "objective_mamba")
                if (value := candidates[key]) is not None
            ]
            if len(architecture_parts) == 2:
                contrasts["architecture_main"] = _mean(architecture_parts)
            if len(objective_parts) == 2:
                contrasts["objective_main"] = _mean(objective_parts)
            if (
                candidates["objective_transformer"] is not None
                and candidates["objective_mamba"] is not None
            ):
                contrasts["interaction"] = (
                    candidates["objective_mamba"]
                    - candidates["objective_transformer"]
                )
            if contrasts:
                board_metrics[metric] = contrasts
        if board_metrics:
            result[board_size] = board_metrics
    return result


def _boards(runs: Sequence[ComparisonRun]) -> list[int]:
    return sorted({run.case.board_size for run in runs})


def _within_board_runner_up(
    runs: Sequence[ComparisonRun], board_size: int, metric: str
) -> tuple[ComparisonRun, ComparisonRun, float] | None:
    eligible = [
        run
        for run in runs
        if run.case.board_size == board_size
        and v2._finite_float(run.metrics.get(metric)) is not None
    ]
    if len(eligible) < 2:
        return None
    ordered = sorted(
        eligible,
        key=lambda run: float(run.metrics[metric]),
        reverse=True,
    )
    leader, runner_up = ordered[:2]
    return (
        leader,
        runner_up,
        float(leader.metrics[metric]) - float(runner_up.metrics[metric]),
    )


def _within_size_observation(
    runs: Sequence[ComparisonRun], board_size: int
) -> str | None:
    parts: list[str] = []
    for metric, label in CORE_METRICS:
        outcome = _within_board_runner_up(runs, board_size, metric)
        if outcome is None:
            continue
        leader, runner_up, gap = outcome
        parts.append(
            f"{label}: **{leader.case.architecture.title()}-"
            f"{leader.case.objective.upper()}** "
            f"{v2._pct(leader.metrics[metric])}, ahead of "
            f"{runner_up.case.architecture.title()}-"
            f"{runner_up.case.objective.upper()} by "
            f"{v2._pp(gap, signed=False)}"
        )
    if not parts:
        return None
    return f"**{board_size}x{board_size}, within-size only.** " + "; ".join(parts) + "."


def _dominance(value: float, contrast: str) -> str:
    if abs(value) * 100.0 < v2.NEGLIGIBLE_PP:
        return "near parity"
    if contrast.startswith("architecture"):
        return "Mamba" if value > 0 else "Transformer"
    if contrast.startswith("objective"):
        return "JEPA" if value > 0 else "AR"
    return "positive" if value > 0 else "negative"


def _trend_kind(values: Sequence[float]) -> str:
    first, last = values[0], values[-1]
    threshold = v2.NEGLIGIBLE_PP / 100.0
    if (
        abs(first) >= threshold
        and abs(last) >= threshold
        and first * last < 0
    ):
        return "dominance reversal"
    nonzero = [value for value in values if abs(value) >= threshold]
    if len(nonzero) >= 2 and any(
        left * right < 0 for left, right in zip(nonzero, nonzero[1:])
    ):
        return "intermediate sign reversal; ends without reversed dominance"
    change = abs(last) - abs(first)
    if abs(change) < threshold:
        return "approximately stable"
    return "narrows toward parity" if change < 0 else "widens away from parity"


def contrast_trajectory(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
    metric: str,
    contrast: str,
) -> list[tuple[int, float]]:
    return [
        (board_size, value)
        for board_size in sorted(contrasts)
        if (
            value := contrasts[board_size].get(metric, {}).get(contrast)
        )
        is not None
    ]


def _trend_observations(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    observations: list[str] = []
    metric_labels = dict(CORE_METRICS)
    priority = (
        "architecture_ar",
        "architecture_jepa",
        "objective_transformer",
        "objective_mamba",
        "interaction",
    )
    for metric, _ in CORE_METRICS:
        for contrast in priority:
            points = contrast_trajectory(contrasts, metric, contrast)
            if len(points) < 2:
                continue
            values = [value for _, value in points]
            kind = _trend_kind(values)
            if kind == "approximately stable" and contrast == "interaction":
                continue
            sequence = " -> ".join(
                f"{size}x{size} {v2._pp(value)}" for size, value in points
            )
            dominance = (
                f"{_dominance(values[0], contrast)} to "
                f"{_dominance(values[-1], contrast)}"
            )
            observations.append(
                f"**{metric_labels[metric]} — {CONTRAST_LABELS[contrast]}:** "
                f"{sequence}; the controlled gap **{kind}** ({dominance})."
            )
    reversals = [item for item in observations if "dominance reversal" in item]
    others = [item for item in observations if "dominance reversal" not in item]
    return (reversals + others)[:6]


def _executive_lines(
    runs: Sequence[ComparisonRun],
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    if not runs:
        return [
            "No result-level observation was generated because every selected "
            "cell is pending or invalid."
        ]
    if len(runs) == 1:
        return v2._leader_observations(runs)
    lines = [
        "Raw levels are ranked only among cells with the **same board size**; "
        "cross-board conclusions below concern trajectories of within-size "
        "percentage-point contrasts, never a global 8x8-vs-16x16 winner."
    ]
    for board_size in _boards(runs):
        observation = _within_size_observation(runs, board_size)
        if observation:
            lines.append(observation)
    if len(_boards(runs)) >= 2:
        lines.extend(_trend_observations(contrasts))
    lines.append(
        "Legality contrasts remain system-level (native AR head versus frozen "
        "JEPA readout); board-state contrasts are encoder-matched. Every gap "
        "is a one-seed descriptive estimate."
    )
    return lines


def _contrast_rows(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[list[str]]:
    labels = dict(CONTRAST_METRICS)
    rows: list[list[str]] = []
    for board_size, board_metrics in contrasts.items():
        for metric, values in board_metrics.items():
            rows.append(
                [
                    f"{board_size}x{board_size}",
                    labels.get(metric, metric),
                    v2._pp(values.get("architecture_ar")),
                    v2._pp(values.get("architecture_jepa")),
                    v2._pp(values.get("objective_transformer")),
                    v2._pp(values.get("objective_mamba")),
                    v2._pp(values.get("architecture_main")),
                    v2._pp(values.get("objective_main")),
                    v2._pp(values.get("interaction")),
                ]
            )
    return rows


def _trajectory_rows(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[list[str]]:
    labels = dict(CONTRAST_METRICS)
    board_sizes = sorted(contrasts)
    if len(board_sizes) < 2:
        return []
    rows: list[list[str]] = []
    for metric, _ in CONTRAST_METRICS:
        for contrast in SIMPLE_CONTRASTS:
            points = contrast_trajectory(contrasts, metric, contrast)
            if len(points) < 2:
                continue
            by_board = dict(points)
            values = [value for _, value in points]
            rows.append(
                [
                    labels[metric],
                    CONTRAST_LABELS[contrast],
                    *[
                        v2._pp(by_board.get(board_size))
                        for board_size in board_sizes
                    ],
                    v2._pp(values[-1] - values[0]),
                    _trend_kind(values),
                ]
            )
    return rows


def compute_efficiency_ratios(
    runs: Sequence[ComparisonRun],
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for board_size in _boards(runs):
        hours = {
            run.system_key: value
            for run in runs
            if run.case.board_size == board_size
            and (value := v2._finite_float(run.metrics.get("training_hours")))
            is not None
            and value > 0
        }
        ratios: dict[str, float] = {}
        pairs = {
            "mamba_over_transformer_ar": (("mamba", "ar"), ("transformer", "ar")),
            "mamba_over_transformer_jepa": (
                ("mamba", "jepa"),
                ("transformer", "jepa"),
            ),
            "jepa_over_ar_transformer": (
                ("transformer", "jepa"),
                ("transformer", "ar"),
            ),
            "jepa_over_ar_mamba": (("mamba", "jepa"), ("mamba", "ar")),
        }
        for key, (numerator, denominator) in pairs.items():
            if numerator in hours and denominator in hours:
                ratios[key] = hours[numerator] / hours[denominator]
        if ratios:
            result[board_size] = ratios
    return result


EFFICIENCY_LABELS = {
    "mamba_over_transformer_ar": "Mamba / Transformer | AR",
    "mamba_over_transformer_jepa": "Mamba / Transformer | JEPA",
    "jepa_over_ar_transformer": "JEPA / AR | Transformer",
    "jepa_over_ar_mamba": "JEPA / AR | Mamba",
}


def _efficiency_ratio_rows(
    ratios: Mapping[int, Mapping[str, float]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for board_size, values in ratios.items():
        rows.append(
            [
                f"{board_size}x{board_size}",
                *[
                    f"{values[key]:.2f}x" if key in values else "n/a"
                    for key in EFFICIENCY_LABELS
                ],
            ]
        )
    return rows


def _plot_within_size_dumbbells(
    runs: Sequence[ComparisonRun], path: Path
) -> bool:
    rows: list[tuple[int, str, ComparisonRun, ComparisonRun]] = []
    by_key = {
        (run.case.board_size, run.case.architecture, run.case.objective): run
        for run in runs
    }
    for board_size in _boards(runs):
        for objective in OBJECTIVES:
            transformer = by_key.get((board_size, "transformer", objective))
            mamba = by_key.get((board_size, "mamba", objective))
            if transformer is not None and mamba is not None:
                rows.append((board_size, objective, transformer, mamba))
    if not rows:
        return False
    plt = v2._pyplot()
    fig, axes = plt.subplots(1, len(CORE_METRICS), figsize=(15.0, 5.4))
    if len(CORE_METRICS) == 1:
        axes = [axes]
    y_positions = list(range(len(rows)))
    for axis, (metric, label) in zip(axes, CORE_METRICS):
        drew = False
        for y, (_, _, transformer, mamba) in zip(y_positions, rows):
            left = v2._finite_float(transformer.metrics.get(metric))
            right = v2._finite_float(mamba.metrics.get(metric))
            if left is None or right is None:
                continue
            drew = True
            axis.plot(
                [100 * left, 100 * right],
                [y, y],
                color="0.65",
                linewidth=2,
            )
            axis.scatter(
                100 * left,
                y,
                marker="o",
                s=46,
                color=v2.SYSTEM_COLORS[transformer.system_key],
            )
            axis.scatter(
                100 * right,
                y,
                marker="s",
                s=46,
                color=v2.SYSTEM_COLORS[mamba.system_key],
            )
            axis.text(
                100 * left,
                y - 0.18,
                f"{100 * left:.1f}",
                ha="center",
                fontsize=7,
            )
            axis.text(
                100 * right,
                y + 0.25,
                f"{100 * right:.1f}",
                ha="center",
                fontsize=7,
            )
        if not drew:
            axis.set_visible(False)
            continue
        axis.set_title(label)
        axis.set_xlabel("Level (%) — compare endpoints only within each row")
        axis.grid(axis="x", alpha=0.25)
        axis.set_yticks(
            y_positions,
            labels=[
                f"{size}x{size} · {objective.upper()}"
                for size, objective, _, _ in rows
            ],
        )
        axis.invert_yaxis()
    fig.suptitle(
        "Within-size architecture gaps: Transformer (circle) vs Mamba (square)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_contrast_heatmap(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
    path: Path,
) -> bool:
    if not contrasts:
        return False
    plt = v2._pyplot()
    import matplotlib.colors as colors
    import numpy as np

    boards = sorted(contrasts)
    matrices: list[Any] = []
    for metric, _ in CORE_METRICS:
        matrices.append(
            np.asarray(
                [
                    [
                        contrasts.get(board, {}).get(metric, {}).get(key, np.nan)
                        * 100.0
                        for board in boards
                    ]
                    for key in SIMPLE_CONTRASTS
                ],
                dtype=float,
            )
        )
    finite = [
        abs(value)
        for matrix in matrices
        for value in matrix.flat
        if np.isfinite(value)
    ]
    if not finite:
        return False
    limit = max(1.0, max(finite))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(
        1,
        len(CORE_METRICS),
        figsize=(17.8, 5.8),
        sharey=True,
        layout="constrained",
    )
    if len(CORE_METRICS) == 1:
        axes = [axes]
    image = None
    chart_labels = {
        "legal_preferred_top1": "Legal top-1 / best head",
        "legal_preferred_mass": "Legal mass / best head",
        "board_macro_mean": "Board-state macro mean",
    }
    for axis, ((metric, _), matrix) in zip(axes, zip(CORE_METRICS, matrices)):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu", norm=norm)
        axis.set_title(chart_labels[metric])
        axis.set_xticks(
            range(len(boards)),
            labels=[f"{board}x{board}" for board in boards],
        )
        axis.set_yticks(
            range(len(SIMPLE_CONTRASTS)),
            labels=[CONTRAST_LABELS[key] for key in SIMPLE_CONTRASTS],
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    "n/a" if not np.isfinite(value) else f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
    if image is not None:
        bar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.035)
        bar.set_label("Controlled effect (percentage points)")
    fig.suptitle(
        "Controlled within-size effects — columns are board sizes, not pooled tasks"
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_contrast_trajectories(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
    path: Path,
) -> bool:
    if len(contrasts) < 2:
        return False
    plt = v2._pyplot()
    styles = {
        "architecture_ar": ("#F58518", "o", "-"),
        "architecture_jepa": ("#E45756", "s", "-"),
        "objective_transformer": ("#4C78A8", "^", "--"),
        "objective_mamba": ("#72B7B2", "D", "--"),
        "interaction": ("#333333", "X", ":"),
    }
    fig, axes = plt.subplots(1, len(CORE_METRICS), figsize=(15.4, 4.9), sharex=True)
    if len(CORE_METRICS) == 1:
        axes = [axes]
    drew = False
    for axis, (metric, label) in zip(axes, CORE_METRICS):
        for contrast in SIMPLE_CONTRASTS:
            points = contrast_trajectory(contrasts, metric, contrast)
            if len(points) < 2:
                continue
            drew = True
            color, marker, linestyle = styles[contrast]
            axis.plot(
                [size for size, _ in points],
                [100 * value for _, value in points],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                label=CONTRAST_LABELS[contrast],
            )
        axis.axhline(0.0, color="0.35", linewidth=1)
        axis.set_title(label)
        axis.set_xticks(
            BOARD_SIZES,
            labels=[f"{size}x{size}" for size in BOARD_SIZES],
        )
        axis.set_xlabel("Board size")
        axis.set_ylabel("Controlled effect (pp)")
        axis.grid(alpha=0.22)
    if not drew:
        plt.close(fig)
        return False
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "How controlled gaps change with board size — zero crossing means "
        "dominance reversal"
    )
    fig.subplots_adjust(bottom=0.27, top=0.84, wspace=0.3)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_efficiency_ratios(
    ratios: Mapping[int, Mapping[str, float]], path: Path
) -> bool:
    if not ratios:
        return False
    plt = v2._pyplot()
    import matplotlib.colors as colors
    import numpy as np

    boards = sorted(ratios)
    keys = tuple(EFFICIENCY_LABELS)
    raw = np.asarray(
        [[ratios.get(board, {}).get(key, np.nan) for board in boards] for key in keys],
        dtype=float,
    )
    if not np.isfinite(raw).any():
        return False
    logged = np.log2(raw)
    limit = max(0.25, float(np.nanmax(np.abs(logged))))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axis = plt.subplots(figsize=(8.6, 4.2))
    image = axis.imshow(logged, aspect="auto", cmap="PuOr", norm=norm)
    axis.set_xticks(range(len(boards)), labels=[f"{board}x{board}" for board in boards])
    axis.set_yticks(range(len(keys)), labels=[EFFICIENCY_LABELS[key] for key in keys])
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            value = raw[row, column]
            axis.text(
                column,
                row,
                "n/a" if not np.isfinite(value) else f"{value:.2f}x",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_title("Size-matched training-time ratios (1.00x = parity)")
    bar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    bar.set_label("log2 time ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_figures(
    runs: Sequence[ComparisonRun],
    output_dir: Path,
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
    efficiency_ratios: Mapping[int, Mapping[str, float]],
) -> dict[str, Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "within_size",
            "within_size_dumbbells.png",
            _plot_within_size_dumbbells,
            (runs,),
        ),
        (
            "contrast_heatmap",
            "controlled_effects_heatmap.png",
            _plot_contrast_heatmap,
            (contrasts,),
        ),
        (
            "contrast_trajectories",
            "controlled_effect_trajectories.png",
            _plot_contrast_trajectories,
            (contrasts,),
        ),
        (
            "efficiency_ratios",
            "size_matched_efficiency.png",
            _plot_efficiency_ratios,
            (efficiency_ratios,),
        ),
        (
            "phase",
            "legal_phase_heatmap.png",
            v2._plot_phase_heatmap,
            (runs,),
        ),
        (
            "position",
            "legal_position_curve.png",
            v2._plot_position_curve,
            (runs,),
        ),
        (
            "board_layers",
            "board_layerwise_curves.png",
            v2._plot_board_layers,
            (runs,),
        ),
    ]
    created: dict[str, Path] = {}
    for key, filename, writer, args in specs:
        path = figure_dir / filename
        if writer(*args, path):
            created[key] = path
    return created


def _architecture_gap_readings(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    readings: list[str] = []
    for metric, label in CORE_METRICS:
        parts: list[str] = []
        for board_size in sorted(contrasts):
            values = contrasts[board_size].get(metric, {})
            ar = values.get("architecture_ar")
            jepa = values.get("architecture_jepa")
            if ar is None and jepa is None:
                continue
            parts.append(
                f"{board_size}x{board_size}: AR {v2._pp(ar)}, "
                f"JEPA {v2._pp(jepa)}"
            )
        if parts:
            readings.append(
                f"**{label}:** " + "; ".join(parts) + " (Mamba - Transformer)."
            )
    return readings


def _strongest_effect_readings(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    readings: list[str] = []
    for metric, label in CORE_METRICS:
        candidates = [
            (abs(value), board_size, contrast, value)
            for board_size, board_metrics in contrasts.items()
            for contrast, value in board_metrics.get(metric, {}).items()
            if contrast in SIMPLE_CONTRASTS
        ]
        if not candidates:
            continue
        _, board_size, contrast, value = max(candidates)
        readings.append(
            f"**{label}:** the largest displayed controlled effect is "
            f"{CONTRAST_LABELS[contrast]} on {board_size}x{board_size} "
            f"({v2._pp(value)})."
        )
    return readings


def _efficiency_readings(
    ratios: Mapping[int, Mapping[str, float]],
) -> list[str]:
    readings: list[str] = []
    for board_size, values in ratios.items():
        if not values:
            continue
        key, ratio = max(
            values.items(),
            key=lambda item: abs(math.log2(item[1])),
        )
        readings.append(
            f"**{board_size}x{board_size}:** the largest recorded same-size "
            f"time asymmetry is {EFFICIENCY_LABELS[key]} at {ratio:.2f}x."
        )
    readings.append(
        "Wall-clock ratios remain hardware/runtime measurements; interpret "
        "them as algorithmic cost only when the paired runs used comparable "
        "hardware and logging coverage."
    )
    return readings


def _phase_readings(runs: Sequence[ComparisonRun]) -> list[str]:
    readings: list[str] = []
    for board_size in _boards(runs):
        phase_runs = [
            run
            for run in runs
            if run.case.board_size == board_size
            and run.metrics.get("legal_phase_bins")
        ]
        if len(phase_runs) < 2:
            continue
        width = min(len(run.metrics["legal_phase_bins"]) for run in phase_runs)
        spreads: list[
            tuple[int, float, ComparisonRun, ComparisonRun]
        ] = []
        for index in range(width):
            available = [
                (run, value)
                for run in phase_runs
                if (
                    value := v2._finite_float(
                        run.metrics["legal_phase_bins"][index].get(
                            "top1_legal"
                        )
                    )
                )
                is not None
            ]
            if len(available) < 2:
                continue
            leader, high = max(available, key=lambda item: item[1])
            laggard, low = min(available, key=lambda item: item[1])
            spreads.append((index, high - low, leader, laggard))
        if spreads:
            index, spread, leader, laggard = max(
                spreads, key=lambda item: item[1]
            )
            readings.append(
                f"**{board_size}x{board_size}:** the widest within-size phase "
                f"separation is bin {index + 1}, where {leader.label} leads "
                f"{laggard.label} by {v2._pp(spread, signed=False)}."
            )
    phase_runs = [run for run in runs if run.metrics.get("legal_phase_bins")]
    if phase_runs:
        readings.append(
            "Rows use each system's best available head, so phase gaps retain "
            f"the same provenance boundary: {v2._head_provenance_note(phase_runs)}."
        )
    return readings


def _visual_lines(
    runs: Sequence[ComparisonRun],
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
    ratios: Mapping[int, Mapping[str, float]],
    figure_paths: Mapping[str, Path],
    report_path: Path,
) -> list[str]:
    if not figure_paths:
        return []
    effects = v2._factorial_effects(runs)
    descriptions = {
        "within_size": (
            "Within-size architecture dumbbells",
            "Each row fixes board size and objective; only the "
            "Transformer-to-Mamba endpoint gap is interpreted.",
            _architecture_gap_readings(contrasts),
        ),
        "contrast_heatmap": (
            "Controlled-effect heatmap",
            "Diverging cells show conditional architecture/objective effects "
            "and interaction independently at each board size.",
            _strongest_effect_readings(contrasts),
        ),
        "contrast_trajectories": (
            "Cross-board trajectories of controlled gaps",
            "Lines connect within-size effects, not raw performance levels; "
            "crossing zero marks a change in the dominant condition.",
            _trend_observations(contrasts),
        ),
        "efficiency_ratios": (
            "Size-matched efficiency ratios",
            "Every cell is a training-time ratio between conditions on the "
            "same board; 1.00x is parity.",
            _efficiency_readings(ratios),
        ),
        "phase": (
            "Game-phase robustness",
            "Top-1 legality by normalized game phase. Cross-board levels are "
            "descriptive; controlled claims remain in the effect panels.",
            _phase_readings(runs),
        ),
        "position": (
            "Legality by move index",
            "Full-resolution diagnostic showing where legality is lost inside "
            "each task; it is not a cross-board leaderboard.",
            v2._interpret_position(runs, effects),
        ),
        "board_layers": (
            "Board decodability by normalized depth",
            "Layerwise macro accuracy with validation-selected layers marked; "
            "normalized depth is used across architectures.",
            v2._interpret_board_layers(runs, effects),
        ),
    }
    order = (
        "within_size",
        "contrast_heatmap",
        "contrast_trajectories",
        "efficiency_ratios",
        "phase",
        "position",
        "board_layers",
    )
    lines = ["## Visual analysis", ""]
    for key in order:
        path = figure_paths.get(key)
        if path is None:
            continue
        title, description, readings = descriptions[key]
        relative = path.relative_to(report_path.parent).as_posix()
        lines += [
            f"### {title}",
            "",
            f"![{title}]({relative})",
            "",
            f"**What it shows.** {description}",
            "",
            "**What the data says.**",
            "",
        ]
        lines += [f"- {reading}" for reading in readings]
        if not readings:
            lines.append(
                "- The selected slice does not support a multi-size trend statement."
            )
        lines.append("")
    return lines


def _write_contrast_csv(
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = dict(CONTRAST_METRICS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "board_size",
                "metric",
                "metric_label",
                "contrast",
                "contrast_label",
                "effect_proportion",
                "effect_percentage_points",
            ]
        )
        for board_size, board_metrics in contrasts.items():
            for metric, values in board_metrics.items():
                for contrast, value in values.items():
                    writer.writerow(
                        [
                            board_size,
                            metric,
                            labels.get(metric, metric),
                            contrast,
                            CONTRAST_LABELS[contrast],
                            f"{value:.10f}",
                            f"{100 * value:.6f}",
                        ]
                    )
    return path


def _report_lines(
    *,
    selection: ComparisonSelection,
    runs: Sequence[ComparisonRun],
    missing: Mapping[str, str],
    artifacts_root: Path,
    report_path: Path,
    figure_paths: Mapping[str, Path],
    export_paths: Mapping[str, Path],
    contrasts: Mapping[int, Mapping[str, Mapping[str, float]]],
    efficiency_ratios: Mapping[int, Mapping[str, float]],
    protocol_id: str,
) -> list[str]:
    mode = (
        "Single-run dossier"
        if selection.is_single_case
        else "Controlled comparative analysis"
    )
    lines = [
        "# Thesis Common Evaluation — Controlled Comparative Report (v3)",
        "",
        f"- **Selection:** {selection.display}",
        f"- **Mode:** {mode}",
        f"- **Coverage:** {len(runs)}/{selection.expected_count} selected "
        "canonical cells ready",
        f"- **Protocol:** `{protocol_id}`",
        f"- **Report schema:** `{COMPARISON_REPORT_VERSION}`",
        "- **Generated (UTC):** "
        f"`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
    ]
    lines += v2._provenance_lines(runs)
    lines += ["", "## Executive view", ""]
    lines += [f"- {item}" for item in _executive_lines(runs, contrasts)]
    if missing:
        lines.append(
            f"- **{len(missing)} selected cell(s) are pending or invalid.** "
            "No missing cell is imputed and no incomplete 2x2 is called a "
            "factorial result."
        )
    lines += [
        "",
        "## Analytical rule",
        "",
        "Raw performance levels are compared only within one board size. "
        "Cross-board analysis compares how those within-size effects change "
        "from 8x8 to 12x12 to 16x16. This separates scaling of an architecture/"
        "objective gap from the changing action space, game length, class mix, "
        "and difficulty of the board-size task itself.",
        "",
        "## Comparability gates",
        "",
    ]
    lines += v2._md_table(
        ("Gate", "Status", "Evidence / interpretation"),
        v2._comparability_gates(runs, selection, protocol_id),
    )
    lines += ["", "## Run coverage and provenance", ""]
    lines += v2._md_table(
        ("Cell", "Status", "Run", "Split manifest", "Source / reason"),
        v2._coverage_rows(runs, missing, artifacts_root),
    )
    if not runs:
        lines += [
            "",
            "## Result",
            "",
            "No metric table or chart was produced because no selected "
            "current-protocol result is complete.",
            "",
        ]
        return lines

    lines += [""] + v2._metric_glossary_lines()
    lines += [
        "## Next-move head selection",
        "",
        "AR contributes its single native head. JEPA contributes the frozen "
        "Linear or MLP readout chosen on validation, never test. The legality "
        "effects are therefore system-level; the board-probe effects below are "
        "the encoder-matched comparison.",
        "",
    ]
    lines += v2._md_table(
        (
            "Run",
            "Head kind",
            "Headline head",
            "Validation selection score",
            "Best / completed shards",
            "Selected on",
        ),
        v2._head_selection_rows(runs),
    )
    lines += [
        "",
        "## Legal-move compatibility",
        "",
        "Every available head is listed; ★ marks the validation-selected "
        "headline row used in controlled legality effects. Confidence intervals "
        "cover game-level test sampling only, not training-seed variance.",
        "",
    ]
    lines += v2._md_table(
        (
            "Run",
            "Head",
            "Top-1 legal",
            "95% CI",
            "Legal mass",
            "Random baseline",
            "Normalized lift",
            "Prec@3",
            "Prec@5",
        ),
        v2._legal_rows(runs),
        numeric_columns=(2, 4, 5, 6, 7, 8),
    )
    lines += [
        "",
        "## Board-state decodability",
        "",
        "Identical frozen-encoder probe protocol across all four conditions; "
        "within-size gaps are the controlled representation comparison.",
        "",
    ]
    lines += v2._md_table(
        (
            "Run",
            "Probe",
            "Absolute macro",
            "Layer / depth",
            "Relative macro",
            "Layer / depth",
            "Two-mode mean",
            "Majority baseline",
        ),
        v2._board_rows(runs),
        numeric_columns=(2, 4, 6, 7),
    )
    lines += [
        "",
        "## Efficiency and footprint",
        "",
        "Absolute resources are shown for provenance. Cross-board efficiency "
        "claims use the size-matched ratios below, not a global fastest/slowest pair.",
        "",
    ]
    lines += v2._md_table(
        (
            "Run",
            "Encoder params",
            "Total params",
            "Checkpoint",
            "Train h",
            "Eval h",
            "Median games/s",
            "Train precision",
        ),
        v2._efficiency_rows(runs),
        numeric_columns=(1, 2, 3, 4, 5, 6),
    )

    lines += [
        "",
        "## Within-size controlled contrasts",
        "",
        "All entries are percentage-point differences. Positive architecture "
        "effects favour Mamba; positive objective effects favour JEPA. The "
        "interaction is `(Mamba-JEPA - Mamba-AR) - "
        "(Transformer-JEPA - Transformer-AR)`.",
        "",
    ]
    contrast_rows = _contrast_rows(contrasts)
    if contrast_rows:
        lines += v2._md_table(
            (
                "Board",
                "Metric",
                "M-T | AR",
                "M-T | JEPA",
                "JEPA-AR | T",
                "JEPA-AR | M",
                "Arch main",
                "Obj main",
                "Interaction",
            ),
            contrast_rows,
            numeric_columns=(2, 3, 4, 5, 6, 7, 8),
        )
    else:
        lines.append("The selected slice contains no estimable paired contrast.")

    trajectory_rows = _trajectory_rows(contrasts)
    if trajectory_rows:
        board_sizes = sorted(contrasts)
        lines += [
            "",
            "## Cross-board evolution of controlled gaps",
            "",
            "The final change column is `last available board - first available "
            "board`. A sign change is reported as a dominance reversal only "
            "when both sides exceed the 0.5 pp negligible-gap threshold.",
            "",
        ]
        lines += v2._md_table(
            (
                "Metric",
                "Controlled contrast",
                *[f"{size}x{size}" for size in board_sizes],
                "First->last change",
                "Pattern",
            ),
            trajectory_rows,
            numeric_columns=tuple(range(2, 3 + len(board_sizes))),
        )

    ratio_rows = _efficiency_ratio_rows(efficiency_ratios)
    if ratio_rows:
        lines += [
            "",
            "## Size-matched efficiency ratios",
            "",
            "A ratio above 1.00x means the numerator condition took longer on "
            "the same board size. This is the controlled alternative to a "
            "global fastest-8x8 versus slowest-16x16 comparison.",
            "",
        ]
        lines += v2._md_table(
            (
                "Board",
                *EFFICIENCY_LABELS.values(),
            ),
            ratio_rows,
            numeric_columns=(1, 2, 3, 4),
        )

    lines += [""] + _visual_lines(
        runs,
        contrasts,
        efficiency_ratios,
        figure_paths,
        report_path,
    )
    lines += [
        "## Interpretation boundaries",
        "",
        "- No global raw-score winner is declared across different board sizes.",
        "- Cross-board statements concern changes in within-size controlled gaps; "
        "they are not zero-shot transfer or equal-difficulty claims.",
        "- Legality is system-level because AR and JEPA have different head "
        "provenance; board probes are encoder-matched but measure decodability.",
        "- Percentage-point effects and dominance reversals are descriptive. "
        "One seed per cell cannot establish seed-level architecture effects.",
        "- Missing cells are visible and never imputed, pooled, or borrowed from "
        "legacy protocols.",
        "",
        "## Reproduction",
        "",
        f"- Artifact root: `{artifacts_root}`",
        f"- Registry-driven selection: `{selection.slug}`",
        f"- Markdown report: `{report_path}`",
    ]
    for name, path in sorted(export_paths.items()):
        lines.append(f"- {name.upper()} export: `{path}`")
    lines.append("")
    return lines


def build_comparison_report(
    *,
    artifacts_root: str | Path,
    architecture: str = "both",
    objective: str = "both",
    board_size: int | str = "all",
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_root: str | Path | None = None,
    protocol_id: str = PROTOCOL_ID,
) -> ComparisonReport:
    """Create a v3 report using within-size effects as the analytical unit."""

    root = Path(artifacts_root)
    selection = ComparisonSelection.from_values(
        architecture=architecture,
        objective=objective,
        board_size=board_size,
    )
    runs, missing = v2.collect_comparison_runs(
        root,
        selection,
        registry_path=registry_path,
        protocol_id=protocol_id,
    )
    base_output = (
        Path(output_root)
        if output_root is not None
        else root / "reports" / "thesis_eval" / "comparisons_v3" / protocol_id
    )
    output_dir = base_output / selection.slug
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"
    contrasts = compute_controlled_contrasts(runs)
    efficiency_ratios = compute_efficiency_ratios(runs)
    figures = (
        _write_figures(runs, output_dir, contrasts, efficiency_ratios)
        if runs
        else {}
    )
    exports = v2._write_exports(runs, output_dir)
    if contrasts:
        exports["contrasts_csv"] = _write_contrast_csv(
            contrasts, output_dir / "controlled_contrasts.csv"
        )
    report_path.write_text(
        "\n".join(
            _report_lines(
                selection=selection,
                runs=runs,
                missing=missing,
                artifacts_root=root,
                report_path=report_path,
                figure_paths=figures,
                export_paths=exports,
                contrasts=contrasts,
                efficiency_ratios=efficiency_ratios,
                protocol_id=protocol_id,
            )
        ),
        encoding="utf-8",
    )
    warnings: list[str] = []
    if len(selection.board_sizes) > 1:
        warnings.append(
            "Cross-board raw levels are descriptive only; v3 conclusions use "
            "within-size controlled-effect trajectories."
        )
    return ComparisonReport(
        report_path=report_path,
        figure_paths=figures,
        runs=tuple(runs),
        missing=missing,
        selection=selection,
        export_paths=exports,
        warnings=tuple(warnings),
    )


__all__ = [
    "COMPARISON_REPORT_VERSION",
    "ComparisonReport",
    "ComparisonRun",
    "ComparisonSelection",
    "build_comparison_report",
    "compute_controlled_contrasts",
    "compute_efficiency_ratios",
    "contrast_trajectory",
    "metric_glossary_markdown",
]
