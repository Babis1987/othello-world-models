"""Within-board controlled contrasts and cross-board trajectories."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from othello_thesis.evaluation import comparison_v2 as v2


ComparisonRun = v2.ComparisonRun

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
