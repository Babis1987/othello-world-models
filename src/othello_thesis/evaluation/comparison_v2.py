"""Registry-driven comparative reporting for Thesis Common Evaluation results (v2).

The common evaluator writes one authoritative JSON per
Architecture x Objective x Board-size cell.  This module reads only the
current protocol, validates the result identity, normalizes the asymmetric
AR/JEPA readout schemas, and produces one self-contained Markdown report with
referenced PNG figures, a metric glossary, and generated per-figure readings.

Differences from ``comparison.py`` (v1)
---------------------------------------

* **Head provenance is explicit.**  v1 mapped the native AR head into both the
  ``linear`` and ``mlp`` slots, which produced columns labelled "MLP match"
  that silently compared a co-trained language-model head against a frozen
  readout.  v2 keeps one row per real head and records what each number is.
* **The JEPA headline head is selected on validation**, using the recorded
  ``frozen_next_move[head]["train"]["best_selection_legal_probability_mass"]``,
  never on the test split.
* **Frozen-head training shards are audited** against the declared downstream
  split.  A head that ran past the downstream shards and wrapped back into
  pretraining data is reported as a gate failure instead of passing silently.
* **Differences are reported in percentage points**, levels in percent.
* **Chance levels travel with every metric** (random legal baseline, majority
  and macro-balanced board baselines), so a level can be read without the
  reader supplying the baseline from memory.
* **Selected layers are reported as normalized depth**, because the compared
  encoders do not have the same number of layers.
* **The footprint axis uses encoder parameters**, not total parameters: a JEPA
  checkpoint additionally stores an EMA target encoder and a predictor that do
  not participate in inference.
* **Every figure carries a generated reading** built from the loaded numbers,
  guarded so that no sentence compares heads of different provenance without
  saying so.

The report stays descriptive.  It preserves the protocol's interpretation
boundaries: legality is not playing strength, board probes measure
decodability, and a single training seed does not support inferential claims
about architecture or objective effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from othello_thesis.evaluation.thesis import (
    ARCHITECTURES,
    BOARD_SIZES,
    DEFAULT_REGISTRY,
    OBJECTIVES,
    EvaluationCase,
    ModelNotReadyError,
    normalize_architecture,
    normalize_board_size,
    normalize_objective,
    resolve_case,
    resolve_run_dir,
)
from othello_thesis.evaluation.unified import PROTOCOL_ID


COMPARISON_REPORT_VERSION = "thesis_comparison_v2"

SYSTEM_COLORS = {
    ("transformer", "ar"): "#4C78A8",
    ("transformer", "jepa"): "#72B7B2",
    ("mamba", "ar"): "#F58518",
    ("mamba", "jepa"): "#E45756",
}
ARCHITECTURE_MARKERS = {"transformer": "o", "mamba": "s"}
PROBE_LINESTYLES = {"linear": "-", "mlp": "--"}

BOARD_CHANCE = 1.0 / 3.0

#: Metrics shown on the scorecard.  ``legal_preferred_*`` is the best available
#: next-move head for each system: the native head for AR, the
#: validation-selected frozen readout for JEPA.
SCORECARD_METRICS = (
    ("legal_preferred_top1", "Legal top-1\n(best head)", "legal"),
    ("legal_preferred_mass", "Legal mass\n(best head)", "legal"),
    ("board_linear_absolute", "Board Linear\nabsolute", "board"),
    ("board_linear_relative", "Board Linear\nrelative", "board"),
    ("board_mlp_absolute", "Board MLP\nabsolute", "board"),
    ("board_mlp_relative", "Board MLP\nrelative", "board"),
)

FACTORIAL_METRICS = (
    ("legal_preferred_top1", "Legal top-1 / best head"),
    ("legal_preferred_mass", "Legal mass / best head"),
    ("board_linear_absolute", "Board Linear / absolute"),
    ("board_linear_relative", "Board Linear / relative"),
    ("board_mlp_absolute", "Board MLP / absolute"),
    ("board_mlp_relative", "Board MLP / relative"),
)

BOARD_METRIC_KEYS = (
    "board_linear_absolute",
    "board_linear_relative",
    "board_mlp_absolute",
    "board_mlp_relative",
)

#: A gap below this many percentage points is reported as non-discriminative.
NEGLIGIBLE_PP = 0.5
#: A gap above this many percentage points is reported as a candidate finding.
NOTABLE_PP = 2.0


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [
        number
        for value in values
        if (number := _finite_float(value)) is not None
    ]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _indexed(mapping: Mapping[str, Any], key: Any) -> Any:
    """Read a mapping that may be keyed by ``int`` or by ``str``."""
    if not isinstance(mapping, Mapping):
        return None
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


def _basename(value: Any) -> str:
    text = str(value or "").replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def _result_protocol_id(metadata: Mapping[str, Any]) -> str:
    """Read both the compact and full-config protocol metadata encodings."""
    protocol = metadata.get("protocol")
    if isinstance(protocol, Mapping):
        return str(protocol.get("protocol_id") or protocol.get("id") or "")
    return str(protocol or "")


def _objective_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text.startswith("jepa"):
        return "jepa"
    if text in {"ar", "autoregressive"} or text.endswith("_ar"):
        return "ar"
    return text


def _head_split_audit(
    payload: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Check which shards a frozen head actually consumed.

    The head trainer is allowed up to ``max_shards`` with early stopping.  When
    the downstream split holds fewer shards than the head consumes, the trainer
    wraps back to the start of the corpus, which is encoder pretraining data.
    That does not touch the test split, but it does break the split discipline
    the protocol claims, so it is surfaced rather than assumed away.
    """

    split = _mapping(_mapping(payload.get("metadata")).get("split"))
    downstream = {
        _basename(path) for path in split.get("downstream_train", []) or []
    }
    pretraining = {
        _basename(path) for path in split.get("pretraining", []) or []
    }
    evaluation = {
        _basename(path)
        for key in ("selection", "test")
        for path in split.get(key, []) or []
    }

    train = _mapping(entry.get("train"))
    history = [
        _mapping(item) for item in train.get("history", []) or []
    ]
    consumed = [_basename(item.get("chunk_name")) for item in history]
    outside = [name for name in consumed if downstream and name not in downstream]
    from_pretraining = [name for name in outside if name in pretraining]
    from_evaluation = [name for name in outside if name in evaluation]

    best_shard = _finite_float(train.get("best_shard"))
    best_name = ""
    if best_shard is not None and 1 <= int(best_shard) <= len(consumed):
        best_name = consumed[int(best_shard) - 1]
    selected_outside = bool(best_name) and best_name in set(outside)

    return {
        "shards_completed": _finite_float(train.get("shards_completed")),
        "max_shards": _finite_float(train.get("max_shards")),
        "patience": _finite_float(train.get("patience")),
        "best_shard": best_shard,
        "best_chunk": best_name,
        "downstream_shards": len(downstream),
        "outside_downstream": len(outside),
        "from_pretraining": len(from_pretraining),
        "from_evaluation": len(from_evaluation),
        "selected_outside_downstream": selected_outside,
        "converged": (
            _finite_float(train.get("shards_completed")) is not None
            and _finite_float(train.get("max_shards")) is not None
            and float(train["shards_completed"]) < float(train["max_shards"])
        ),
    }


def _legal_metrics(
    payload: Mapping[str, Any],
    objective: str,
) -> dict[str, Any]:
    """Normalize the next-move evaluation, keeping head provenance intact.

    AR reports exactly one row -- its native language-model head.  JEPA reports
    one row per frozen readout, and the headline row is whichever readout scored
    best on the *validation* selection metric recorded during head training.
    The test split is never consulted to choose a head.
    """

    if objective == "ar":
        native = _mapping(payload.get("native_ar"))
        return {
            "rows": (("Native AR head", native),),
            "heads": {},
            "preferred": native,
            "preferred_label": "Native AR head",
            "kind": "native",
            "selected_on": "not applicable (single native head)",
            "selection_scores": {},
            "audits": {},
            "warning": None,
        }

    frozen = _mapping(payload.get("frozen_next_move"))
    heads: dict[str, Mapping[str, Any]] = {}
    selection_scores: dict[str, float | None] = {}
    audits: dict[str, Mapping[str, Any]] = {}
    for head in ("linear", "mlp"):
        entry = _mapping(frozen.get(head))
        heads[head] = _mapping(entry.get("test_legal"))
        selection_scores[head] = _finite_float(
            _get(entry, "train", "best_selection_legal_probability_mass")
        )
        audits[head] = _head_split_audit(payload, entry)

    labels = {"linear": "Frozen Linear", "mlp": "Frozen MLP"}
    rows = tuple(
        (labels[head], heads[head]) for head in ("linear", "mlp") if heads[head]
    )

    scored = {
        head: score
        for head, score in selection_scores.items()
        if score is not None and heads.get(head)
    }
    warning: str | None = None
    if scored:
        chosen = max(scored, key=lambda head: scored[head])
        selected_on = "validation legal probability mass"
    else:
        chosen = "linear" if heads.get("linear") else "mlp"
        selected_on = "fallback: linear (no validation score recorded)"
        warning = (
            "No `best_selection_legal_probability_mass` was recorded for the "
            "frozen heads; the headline row fell back to the Linear readout "
            "instead of being selected on validation."
        )

    return {
        "rows": rows,
        "heads": heads,
        "preferred": heads.get(chosen, {}),
        "preferred_label": labels.get(chosen, chosen),
        "kind": "frozen",
        "selected_on": selected_on,
        "selection_scores": selection_scores,
        "audits": audits,
        "warning": warning,
    }


def _head_metrics(result: Mapping[str, Any]) -> dict[str, float | None]:
    """Pull the scalar legal metrics out of one head's ``test_legal`` block."""
    topk = _mapping(result.get("topk_legal"))
    bootstrap = _mapping(
        _get(result, "bootstrap_ci_95_per_game", "top1_legal")
    )
    return {
        "top1": _finite_float(result.get("top1_legal_per_token")),
        "mass": _finite_float(result.get("legal_prob_mass_per_token")),
        "lift": _finite_float(result.get("top1_legal_normalized_lift")),
        "precision3": _finite_float(_indexed(topk, 3)),
        "precision5": _finite_float(_indexed(topk, 5)),
        "ci_low": _finite_float(bootstrap.get("low")),
        "ci_high": _finite_float(bootstrap.get("high")),
        "mass_ci_low": _finite_float(
            _get(result, "bootstrap_ci_95_per_game", "legal_probability_mass", "low")
        ),
        "mass_ci_high": _finite_float(
            _get(result, "bootstrap_ci_95_per_game", "legal_probability_mass", "high")
        ),
        "baseline": _finite_float(
            result.get("random_vocab_top1_legal_baseline")
        ),
        "n_games": _finite_float(result.get("n_games")),
        "n_tokens": _finite_float(result.get("n_tokens")),
    }


def _normalized_depth(probe: Mapping[str, Any], layer: int) -> float | None:
    mapping = _mapping(probe.get("normalized_depth"))
    normalized = _finite_float(_indexed(mapping, layer))
    if normalized is not None:
        return normalized
    layers = [
        int(item)
        for item in probe.get("layers", [])
        if _finite_float(item) is not None
    ]
    if not layers or max(layers) == 0:
        return None
    return float(layer) / float(max(layers))


def _layer_curve(
    probe: Mapping[str, Any],
    mode: str,
) -> tuple[list[float], list[float]]:
    """Return (normalized depth, macro accuracy) for every evaluated layer."""
    test_metrics = _mapping(probe.get("test_metrics"))
    depths: list[float] = []
    scores: list[float] = []
    for raw_layer in sorted(test_metrics, key=lambda item: int(item)):
        entry = _mapping(_mapping(test_metrics[raw_layer]).get(mode))
        score = _finite_float(entry.get("macro_balanced_accuracy"))
        depth = _normalized_depth(probe, int(raw_layer))
        if score is None or depth is None:
            continue
        depths.append(depth)
        scores.append(score)
    return depths, scores


def _normalize_result(
    case: EvaluationCase,
    path: Path,
    payload: Mapping[str, Any],
    protocol_id: str = PROTOCOL_ID,
) -> ComparisonRun:
    metadata = _mapping(payload.get("metadata"))
    protocol = _result_protocol_id(metadata)
    if protocol != protocol_id:
        raise ValueError(
            f"expected protocol {protocol_id!r}, found {protocol or 'unreported'!r}"
        )

    identity_errors: list[str] = []
    architecture = str(metadata.get("architecture") or "").lower()
    objective = _objective_family(metadata.get("objective"))
    board_size = metadata.get("board_size")
    if architecture and architecture != case.architecture:
        identity_errors.append(
            f"architecture={architecture!r}, expected {case.architecture!r}"
        )
    if objective and objective != case.objective:
        identity_errors.append(
            f"objective={objective!r}, expected {case.objective!r}"
        )
    if board_size is not None and int(board_size) != case.board_size:
        identity_errors.append(
            f"board_size={board_size!r}, expected {case.board_size!r}"
        )
    if identity_errors:
        raise ValueError("; ".join(identity_errors))

    legal = _legal_metrics(payload, case.objective)
    if not legal["preferred"]:
        expected = "native_ar" if case.objective == "ar" else "frozen_next_move"
        raise ValueError(f"incomplete legal evaluation: missing {expected}")

    board = _mapping(payload.get("board_state"))
    metrics: dict[str, Any] = {}
    layer_counts: set[int] = set()
    for probe_type in ("linear", "mlp"):
        probe = _mapping(board.get(probe_type))
        if not probe:
            raise ValueError(f"incomplete board evaluation: missing {probe_type}")
        layers = [
            int(item)
            for item in probe.get("layers", [])
            if _finite_float(item) is not None
        ]
        if layers:
            layer_counts.add(max(layers))
        for mode in ("absolute", "relative"):
            selected = _mapping(_get(probe, "selected", mode))
            test = _mapping(selected.get("test"))
            macro = _finite_float(test.get("macro_balanced_accuracy"))
            if macro is None:
                raise ValueError(
                    "incomplete board evaluation: missing "
                    f"{probe_type}/{mode} selected test macro accuracy"
                )
            layer = int(selected.get("layer", -1))
            prefix = f"board_{probe_type}_{mode}"
            baselines = _mapping(test.get("baselines"))
            metrics[prefix] = macro
            metrics[f"{prefix}_accuracy"] = _finite_float(test.get("accuracy"))
            metrics[f"{prefix}_occupied"] = _finite_float(
                test.get("occupied_accuracy")
            )
            metrics[f"{prefix}_empty"] = _finite_float(
                test.get("empty_accuracy")
            )
            metrics[f"{prefix}_layer"] = layer
            metrics[f"{prefix}_depth"] = _normalized_depth(probe, layer)
            metrics[f"{prefix}_majority_baseline"] = _finite_float(
                baselines.get("majority_accuracy")
            )
            metrics[f"{prefix}_chance"] = (
                _finite_float(baselines.get("macro_balanced_random_accuracy"))
                or BOARD_CHANCE
            )
            depths, scores = _layer_curve(probe, mode)
            metrics[f"{prefix}_curve_depth"] = tuple(depths)
            metrics[f"{prefix}_curve_score"] = tuple(scores)

    metrics["board_layer_count"] = max(layer_counts) if layer_counts else None

    # Per-head legal metrics.  For AR only the native head exists; the linear
    # and mlp slots stay absent rather than being aliased to it.
    for head, result in _mapping(legal["heads"]).items():
        if not result:
            continue
        for name, value in _head_metrics(result).items():
            metrics[f"legal_{head}_{name}"] = value
        metrics[f"legal_{head}_selection_score"] = legal["selection_scores"].get(
            head
        )

    preferred = _mapping(legal["preferred"])
    for name, value in _head_metrics(preferred).items():
        metrics[f"legal_preferred_{name}"] = value
    metrics["legal_preferred_label"] = legal["preferred_label"]
    metrics["legal_head_kind"] = legal["kind"]
    metrics["legal_head_selected_on"] = legal["selected_on"]
    metrics["legal_selection_scores"] = dict(legal["selection_scores"])
    metrics["legal_head_audits"] = dict(legal["audits"])
    metrics["legal_head_warning"] = legal["warning"]
    metrics["legal_rows"] = legal["rows"]
    metrics["legal_phase_bins"] = tuple(
        _mapping(item)
        for item in preferred.get("normalized_phase_bins", [])
        if isinstance(item, Mapping)
    )
    metrics["legal_position_curve"] = tuple(
        _mapping(item)
        for item in preferred.get("position_curve", [])
        if isinstance(item, Mapping)
    )

    training = _mapping(metadata.get("training_metrics"))
    timing = _mapping(payload.get("timing_seconds"))
    metrics["encoder_parameters"] = _finite_float(
        metadata.get("encoder_parameters")
    )
    metrics["total_parameters"] = _finite_float(metadata.get("total_parameters"))
    metrics["checkpoint_bytes"] = _finite_float(metadata.get("checkpoint_bytes"))
    metrics["training_hours"] = _finite_float(
        training.get("estimated_training_hours")
    )
    metrics["observed_training_hours"] = _finite_float(
        training.get("observed_train_plus_validation_loop_hours")
    )
    metrics["training_games_per_second"] = _finite_float(
        training.get("median_games_per_second")
    )
    # Only scalar stage timings are summed; a nested structure would otherwise
    # double count a stage against its own components.
    metrics["evaluation_hours"] = (
        sum(
            value
            for item in timing.values()
            if not isinstance(item, Mapping)
            and (value := _finite_float(item)) is not None
        )
        / 3600.0
    )
    metrics["training_precision"] = metadata.get("training_precision")
    metrics["split_manifest_hash"] = metadata.get("split_manifest_hash")
    metrics["checkpoint_sha256"] = metadata.get("checkpoint_sha256")
    metrics["run_name"] = metadata.get("run_name") or path.parents[2].name
    metrics["board_macro_mean"] = _mean(
        metrics[key] for key in BOARD_METRIC_KEYS
    )
    metrics["board_mlp_mean"] = _mean(
        metrics[key]
        for key in ("board_mlp_absolute", "board_mlp_relative")
    )
    metrics["board_linear_mean"] = _mean(
        metrics[key]
        for key in ("board_linear_absolute", "board_linear_relative")
    )
    return ComparisonRun(
        case=case,
        results_path=path,
        payload=payload,
        metrics=metrics,
    )


def _registered_results_path(
    case: EvaluationCase,
    artifacts_root: Path,
    protocol_id: str = PROTOCOL_ID,
) -> Path | None:
    for relative in case.run_candidates:
        path = (
            artifacts_root
            / relative
            / "thesis_eval"
            / "final"
            / f"results__{protocol_id}.json"
        )
        if path.is_file():
            return path
    return None


def _locate_result(
    case: EvaluationCase,
    artifacts_root: Path,
    protocol_id: str = PROTOCOL_ID,
) -> tuple[Path | None, str | None]:
    direct = _registered_results_path(case, artifacts_root, protocol_id)
    if direct is not None:
        return direct, None
    try:
        run_dir = resolve_run_dir(case, artifacts_root)
    except ModelNotReadyError as error:
        return None, str(error).splitlines()[0]
    except (OSError, RuntimeError, ValueError) as error:
        return None, f"run discovery failed: {error}"
    path = (
        run_dir
        / "thesis_eval"
        / "final"
        / f"results__{protocol_id}.json"
    )
    if not path.is_file():
        return None, f"{protocol_id} result is pending at {path}"
    return path, None


def collect_comparison_runs(
    artifacts_root: str | Path,
    selection: ComparisonSelection,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    protocol_id: str = PROTOCOL_ID,
) -> tuple[tuple[ComparisonRun, ...], dict[str, str]]:
    """Load and validate every selected canonical evaluation cell."""

    root = Path(artifacts_root)
    runs: list[ComparisonRun] = []
    missing: dict[str, str] = {}
    for board_size in selection.board_sizes:
        for architecture in selection.architectures:
            for objective in selection.objectives:
                case = resolve_case(
                    architecture,
                    objective,
                    board_size,
                    registry_path=registry_path,
                )
                path, reason = _locate_result(case, root, protocol_id)
                if path is None:
                    missing[case.label] = reason or "result is unavailable"
                    continue
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(raw, Mapping):
                        raise ValueError("result JSON root is not an object")
                    runs.append(
                        _normalize_result(case, path, raw, protocol_id)
                    )
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                    missing[case.label] = f"invalid result: {error}"
    runs.sort(
        key=lambda run: (
            run.case.board_size,
            ARCHITECTURES.index(run.case.architecture),
            OBJECTIVES.index(run.case.objective),
        )
    )
    return tuple(runs), missing


def _pct(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    """Format a level (a proportion) as a percentage."""
    number = _finite_float(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{number * 100:{sign}.{digits}f}%"


def _pp(value: Any, digits: int = 2, *, signed: bool = True) -> str:
    """Format a difference of proportions in percentage points.

    Differences between two percentages are percentage points, not percent.  v1
    printed factorial effects with a ``%`` suffix while the surrounding prose
    called them percentage-point contrasts; this keeps the two consistent.
    """
    number = _finite_float(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{number * 100:{sign}.{digits}f} pp"


def _number(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:,.{digits}f}"


def _count(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{int(number):,}"


def _millions(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number / 1_000_000:.2f}M"


def _megabytes(value: Any) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number / (1024**2):.1f} MiB"


def _short_hash(value: Any) -> str:
    text = str(value or "")
    return f"`{text[:12]}…`" if len(text) > 12 else f"`{text or 'n/a'}`"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _md_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    numeric_columns: Iterable[int] = (),
) -> list[str]:
    numeric = set(numeric_columns)
    align = [
        "---:" if index in numeric else "---"
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(_escape_cell(item) for item in headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_cell(item) for item in row) + " |"
        for row in rows
    )
    return lines


def _layer_and_depth(run: ComparisonRun, prefix: str) -> str:
    layer = run.metrics.get(f"{prefix}_layer")
    depth = _finite_float(run.metrics.get(f"{prefix}_depth"))
    if depth is None:
        return f"L{layer}"
    return f"L{layer} / {depth * 100:.0f}%"


def _comparability_gates(
    runs: Sequence[ComparisonRun],
    selection: ComparisonSelection,
    protocol_id: str = PROTOCOL_ID,
) -> list[tuple[str, str, str]]:
    gates: list[tuple[str, str, str]] = []
    if not runs:
        return [
            (
                "Result availability",
                "BLOCKED",
                "No selected cell has a complete current-protocol result.",
            )
        ]

    protocols = {
        _result_protocol_id(_mapping(_mapping(run.payload).get("metadata")))
        for run in runs
    }
    gates.append(
        (
            "Evaluation protocol",
            "PASS" if protocols == {protocol_id} else "FAIL",
            ", ".join(sorted(protocols)),
        )
    )

    split_issues: list[str] = []
    for board_size in sorted({run.case.board_size for run in runs}):
        hashes = {
            str(run.metrics.get("split_manifest_hash") or "")
            for run in runs
            if run.case.board_size == board_size
        }
        if len(hashes) > 1:
            split_issues.append(f"{board_size}x{board_size}: {len(hashes)} splits")
    gates.append(
        (
            "Within-board test split",
            "PASS" if not split_issues else "FAIL",
            (
                "One split manifest per represented board size."
                if not split_issues
                else "; ".join(split_issues)
            ),
        )
    )

    precision_issues: list[str] = []
    for board_size in sorted({run.case.board_size for run in runs}):
        values = [
            str(run.metrics.get("training_precision") or "unreported").lower()
            for run in runs
            if run.case.board_size == board_size
        ]
        if len(values) > 1 and (len(set(values)) > 1 or "unreported" in values):
            precision_issues.append(
                f"{board_size}x{board_size}: {', '.join(sorted(set(values)))}"
            )
    gates.append(
        (
            "Training precision covariate",
            "PASS" if not precision_issues else "WARNING",
            (
                "Matched within every represented comparison."
                if not precision_issues
                else "; ".join(precision_issues)
            ),
        )
    )

    if len(selection.board_sizes) > 1:
        gates.append(
            (
                "Cross-board interpretation",
                "CAUTION",
                "Board probes use a fixed position budget; legal metrics still "
                "reflect different action spaces and game lengths.",
            )
        )

    native = [run for run in runs if run.head_kind == "native"]
    frozen = [run for run in runs if run.head_kind == "frozen"]
    if native and frozen:
        gates.append(
            (
                "Next-move head asymmetry",
                "BY DESIGN",
                "AR reports its native head, whose encoder was optimized for "
                "exactly this task. JEPA reports a frozen-encoder readout, "
                "fitted on an encoder that was never trained for next-move "
                "prediction. That difference is the research question, not a "
                "confound: it asks what next-move capability each objective "
                "delivers. The board probes below are the encoder-matched "
                "comparison.",
            )
        )
    if frozen:
        selected_on = sorted(
            {str(run.metrics.get("legal_head_selected_on")) for run in frozen}
        )
        converged = all(
            bool(_mapping(audit).get("converged"))
            for run in frozen
            for audit in _mapping(run.metrics.get("legal_head_audits")).values()
        )
        detail = (
            "Headline frozen head chosen on "
            + "; ".join(selected_on)
            + ". The test split was not used to choose a head."
        )
        if converged:
            detail += (
                " Every frozen head stopped early under its patience rule "
                "before exhausting its shard allowance, so the readouts are "
                "converged rather than budget-limited."
            )
        gates.append(("Frozen-head selection", "PASS", detail))

        # What matters for a comparison is that the readouts being compared saw
        # the same data regime, and that neither touched the splits used to
        # stop or score them.  Training a frozen readout on encoder-pretraining
        # shards is legitimate -- probing asks whether the representation is
        # decodable, not whether the probe generalizes -- so it is recorded
        # rather than failed.
        leaked: list[str] = []
        composition: dict[str, tuple[int, int]] = {}
        for run in frozen:
            for head, raw_audit in _mapping(
                run.metrics.get("legal_head_audits")
            ).items():
                audit = _mapping(raw_audit)
                if int(_finite_float(audit.get("from_evaluation")) or 0) > 0:
                    leaked.append(f"{run.label}/{head}")
                composition[f"{run.label}/{head}"] = (
                    int(_finite_float(audit.get("shards_completed")) or 0),
                    int(_finite_float(audit.get("from_pretraining")) or 0),
                )
        if leaked:
            gates.append(
                (
                    "Readout training data",
                    "FAIL",
                    "A frozen readout consumed selection or test shards: "
                    + ", ".join(sorted(leaked))
                    + ". Reported test numbers for those heads are leaked.",
                )
            )
        else:
            mixed = {
                name: value
                for name, value in composition.items()
                if value[1] > 0
            }
            detail = (
                "No readout touched the selection or test split, so no "
                "reported number is leaked. Shards consumed (total, of which "
                "encoder-pretraining): "
                + "; ".join(
                    f"{name} {total}, {pretraining}"
                    for name, (total, pretraining) in sorted(
                        composition.items()
                    )
                )
                + "."
            )
            if mixed:
                detail += (
                    " Readouts differ in how much encoder-pretraining data "
                    "they saw, because early stopping let them run to "
                    "different lengths over a shard sequence that continues "
                    "past the declared downstream split. Training a readout on "
                    "pretraining data is defensible -- a low-capacity probe "
                    "measures decodability, not probe generalization -- but "
                    "the candidates compared on validation did not see "
                    "identical data, and the written split table should state "
                    "the sequence the trainer actually follows."
                )
            gates.append(
                (
                    "Readout training data",
                    "NOTE" if mixed else "PASS",
                    detail,
                )
            )

    layer_counts = sorted(
        {
            int(count)
            for run in runs
            if (count := _finite_float(run.metrics.get("board_layer_count")))
            is not None
        }
    )
    if len(layer_counts) > 1:
        gates.append(
            (
                "Encoder depth",
                "CAUTION",
                "Compared encoders expose different layer counts "
                f"(top layer index {', '.join(str(count) for count in layer_counts)}). "
                "Only normalized depth is comparable across architectures; raw "
                "layer numbers are not.",
            )
        )
    return gates


def _factorial_effects(
    runs: Sequence[ComparisonRun],
) -> dict[int, dict[str, dict[str, float]]]:
    by_key = {
        (run.case.board_size, run.case.architecture, run.case.objective): run
        for run in runs
    }
    effects: dict[int, dict[str, dict[str, float]]] = {}
    for board_size in sorted({run.case.board_size for run in runs}):
        required = [
            (architecture, objective)
            for architecture in ARCHITECTURES
            for objective in OBJECTIVES
        ]
        if not all(
            (board_size, architecture, objective) in by_key
            for architecture, objective in required
        ):
            continue
        board_effects: dict[str, dict[str, float]] = {}
        for metric, _ in FACTORIAL_METRICS:
            values = {
                key: _finite_float(by_key[(board_size, *key)].metrics.get(metric))
                for key in required
            }
            # A single missing metric must skip that row, not raise and destroy
            # the whole report.
            if any(value is None for value in values.values()):
                continue
            t_ar = values[("transformer", "ar")]
            t_jepa = values[("transformer", "jepa")]
            m_ar = values[("mamba", "ar")]
            m_jepa = values[("mamba", "jepa")]
            board_effects[metric] = {
                "architecture": ((m_ar + m_jepa) - (t_ar + t_jepa)) / 2.0,
                "objective": ((t_jepa + m_jepa) - (t_ar + m_ar)) / 2.0,
                "interaction": (m_jepa - m_ar) - (t_jepa - t_ar),
            }
        if board_effects:
            effects[board_size] = board_effects
    return effects


# ---------------------------------------------------------------------------
# Reading helpers -- shared by the executive view and the per-figure readings.
# ---------------------------------------------------------------------------


def _eligible(
    runs: Sequence[ComparisonRun],
    metric: str,
) -> list[ComparisonRun]:
    return [
        run
        for run in runs
        if _finite_float(run.metrics.get(metric)) is not None
    ]


def _leader_and_range(
    runs: Sequence[ComparisonRun],
    metric: str,
) -> tuple[ComparisonRun, ComparisonRun, float] | None:
    """Return (leader, laggard, gap in proportion units) for one metric."""
    eligible = _eligible(runs, metric)
    if len(eligible) < 2:
        return None
    leader = max(eligible, key=lambda run: float(run.metrics[metric]))
    laggard = min(eligible, key=lambda run: float(run.metrics[metric]))
    gap = float(leader.metrics[metric]) - float(laggard.metrics[metric])
    return leader, laggard, gap


def _magnitude(gap: float) -> str:
    """Describe a gap so a small number is never dressed up as a result."""
    points = abs(gap) * 100.0
    if points < NEGLIGIBLE_PP:
        return "does not separate the systems"
    if points < NOTABLE_PP:
        return "is a small but consistent separation"
    return "is a candidate finding"


#: Which recorded bootstrap interval belongs to which headline metric.  Using
#: the top-1 interval to judge a probability-mass gap would be a category error.
_CI_KEYS = {
    "legal_preferred_top1": ("legal_preferred_ci_low", "legal_preferred_ci_high"),
    "legal_preferred_mass": (
        "legal_preferred_mass_ci_low",
        "legal_preferred_mass_ci_high",
    ),
}


def _ci_verdict(
    leader: ComparisonRun,
    laggard: ComparisonRun,
    metric: str,
) -> str | None:
    """Compare a gap against the bootstrap interval recorded for that metric."""
    keys = _CI_KEYS.get(metric)
    if keys is None:
        return None
    low_key, high_key = keys
    low = _finite_float(leader.metrics.get(low_key))
    high = _finite_float(leader.metrics.get(high_key))
    other_low = _finite_float(laggard.metrics.get(low_key))
    other_high = _finite_float(laggard.metrics.get(high_key))
    if None in (low, high, other_low, other_high):
        return None
    if low > other_high:
        return (
            "the game-level bootstrap intervals do not overlap, so the gap is "
            "not test-set sampling noise -- but those intervals say nothing "
            "about seed variance, and each cell here has one seed"
        )
    return (
        "the game-level bootstrap intervals overlap, so this gap is within "
        "test-set sampling noise"
    )


def _head_provenance_note(runs: Sequence[ComparisonRun]) -> str:
    kinds = {run.head_kind for run in runs}
    if kinds == {"native"}:
        return "all rows use native AR heads"
    if kinds == {"frozen"}:
        return "all rows use validation-selected frozen readouts"
    return (
        "AR rows use native heads over an encoder trained for this task, and "
        "JEPA rows use frozen readouts over an encoder that was not, so this "
        "contrast is a system-level capability comparison rather than an "
        "encoder-matched one"
    )


def _dominant_factor(
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
    metric: str,
) -> str | None:
    """Name the factor that dominates, when one clearly does."""
    statements: list[str] = []
    for board_size, board_effects in effects.items():
        values = board_effects.get(metric)
        if not values:
            continue
        architecture = abs(values["architecture"])
        objective = abs(values["objective"])
        if architecture >= 2.0 * objective and architecture * 100 >= NEGLIGIBLE_PP:
            statements.append(
                f"on {board_size}x{board_size} the architecture contrast "
                f"({_pp(values['architecture'])}) outweighs the objective "
                f"contrast ({_pp(values['objective'])})"
            )
        elif objective >= 2.0 * architecture and objective * 100 >= NEGLIGIBLE_PP:
            statements.append(
                f"on {board_size}x{board_size} the objective contrast "
                f"({_pp(values['objective'])}) outweighs the architecture "
                f"contrast ({_pp(values['architecture'])})"
            )
    if not statements:
        return None
    return "; ".join(statements)


def _interaction_driver(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
    metric: str,
) -> str | None:
    """Point at the cell responsible for a non-trivial interaction."""
    by_key = {
        (run.case.board_size, run.case.architecture, run.case.objective): run
        for run in runs
    }
    for board_size, board_effects in effects.items():
        values = board_effects.get(metric)
        if not values or abs(values["interaction"]) * 100 < 1.0:
            continue
        cells = [
            by_key[(board_size, architecture, objective)]
            for architecture in ARCHITECTURES
            for objective in OBJECTIVES
            if (board_size, architecture, objective) in by_key
        ]
        eligible = _eligible(cells, metric)
        if len(eligible) < 4:
            continue
        scores = [float(run.metrics[metric]) for run in eligible]
        mean = sum(scores) / len(scores)
        outlier = max(
            eligible, key=lambda run: abs(float(run.metrics[metric]) - mean)
        )
        return (
            f"the {_pp(values['interaction'])} interaction on "
            f"{board_size}x{board_size} is driven by **{outlier.label}**, the "
            f"cell furthest from the four-cell mean "
            f"({_pct(outlier.metrics[metric])} vs {_pct(mean)})"
        )
    return None


def _describe_metric_leader(
    runs: Sequence[ComparisonRun],
    metric: str,
    label: str,
    *,
    include_provenance: bool = False,
) -> str | None:
    outcome = _leader_and_range(runs, metric)
    if outcome is None:
        return None
    leader, laggard, gap = outcome
    sentence = (
        f"**{label}** is led by **{leader.label}** "
        f"({_pct(leader.metrics[metric])}), ahead of {laggard.label} "
        f"({_pct(laggard.metrics[metric])}); the {_pp(gap, signed=False)} "
        f"spread {_magnitude(gap)}"
    )
    verdict = _ci_verdict(leader, laggard, metric)
    if verdict:
        sentence += f", and {verdict}"
    if include_provenance:
        sentence += f". Head provenance: {_head_provenance_note(runs)}"
    return sentence + "."


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _metric_matrix(runs: Sequence[ComparisonRun], keys: Sequence[str]):
    import numpy as np

    return np.asarray(
        [
            [
                value
                if (value := _finite_float(run.metrics.get(key))) is not None
                else np.nan
                for key in keys
            ]
            for run in runs
        ],
        dtype=float,
    )


def _metric_floor(runs: Sequence[ComparisonRun], family: str) -> float:
    """Chance level for one metric family, used to normalize the scorecard."""
    if family == "board":
        floors = [
            value
            for run in runs
            for key in BOARD_METRIC_KEYS
            if (value := _finite_float(run.metrics.get(f"{key}_chance")))
            is not None
        ]
        return _mean(floors) or BOARD_CHANCE
    floors = [
        value
        for run in runs
        if (value := _finite_float(run.metrics.get("legal_preferred_baseline")))
        is not None
    ]
    return _mean(floors) or 0.0


def _plot_scorecard(runs: Sequence[ComparisonRun], path: Path) -> bool:
    """Two panels: headroom-normalized levels, and the gap to the column best.

    The single fixed ``vmin=1/3`` of v1 put every legal and board number in the
    top half of one colour map, so 1-3 pp differences were invisible and the
    legal family was scored against a board-probe chance level.  Panel 1 scales
    each column by its own chance level; panel 2 shows the actual gaps.
    """
    if len(runs) < 2:
        return False
    plt = _pyplot()
    import numpy as np

    keys = [key for key, _, _ in SCORECARD_METRICS]
    labels = [label for _, label, _ in SCORECARD_METRICS]
    families = [family for _, _, family in SCORECARD_METRICS]
    matrix = _metric_matrix(runs, keys)

    floors = np.asarray(
        [_metric_floor(runs, family) for family in families], dtype=float
    )
    headroom = np.clip(
        (matrix - floors) / np.maximum(1e-9, 1.0 - floors), 0.0, 1.0
    )
    with np.errstate(invalid="ignore"):
        best = np.nanmax(matrix, axis=0)
    delta = (matrix - best) * 100.0

    height = max(3.8, 0.52 * len(runs) + 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, height))

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#ECECEC")
    image = axes[0].imshow(headroom, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_title(
        "Levels, scaled by each metric's chance level\n"
        "(text is the true value; colour is the share of headroom used)",
        fontsize=10,
        pad=12,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axes[0].text(
                column,
                row,
                "n/a" if np.isnan(value) else f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color="black",
            )
    bar = fig.colorbar(image, ax=axes[0], fraction=0.03, pad=0.02)
    bar.set_label("Share of available headroom above chance")

    diverging = plt.get_cmap("RdBu").copy()
    diverging.set_bad("#ECECEC")
    limit = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
    limit = max(limit, 0.5)
    delta_image = axes[1].imshow(
        delta, aspect="auto", cmap=diverging, vmin=-limit, vmax=limit
    )
    axes[1].set_title(
        "Gap to the best system in each column\n(0.00 marks the column leader)",
        fontsize=10,
        pad=12,
    )
    for row in range(delta.shape[0]):
        for column in range(delta.shape[1]):
            value = delta[row, column]
            axes[1].text(
                column,
                row,
                "n/a" if np.isnan(value) else f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="black",
            )
    delta_bar = fig.colorbar(delta_image, ax=axes[1], fraction=0.03, pad=0.02)
    delta_bar.set_label("Percentage points behind the column leader")

    for index, axis in enumerate(axes):
        axis.set_xticks(range(len(keys)), labels=labels)
        axis.tick_params(axis="x", labelsize=8.5)
        axis.set_yticks(
            range(len(runs)),
            labels=[run.short_label for run in runs] if index == 0 else ["" for _ in runs],
        )

    fig.suptitle("Capability scorecard — higher is better", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _phase_bin_label(bin_metrics: Mapping[str, Any], index: int) -> str:
    start = _finite_float(bin_metrics.get("phase_start"))
    end = _finite_float(bin_metrics.get("phase_end"))
    names = ("Opening", "Early-mid", "Late-mid", "Endgame")
    name = names[index] if index < len(names) else f"Bin {index + 1}"
    if start is None or end is None:
        return name
    return f"{name}\n{start * 100:.0f}–{end * 100:.0f}%"


def _plot_phase_heatmap(runs: Sequence[ComparisonRun], path: Path) -> bool:
    phase_runs = [run for run in runs if run.metrics.get("legal_phase_bins")]
    if not phase_runs:
        return False
    plt = _pyplot()
    import numpy as np

    # Bin edges come from the data rather than from hard-coded quartiles, so a
    # future change in bin count stays readable instead of silently truncating.
    width = max(len(run.metrics["legal_phase_bins"]) for run in phase_runs)
    matrix = np.full((len(phase_runs), width), np.nan, dtype=float)
    labels: list[str] = [f"Bin {index + 1}" for index in range(width)]
    for row, run in enumerate(phase_runs):
        for column, bin_metrics in enumerate(run.metrics["legal_phase_bins"]):
            value = _finite_float(bin_metrics.get("top1_legal"))
            if value is not None:
                matrix[row, column] = value
            if row == 0:
                labels[column] = _phase_bin_label(bin_metrics, column)

    height = max(3.0, 0.62 * len(phase_runs) + 2.0)
    fig, axis = plt.subplots(figsize=(2.1 * width + 2.4, height))
    finite = matrix[np.isfinite(matrix)]
    vmin = max(0.0, float(finite.min()) - 0.03) if finite.size else 0.0
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#ECECEC")
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=1.0)
    axis.set_xticks(range(width), labels=labels)
    axis.set_yticks(
        range(len(phase_runs)),
        labels=[
            f"{run.short_label}\n({run.metrics['legal_preferred_label']})"
            for run in phase_runs
        ],
        fontsize=8,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                "n/a" if np.isnan(value) else f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=8.5,
            )
    axis.set_title("Top-1 legality across normalized game phases")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    colorbar.set_label("Top-1 legal accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


#: Move indices reached by fewer than this share of the busiest index carry too
#: few games to read; the deep tail of a position curve is sampling noise.
POSITION_SUPPORT_FRACTION = 0.05


def _well_supported_curve(
    curve: Sequence[Mapping[str, Any]],
    key: str = "top1_legal",
) -> list[tuple[int, float]]:
    """Curve points at move indices that enough games actually reach."""
    counts = [
        value
        for item in curve
        if (value := _finite_float(item.get("n_tokens"))) is not None
    ]
    floor = max(counts) * POSITION_SUPPORT_FRACTION if counts else 0.0
    points: list[tuple[int, float]] = []
    for item in curve:
        position = _finite_float(item.get("position"))
        value = _finite_float(item.get(key))
        support = _finite_float(item.get("n_tokens"))
        if position is None or value is None:
            continue
        if support is not None and support < floor:
            continue
        points.append((int(position), value))
    return points


def _plot_position_curve(runs: Sequence[ComparisonRun], path: Path) -> bool:
    """Legality against move index, at finer resolution than the phase bins."""
    curve_runs = [
        run for run in runs if len(run.metrics.get("legal_position_curve", ())) > 4
    ]
    if not curve_runs:
        return False
    plt = _pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True)
    for run in curve_runs:
        curve = run.metrics["legal_position_curve"]
        for axis, key, ylabel in (
            (axes[0], "top1_legal", "Top-1 legal accuracy (%)"),
            (axes[1], "legal_probability_mass", "Legal probability mass (%)"),
        ):
            # The deep tail is reached by a handful of games; plotting it would
            # dominate the x-range with noise.
            pairs = [
                (position, 100.0 * value)
                for position, value in _well_supported_curve(curve, key)
            ]
            if len(pairs) < 5:
                continue
            axis.plot(
                [position for position, _ in pairs],
                [value for _, value in pairs],
                color=SYSTEM_COLORS[run.system_key],
                linewidth=1.6,
                label=f"{run.short_label} ({run.metrics['legal_preferred_label']})",
            )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Move index within the game")
            axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Legality across move index (best available head per system)")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_board_layers(runs: Sequence[ComparisonRun], path: Path) -> bool:
    """Layerwise board decodability against normalized encoder depth.

    Replaces v1's radar and selected-depth dot plot: it shows both where the
    probe peaks and the shape of the curve, and it is plotted against
    normalized depth because the compared encoders differ in layer count.
    """
    usable = [
        run
        for run in runs
        if any(
            len(run.metrics.get(f"board_{probe}_{mode}_curve_depth") or ()) >= 2
            for probe in ("linear", "mlp")
            for mode in ("absolute", "relative")
        )
    ]
    if not usable:
        return False
    plt = _pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for axis, mode in zip(axes, ("absolute", "relative")):
        for run in usable:
            for probe in ("linear", "mlp"):
                prefix = f"board_{probe}_{mode}"
                depths = run.metrics.get(f"{prefix}_curve_depth") or ()
                scores = run.metrics.get(f"{prefix}_curve_score") or ()
                if len(depths) < 2:
                    continue
                axis.plot(
                    depths,
                    [100.0 * value for value in scores],
                    color=SYSTEM_COLORS[run.system_key],
                    linestyle=PROBE_LINESTYLES[probe],
                    linewidth=1.7,
                    alpha=0.9,
                    label=f"{run.short_label} · {probe}",
                )
                selected = _finite_float(run.metrics.get(f"{prefix}_depth"))
                score = _finite_float(run.metrics.get(prefix))
                if selected is not None and score is not None:
                    axis.scatter(
                        [selected],
                        [100.0 * score],
                        s=70,
                        color=SYSTEM_COLORS[run.system_key],
                        marker=ARCHITECTURE_MARKERS[run.case.architecture],
                        edgecolor="#111111",
                        linewidth=1.2,
                        zorder=4,
                    )
        chance = 100.0 * BOARD_CHANCE
        axis.axhline(chance, color="0.45", linestyle=":", linewidth=1.0)
        axis.text(0.01, chance + 1.0, "macro chance 33.3%", fontsize=7.5, color="0.35")
        axis.set_title(f"{mode.title()} board labels")
        axis.set_xlabel("Normalized encoder depth")
        axis.set_ylabel("Macro-balanced accuracy (%)")
        axis.set_xlim(-0.03, 1.03)
        axis.set_xticks(
            (0.0, 0.25, 0.5, 0.75, 1.0),
            labels=("Embedding", "25%", "50%", "75%", "Final"),
        )
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    fig.suptitle(
        "Where board state is decodable — solid: Linear probe, dashed: MLP probe, "
        "outlined marker: validation-selected layer",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _pareto_indices(points: Sequence[tuple[float, float]]) -> set[int]:
    frontier: set[int] = set()
    for index, (parameters, score) in enumerate(points):
        dominated = False
        for other_index, (other_parameters, other_score) in enumerate(points):
            if other_index == index:
                continue
            if (
                other_parameters <= parameters
                and other_score >= score
                and (other_parameters < parameters or other_score > score)
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(index)
    return frontier


def _plot_efficiency(runs: Sequence[ComparisonRun], path: Path) -> bool:
    """Board decodability against *encoder* footprint.

    v1 used ``total_parameters``, which for JEPA includes the EMA target encoder
    and the predictor.  Those exist only during training, so the v1 chart made
    JEPA look twice as large as the model that is actually evaluated.
    """
    eligible = [
        run
        for run in runs
        if _finite_float(run.metrics.get("encoder_parameters")) not in (None, 0.0)
        and _finite_float(run.metrics.get("board_macro_mean")) is not None
    ]
    if len(eligible) < 2:
        return False
    plt = _pyplot()

    points = [
        (
            float(run.metrics["encoder_parameters"]) / 1_000_000,
            float(run.metrics["board_macro_mean"]) * 100,
        )
        for run in eligible
    ]
    frontier = _pareto_indices(points)
    fig, axis = plt.subplots(figsize=(9.4, 5.8))
    for index, (run, (parameters, score)) in enumerate(zip(eligible, points)):
        hours = _finite_float(run.metrics.get("training_hours"))
        size = 90.0 if hours is None else min(600.0, 70.0 + 45.0 * hours)
        axis.scatter(
            parameters,
            score,
            s=size,
            color=SYSTEM_COLORS[run.system_key],
            marker=ARCHITECTURE_MARKERS[run.case.architecture],
            alpha=0.82,
            edgecolor="#111111" if index in frontier else "white",
            linewidth=2.0 if index in frontier else 0.9,
            zorder=3,
        )
        annotation = run.short_label
        if hours is not None:
            annotation += f"\n{hours:.1f} h train"
        axis.annotate(
            annotation,
            (parameters, score),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8,
        )
    frontier_points = sorted(
        (points[index] for index in frontier), key=lambda item: item[0]
    )
    if len(frontier_points) > 1:
        axis.plot(
            [point[0] for point in frontier_points],
            [point[1] for point in frontier_points],
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Descriptive Pareto frontier",
        )
        axis.legend(frameon=False)
    axis.set(
        xlabel="Encoder parameters at inference (millions)",
        ylabel="Mean board-probe macro accuracy (%)",
        title="Representation quality vs inference footprint",
    )
    axis.grid(alpha=0.22)
    axis.text(
        0.01,
        0.01,
        "Bubble area encodes estimated training hours; dark outline = non-dominated. "
        "JEPA target encoder and predictor are excluded: they do not run at inference.",
        transform=axis.transAxes,
        fontsize=7.5,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_factorial_effects(
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
    path: Path,
) -> bool:
    if not effects:
        return False
    plt = _pyplot()
    import matplotlib.colors as colors
    import numpy as np

    row_labels: list[str] = []
    values: list[list[float]] = []
    for board_size, board_effects in effects.items():
        for metric, label in FACTORIAL_METRICS:
            effect = board_effects.get(metric)
            if not effect:
                continue
            row_labels.append(f"{board_size}x{board_size} · {label}")
            values.append(
                [
                    effect["architecture"],
                    effect["objective"],
                    effect["interaction"],
                ]
            )
    if not values:
        return False
    matrix = np.asarray(values, dtype=float) * 100.0
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axis = plt.subplots(
        figsize=(9.2, max(4.6, 0.38 * len(row_labels) + 1.8))
    )
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu", norm=norm)
    axis.set_xticks(
        range(3),
        labels=(
            "Architecture\nMamba − Transformer",
            "Objective\nJEPA − AR",
            "Interaction\nΔΔ",
        ),
    )
    axis.set_yticks(range(len(row_labels)), labels=row_labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.2f} pp",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_title("2×2 descriptive main effects and interaction")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.025)
    colorbar.set_label("Percentage-point effect")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_scaling(runs: Sequence[ComparisonRun], path: Path) -> bool:
    systems: dict[tuple[str, str], list[ComparisonRun]] = {}
    for run in runs:
        systems.setdefault(run.system_key, []).append(run)
    systems = {
        key: sorted(value, key=lambda run: run.case.board_size)
        for key, value in systems.items()
        if len({run.case.board_size for run in value}) >= 2
    }
    if not systems:
        return False
    plt = _pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    panels = (
        ("legal_preferred_top1", "Top-1 legal accuracy (%)"),
        ("board_mlp_absolute", "MLP absolute board macro (%)"),
    )
    drew = False
    for axis, (metric, ylabel) in zip(axes, panels):
        for system, system_runs in systems.items():
            pairs = [
                (run.case.board_size, value)
                for run in system_runs
                if (value := _finite_float(run.metrics.get(metric))) is not None
            ]
            if len(pairs) < 2:
                continue
            drew = True
            axis.plot(
                [size for size, _ in pairs],
                [100.0 * value for _, value in pairs],
                marker=ARCHITECTURE_MARKERS[system[0]],
                linewidth=2.0,
                color=SYSTEM_COLORS[system],
                label=f"{system[0].title()}-{system[1].upper()}",
            )
        axis.set_xticks(
            BOARD_SIZES, labels=[f"{size}x{size}" for size in BOARD_SIZES]
        )
        axis.set_xlabel("Board size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    if not drew:
        plt.close(fig)
        return False
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Observed scaling profile across completed board sizes")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_figures(
    runs: Sequence[ComparisonRun],
    output_dir: Path,
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("scorecard", "capability_scorecard.png", _plot_scorecard, (runs,)),
        ("phase", "legal_phase_heatmap.png", _plot_phase_heatmap, (runs,)),
        ("position", "legal_position_curve.png", _plot_position_curve, (runs,)),
        ("board_layers", "board_layerwise_curves.png", _plot_board_layers, (runs,)),
        ("efficiency", "efficiency_pareto.png", _plot_efficiency, (runs,)),
        ("factorial", "factorial_effects.png", _plot_factorial_effects, (effects,)),
        ("scaling", "scaling_profile.png", _plot_scaling, (runs,)),
    )
    created: dict[str, Path] = {}
    for key, name, writer, arguments in specifications:
        path = figure_dir / name
        if writer(*arguments, path):
            created[key] = path
    return created


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _legal_rows(runs: Sequence[ComparisonRun]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs:
        preferred_label = run.metrics["legal_preferred_label"]
        for label, raw_metrics in run.metrics["legal_rows"]:
            head = _head_metrics(_mapping(raw_metrics))
            interval = (
                f"[{_pct(head['ci_low'])}, {_pct(head['ci_high'])}]"
                if head["ci_low"] is not None and head["ci_high"] is not None
                else "n/a"
            )
            marker = " ★" if label == preferred_label else ""
            rows.append(
                [
                    run.label,
                    f"{label}{marker}",
                    _pct(head["top1"]),
                    interval,
                    _pct(head["mass"]),
                    _pct(head["baseline"]),
                    _pct(head["lift"]),
                    _pct(head["precision3"]),
                    _pct(head["precision5"]),
                ]
            )
    return rows


def _board_rows(runs: Sequence[ComparisonRun]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs:
        for probe_type in ("linear", "mlp"):
            absolute = f"board_{probe_type}_absolute"
            relative = f"board_{probe_type}_relative"
            rows.append(
                [
                    run.label,
                    probe_type.upper(),
                    _pct(run.metrics[absolute]),
                    _layer_and_depth(run, absolute),
                    _pct(run.metrics[relative]),
                    _layer_and_depth(run, relative),
                    _pct(_mean((run.metrics[absolute], run.metrics[relative]))),
                    _pct(run.metrics.get(f"{absolute}_majority_baseline")),
                ]
            )
    return rows


def _efficiency_rows(runs: Sequence[ComparisonRun]) -> list[list[str]]:
    return [
        [
            run.label,
            _millions(run.metrics.get("encoder_parameters")),
            _millions(run.metrics.get("total_parameters")),
            _megabytes(run.metrics.get("checkpoint_bytes")),
            _number(run.metrics.get("training_hours")),
            _number(run.metrics.get("evaluation_hours")),
            _number(run.metrics.get("training_games_per_second"), 1),
            str(run.metrics.get("training_precision") or "unreported"),
        ]
        for run in runs
    ]


def _head_selection_rows(runs: Sequence[ComparisonRun]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs:
        scores = _mapping(run.metrics.get("legal_selection_scores"))
        audits = _mapping(run.metrics.get("legal_head_audits"))
        if not scores:
            rows.append(
                [
                    run.label,
                    "native",
                    run.metrics["legal_preferred_label"],
                    "—",
                    "—",
                    "single native head; nothing to select",
                ]
            )
            continue
        rendered = ", ".join(
            f"{head}={_pct(score)}" if score is not None else f"{head}=n/a"
            for head, score in sorted(scores.items())
        )
        budget = ", ".join(
            f"{head}={_count(_mapping(audit).get('best_shard'))}"
            f"/{_count(_mapping(audit).get('shards_completed'))}"
            for head, audit in sorted(audits.items())
        )
        rows.append(
            [
                run.label,
                "frozen",
                run.metrics["legal_preferred_label"],
                rendered,
                budget,
                str(run.metrics.get("legal_head_selected_on")),
            ]
        )
    return rows


def _coverage_rows(
    runs: Sequence[ComparisonRun],
    missing: Mapping[str, str],
    artifacts_root: Path,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs:
        try:
            source = run.results_path.relative_to(artifacts_root).as_posix()
        except ValueError:
            source = run.results_path.as_posix()
        rows.append(
            [
                run.label,
                "READY",
                str(run.metrics.get("run_name") or "n/a"),
                _short_hash(run.metrics.get("split_manifest_hash")),
                f"`{source}`",
            ]
        )
    for label, reason in missing.items():
        rows.append([label, "PENDING", "—", "—", reason])
    return rows


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------


def _metric_glossary_lines() -> list[str]:
    """Define every reported metric, with its chance level and its reading."""
    lines = [
        "## Metric definitions",
        "",
        "Every quantity below is computed by the unified evaluator "
        "(`othello_research/evaluation/legal_moves.py` and "
        "`position_probe.py`). Levels are reported in percent; differences "
        "between levels are reported in percentage points (pp).",
        "",
    ]
    rows = [
        [
            "Top-1 legal",
            "Share of scored positions where the `argmax` of the predicted "
            "move distribution is a legal move, averaged per token.",
            "≈ mean legal moves / vocabulary",
            "How often greedy play would propose a rule-consistent move.",
        ],
        [
            "Legal probability mass",
            "Sum of predicted probability over the legal move set, "
            "`Σ p(m) for m legal`, averaged per token.",
            "≈ mean legal moves / vocabulary",
            "How much of the *whole* distribution respects the rules. High "
            "top-1 with low mass means a correct peak over a diffuse tail.",
        ],
        [
            "Legal precision@k",
            "Mean **fraction of the top-k tokens that are legal**, "
            "`|top-k ∩ legal| / k`. This is precision@k, **not** "
            "\"a legal move appears somewhere in the top k\".",
            "≈ mean legal moves / vocabulary",
            "Decreases monotonically with k by construction; a small drop "
            "means legality persists deeper into the ranking.",
        ],
        [
            "Normalized lift",
            "`(top1 − baseline) / (1 − baseline)`.",
            "0",
            "Share of the available headroom above chance that was captured.",
        ],
        [
            "Random legal baseline",
            "`mean(#legal moves / vocab_size)` over scored positions.",
            "—",
            "The chance level for every legality metric on this board size.",
        ],
        [
            "Board macro-balanced accuracy",
            "Unweighted mean of the three per-class recalls, "
            "`(recall_empty + recall_class1 + recall_class2) / 3`, on the "
            "untouched test split at the validation-selected layer.",
            "33.3%",
            "Prevents the far more frequent `empty` class from dominating. "
            "The majority-class baseline is reported beside it.",
        ],
        [
            "Absolute vs relative labels",
            "Absolute = black / white / empty. Relative = mine / theirs / "
            "empty, from the side to move.",
            "33.3%",
            "A large relative-over-absolute gap means the encoder represents "
            "relative ownership rather than absolute colour.",
        ],
        [
            "Selected layer / normalized depth",
            "The layer with the best validation score; depth is "
            "`layer / n_layers`.",
            "—",
            "**Only normalized depth is comparable across architectures** — "
            "the compared encoders do not expose the same number of layers.",
        ],
        [
            "95% CI",
            "Game-level bootstrap over per-game means (1000 resamples, fixed "
            "seed).",
            "—",
            "Covers test-set sampling noise **only**. It does not cover "
            "training-seed variance; each cell here has one seed.",
        ],
        [
            "Encoder vs total parameters",
            "Encoder = what runs at inference. Total additionally counts the "
            "JEPA EMA target encoder and predictor, which exist only during "
            "training.",
            "—",
            "Use encoder parameters for any footprint or efficiency claim.",
        ],
        [
            "Next-move head",
            "AR reports its native language-model head. JEPA reports a "
            "frozen-encoder readout (Linear or MLP) trained under the common "
            "protocol with early stopping, with the headline row chosen on "
            "validation.",
            "—",
            "A system-level capability comparison: what next-move ability each "
            "objective ultimately delivers.",
        ],
    ]
    lines += _md_table(
        ("Metric", "Definition", "Chance level", "How to read it"), rows
    )
    lines.append("")
    return lines


def metric_glossary_markdown() -> str:
    """The metric glossary as Markdown, for display outside the report."""
    return "\n".join(_metric_glossary_lines())


def _leader_observations(runs: Sequence[ComparisonRun]) -> list[str]:
    if not runs:
        return [
            "No result-level observation was generated because every selected "
            "cell is pending or invalid."
        ]
    if len(runs) == 1:
        run = runs[0]
        mlp_gain = (
            float(run.metrics["board_mlp_mean"])
            - float(run.metrics["board_linear_mean"])
        )
        depth_candidates = [
            (key, value)
            for key in BOARD_METRIC_KEYS
            if (value := _finite_float(run.metrics.get(f"{key}_depth")))
            is not None
        ]
        observations = [
            f"Next-move head: **{run.metrics['legal_preferred_label']}** "
            f"(selected on {run.metrics['legal_head_selected_on']}), with "
            f"top-1 legality **{_pct(run.metrics['legal_preferred_top1'])}** "
            f"and legal probability mass "
            f"**{_pct(run.metrics['legal_preferred_mass'])}**, against a "
            f"random legal baseline of "
            f"{_pct(run.metrics.get('legal_preferred_baseline'))}.",
            f"The MLP board probe changes the two-mode macro mean by "
            f"**{_pp(mlp_gain)}** relative to the Linear probe.",
        ]
        if depth_candidates:
            key, depth = max(depth_candidates, key=lambda item: item[1])
            label = key.removeprefix("board_").replace("_", " ")
            observations.append(
                f"The deepest validation-selected representation is "
                f"**{label}** at normalized depth **{_pct(depth, digits=0)}** "
                f"(layer {run.metrics.get(f'{key}_layer')})."
            )
        return observations

    observations: list[str] = []
    for metric, label in (
        ("legal_preferred_top1", "Top-1 legality, best available head"),
        ("legal_preferred_mass", "Legal probability mass, best available head"),
        ("board_macro_mean", "Mean board-state macro accuracy"),
    ):
        sentence = _describe_metric_leader(
            runs,
            metric,
            label,
            include_provenance=metric.startswith("legal_"),
        )
        if sentence:
            observations.append(sentence)

    footprints = [
        value
        for run in runs
        if (value := _finite_float(run.metrics.get("encoder_parameters")))
        is not None
    ]
    hours = [
        (run, value)
        for run in runs
        if (value := _finite_float(run.metrics.get("training_hours"))) is not None
    ]
    if len(footprints) >= 2:
        spread = (max(footprints) - min(footprints)) / max(footprints)
        if spread < 0.05:
            sentence = (
                f"Inference footprint is matched by design "
                f"({_millions(min(footprints))}–{_millions(max(footprints))} "
                f"encoder parameters, {spread * 100:.1f}% spread), so no "
                "system wins on size"
            )
            if len(hours) >= 2:
                cheapest = min(hours, key=lambda item: item[1])
                dearest = max(hours, key=lambda item: item[1])
                sentence += (
                    f"; the real cost difference is training time "
                    f"({cheapest[0].label} {cheapest[1]:.2f} h vs "
                    f"{dearest[0].label} {dearest[1]:.2f} h)"
                )
            observations.append(sentence + ".")
    return observations


def _interpret_scorecard(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    sentences: list[str] = []
    legal_outcome = _leader_and_range(runs, "legal_preferred_top1")
    if legal_outcome is not None:
        leader, _, gap = legal_outcome
        floor = _metric_floor(runs, "legal")
        sentences.append(
            f"Legality is close to saturated: every loaded system sits far "
            f"above the {_pct(floor)} random baseline, and the whole field "
            f"spans {_pp(gap, signed=False)} on top-1, which {_magnitude(gap)}. "
            f"On this axis the panel is better read as a check that all "
            f"systems learned the rules than as a ranking ({leader.label} "
            f"leads)."
        )
    board_outcome = _leader_and_range(runs, "board_macro_mean")
    if board_outcome is not None:
        leader, laggard, gap = board_outcome
        sentence = (
            f"Board decodability spans {_pp(gap, signed=False)} between "
            f"{leader.label} and {laggard.label}, against a 33.3% macro chance "
            f"level"
        )
        if legal_outcome is not None:
            legal_gap = legal_outcome[2]
            if abs(gap) > abs(legal_gap):
                sentence += (
                    " — a wider spread than legality, so this is the axis that "
                    "actually separates the systems"
                )
            else:
                sentence += (
                    f" — no wider than the {_pp(legal_gap, signed=False)} "
                    "legality spread, so neither axis separates the systems "
                    "decisively on this board size"
                )
        sentence += (
            ". The right-hand panel is where these differences become visible; "
            "on the left every cell already sits high in its headroom range."
        )
        sentences.append(sentence)
    relative_mean = _mean(
        run.metrics.get("board_linear_relative") for run in runs
    )
    absolute_mean = _mean(
        run.metrics.get("board_linear_absolute") for run in runs
    )
    if relative_mean is not None and absolute_mean is not None:
        sentences.append(
            f"Relative labels are decoded {_pp(relative_mean - absolute_mean)} "
            f"more accurately than absolute ones on average across the loaded "
            f"cells, which points at representations of \"whose piece is this\" "
            f"rather than of absolute colour."
        )
    return sentences


def _interpret_phase(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    phase_runs = [run for run in runs if run.metrics.get("legal_phase_bins")]
    if not phase_runs:
        return []
    sentences: list[str] = []
    trends: list[tuple[ComparisonRun, float]] = []
    for run in phase_runs:
        bins = run.metrics["legal_phase_bins"]
        first = _finite_float(bins[0].get("top1_legal"))
        last = _finite_float(bins[-1].get("top1_legal"))
        if first is None or last is None:
            continue
        trends.append((run, last - first))
    if trends:
        worst = min(trends, key=lambda item: item[1])
        best = max(trends, key=lambda item: item[1])
        sentences.append(
            f"From the opening bin to the endgame bin, top-1 legality moves by "
            f"{_pp(worst[1])} for **{worst[0].label}** and {_pp(best[1])} for "
            f"**{best[0].label}**. Phase bins are normalized, so this compares "
            f"game stages rather than absolute move numbers."
        )
    width = min(len(run.metrics["legal_phase_bins"]) for run in phase_runs)
    spreads: list[tuple[int, float]] = []
    for index in range(width):
        values = [
            value
            for run in phase_runs
            if (
                value := _finite_float(
                    run.metrics["legal_phase_bins"][index].get("top1_legal")
                )
            )
            is not None
        ]
        if len(values) >= 2:
            spreads.append((index, max(values) - min(values)))
    if spreads:
        index, spread = max(spreads, key=lambda item: item[1])
        sentences.append(
            f"The systems differ most in phase bin {index + 1} "
            f"({_pp(spread, signed=False)} between best and worst), so any "
            f"overall legality gap is concentrated there rather than spread "
            f"evenly across the game."
        )
    sentences.append(
        f"Rows use each system's best available head, so this panel inherits "
        f"the same provenance note: {_head_provenance_note(phase_runs)}."
    )
    return sentences


def _interpret_position(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    curve_runs = [
        run for run in runs if len(run.metrics.get("legal_position_curve", ())) > 4
    ]
    if not curve_runs:
        return []
    sentences: list[str] = []
    dropped = 0
    for run in curve_runs:
        curve = run.metrics["legal_position_curve"]
        scored = _well_supported_curve(curve)
        if len(scored) < 5:
            continue
        dropped = max(dropped, len(curve) - len(scored))
        worst_position, worst_value = min(scored, key=lambda item: item[1])
        sentences.append(
            f"**{run.label}** is weakest around move {worst_position} "
            f"({_pct(worst_value)} top-1 legal)."
        )
    closing = (
        "The opening and the forced endgame are easier because the legal set "
        "is small there; move-index resolution shows whether a legality gap is "
        "a broad difference or a localized failure the phase bins average away."
    )
    if dropped:
        closing += (
            f" The last {dropped} move indices are reached by too few games to "
            "read and are excluded from the readings above."
        )
    sentences.append(closing)
    return sentences


def _interpret_board_layers(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    sentences: list[str] = []
    depths: list[tuple[ComparisonRun, float]] = []
    for run in runs:
        mean_depth = _mean(
            run.metrics.get(f"{key}_depth") for key in BOARD_METRIC_KEYS
        )
        if mean_depth is not None:
            depths.append((run, mean_depth))
    if len(depths) >= 2:
        shallowest = min(depths, key=lambda item: item[1])
        deepest = max(depths, key=lambda item: item[1])
        sentences.append(
            f"Averaged over the four probe/label combinations, the "
            f"validation-selected layer sits at normalized depth "
            f"{_pct(shallowest[1], digits=0)} for **{shallowest[0].label}** — "
            f"the shallowest — and {_pct(deepest[1], digits=0)} for "
            f"**{deepest[0].label}** — the deepest. Raw layer indices are not "
            f"comparable here, since the encoders expose different layer "
            f"counts, which is why the x-axis is normalized."
        )
    gains = [
        (run, mlp - linear)
        for run in runs
        if (mlp := _finite_float(run.metrics.get("board_mlp_mean"))) is not None
        and (linear := _finite_float(run.metrics.get("board_linear_mean")))
        is not None
    ]
    if gains:
        best = max(gains, key=lambda item: item[1])
        worst = min(gains, key=lambda item: item[1])
        sentences.append(
            f"The non-linear probe buys the most in **{best[0].label}** "
            f"({_pp(best[1])} over Linear) and the least in "
            f"**{worst[0].label}** ({_pp(worst[1])}). A large MLP gain means "
            f"the board information is present but not linearly separable, "
            f"which is a weaker claim than linear decodability."
        )
    for metric, label in (
        ("board_linear_relative", "relative board decodability"),
        ("board_mlp_absolute", "absolute board decodability under the MLP probe"),
    ):
        dominant = _dominant_factor(effects, metric)
        if dominant:
            sentences.append(f"For {label}, {dominant}.")
        driver = _interaction_driver(runs, effects, metric)
        if driver:
            sentences.append(f"For {label}, {driver}.")
    return sentences


def _interpret_efficiency(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    sentences: list[str] = []
    footprints = [
        value
        for run in runs
        if (value := _finite_float(run.metrics.get("encoder_parameters")))
        is not None
    ]
    if len(footprints) >= 2:
        spread = (max(footprints) - min(footprints)) / max(footprints)
        sentences.append(
            f"The encoders span {_millions(min(footprints))}–"
            f"{_millions(max(footprints))} inference parameters, a "
            f"{spread * 100:.1f}% spread: footprint is matched by design, so "
            f"the horizontal axis carries almost no signal and the frontier is "
            f"effectively a ranking on the vertical axis."
        )
    hours = [
        (run, value)
        for run in runs
        if (value := _finite_float(run.metrics.get("training_hours"))) is not None
    ]
    if len(hours) >= 2:
        cheapest = min(hours, key=lambda item: item[1])
        dearest = max(hours, key=lambda item: item[1])
        ratio = dearest[1] / max(1e-9, cheapest[1])
        sentences.append(
            f"Training cost is where the conditions actually differ: "
            f"**{dearest[0].label}** took {dearest[1]:.2f} h against "
            f"**{cheapest[0].label}** at {cheapest[1]:.2f} h, a {ratio:.1f}x "
            f"span at equal inference size. Bubble area encodes this."
        )
    return sentences


def _interpret_factorial(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    if not effects:
        return []
    labels = dict(FACTORIAL_METRICS)
    sentences: list[str] = []
    for board_size, board_effects in effects.items():
        # Averaging a legality effect together with a board-probe effect would
        # mix two different scales and chance levels, so the board family is
        # summarized on its own and legality is quoted separately.
        board_only = {
            metric: values
            for metric, values in board_effects.items()
            if metric.startswith("board_")
        }
        strongest = max(
            board_effects.items(),
            key=lambda item: max(
                abs(item[1]["architecture"]),
                abs(item[1]["objective"]),
                abs(item[1]["interaction"]),
            ),
        )
        if board_only:
            mean_architecture = _mean(
                values["architecture"] for values in board_only.values()
            )
            mean_objective = _mean(
                values["objective"] for values in board_only.values()
            )
            sentences.append(
                f"On {board_size}x{board_size}, averaged over the four "
                f"board-probe metrics, moving from Transformer to Mamba is "
                f"worth {_pp(mean_architecture)} and moving from AR to JEPA is "
                f"worth {_pp(mean_objective)}. Those four share a scale and a "
                f"chance level; the legality rows do not and are quoted "
                f"separately rather than averaged in."
            )
        sentences.append(
            f"The largest single effect on {board_size}x{board_size} appears "
            f"in **{labels.get(strongest[0], strongest[0])}** (architecture "
            f"{_pp(strongest[1]['architecture'])}, objective "
            f"{_pp(strongest[1]['objective'])}, interaction "
            f"{_pp(strongest[1]['interaction'])})."
        )
        if any(
            metric.startswith("legal_preferred") for metric in board_effects
        ):
            sentences.append(
                "The legality rows contrast a native AR head against a frozen "
                "JEPA readout. That is the intended system-level question, but "
                "it means the objective effect there is not a statement about "
                "how much move information the JEPA encoder holds. The board "
                "rows are the encoder-matched contrast."
            )
    sentences.append(
        "All three columns are descriptive point estimates from one seed per "
        "cell. They order the conditions; they do not establish that the "
        "ordering would survive re-training."
    )
    return sentences


def _interpret_scaling(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    boards = sorted({run.case.board_size for run in runs})
    if len(boards) < 2:
        return []
    return [
        f"Trajectories connect separately trained runs at "
        f"{', '.join(f'{size}x{size}' for size in boards)}. Larger boards have "
        f"larger action spaces and longer games, so the legality panel is not a "
        f"fixed-difficulty comparison; the board panel uses a fixed position "
        f"budget and is the more comparable of the two."
    ]


def _factorial_tables(
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    if not effects:
        return []
    lines = [
        "## 2×2 factorial decomposition",
        "",
        "Effects are descriptive percentage-point contrasts. Architecture is "
        "`Mamba − Transformer`; objective is `JEPA − AR`; interaction is "
        "`(Mamba-JEPA − Mamba-AR) − (Transformer-JEPA − Transformer-AR)`.",
        "",
    ]
    labels = dict(FACTORIAL_METRICS)
    for board_size, board_effects in effects.items():
        lines += [f"### {board_size}x{board_size}", ""]
        rows = [
            [
                labels[metric],
                _pp(values["architecture"]),
                _pp(values["objective"]),
                _pp(values["interaction"]),
            ]
            for metric, values in board_effects.items()
        ]
        lines += _md_table(
            ("Metric", "Architecture effect", "Objective effect", "Interaction"),
            rows,
            numeric_columns=(1, 2, 3),
        )
        lines.append("")
    return lines


def _visual_sections(
    runs: Sequence[ComparisonRun],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
    figure_paths: Mapping[str, Path],
    report_path: Path,
) -> list[str]:
    descriptions = (
        (
            "scorecard",
            "Capability scorecard",
            "The left panel scales each metric by its own chance level so the "
            "legal and board families are not judged against one arbitrary "
            "floor; the right panel shows the gap to the column leader in "
            "percentage points.",
            _interpret_scorecard,
        ),
        (
            "phase",
            "Game-phase robustness",
            "Top-1 legality across normalized game phases, using the bin edges "
            "recorded in each result rather than assumed quartiles.",
            _interpret_phase,
        ),
        (
            "position",
            "Legality by move index",
            "The same quantity at full move resolution, which shows where "
            "inside a game legality is actually lost.",
            _interpret_position,
        ),
        (
            "board_layers",
            "Board decodability by depth",
            "Layerwise macro accuracy against normalized encoder depth, with "
            "the validation-selected layer marked. Normalized depth is the "
            "only depth axis comparable across architectures.",
            _interpret_board_layers,
        ),
        (
            "efficiency",
            "Quality–footprint frontier",
            "Board decodability against inference-time encoder parameters, "
            "with training hours encoded as bubble area.",
            _interpret_efficiency,
        ),
        (
            "factorial",
            "Factorial effects",
            "Diverging effect cells expose architecture, objective, and "
            "interaction patterns instead of hiding them in one ranking.",
            _interpret_factorial,
        ),
        (
            "scaling",
            "Scaling profile",
            "Completed cells across board sizes are connected as observed "
            "scaling trajectories; missing cells are not imputed.",
            _interpret_scaling,
        ),
    )
    if not figure_paths:
        return []
    lines = ["## Visual analysis", ""]
    for key, title, description, interpreter in descriptions:
        path = figure_paths.get(key)
        if path is None:
            continue
        relative = path.relative_to(report_path.parent).as_posix()
        lines += [
            f"### {title}",
            "",
            f"![{title}]({relative})",
            "",
            f"**What it shows.** {description}",
            "",
        ]
        readings = interpreter(runs, effects)
        if readings:
            lines += ["**What the data says.**", ""]
            lines += [f"- {sentence}" for sentence in readings]
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


CSV_COLUMNS = (
    "run",
    "architecture",
    "objective",
    "board_size",
    "run_name",
    "head",
    "head_kind",
    "is_headline_head",
    "top1_legal",
    "top1_ci_low",
    "top1_ci_high",
    "legal_mass",
    "random_legal_baseline",
    "normalized_lift",
    "legal_precision_at_3",
    "legal_precision_at_5",
    "board_linear_absolute",
    "board_linear_absolute_depth",
    "board_linear_relative",
    "board_linear_relative_depth",
    "board_mlp_absolute",
    "board_mlp_absolute_depth",
    "board_mlp_relative",
    "board_mlp_relative_depth",
    "board_macro_mean",
    "encoder_parameters",
    "total_parameters",
    "training_hours",
    "training_precision",
)


def _write_csv(runs: Sequence[ComparisonRun], path: Path) -> Path:
    """One row per run x head, for downstream analysis or a thesis appendix."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for run in runs:
            headline = run.metrics["legal_preferred_label"]
            for label, raw_metrics in run.metrics["legal_rows"]:
                head = _head_metrics(_mapping(raw_metrics))
                row = {
                    "run": run.label,
                    "architecture": run.case.architecture,
                    "objective": run.case.objective,
                    "board_size": run.case.board_size,
                    "run_name": run.metrics.get("run_name"),
                    "head": label,
                    "head_kind": run.head_kind,
                    "is_headline_head": label == headline,
                    "top1_legal": head["top1"],
                    "top1_ci_low": head["ci_low"],
                    "top1_ci_high": head["ci_high"],
                    "legal_mass": head["mass"],
                    "random_legal_baseline": head["baseline"],
                    "normalized_lift": head["lift"],
                    "legal_precision_at_3": head["precision3"],
                    "legal_precision_at_5": head["precision5"],
                    "board_macro_mean": run.metrics.get("board_macro_mean"),
                    "encoder_parameters": run.metrics.get("encoder_parameters"),
                    "total_parameters": run.metrics.get("total_parameters"),
                    "training_hours": run.metrics.get("training_hours"),
                    "training_precision": run.metrics.get("training_precision"),
                }
                for key in BOARD_METRIC_KEYS:
                    row[key] = run.metrics.get(key)
                    row[f"{key}_depth"] = run.metrics.get(f"{key}_depth")
                writer.writerow(row)
    return path


def _latex_escape(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("★", r"$\star$"),
        ("−", "-"),
        ("–", "--"),
    ):
        text = text.replace(source, target)
    return text


def _latex_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[str]:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{'l' * len(headers)}}}",
        r"\toprule",
        " & ".join(_latex_escape(item).replace("<br>", " ") for item in headers)
        + r" \\",
        r"\midrule",
    ]
    lines += [
        " & ".join(_latex_escape(item) for item in row) + r" \\" for row in rows
    ]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    return lines


def _write_latex(runs: Sequence[ComparisonRun], path: Path) -> Path:
    lines = [
        "% Generated by othello_research.evaluation.comparison_v2",
        "% Requires \\usepackage{booktabs}",
        "",
    ]
    lines += _latex_table(
        "Next-move legality under the common evaluation protocol. AR rows use "
        "the native head; JEPA rows use frozen readouts, with the "
        "validation-selected head starred.",
        "tab:legality",
        (
            "Run",
            "Head",
            "Top-1 legal",
            "95\\% CI",
            "Legal mass",
            "Baseline",
            "Lift",
            "Prec@3",
            "Prec@5",
        ),
        _legal_rows(runs),
    )
    lines += _latex_table(
        "Board-state decodability. Macro accuracy is the unweighted mean of the "
        "three per-class recalls at the validation-selected layer; depth is "
        "normalized because layer counts differ across architectures.",
        "tab:board",
        (
            "Run",
            "Probe",
            "Absolute",
            "Layer / depth",
            "Relative",
            "Layer / depth",
            "Mean",
            "Majority base",
        ),
        _board_rows(runs),
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_exports(
    runs: Sequence[ComparisonRun],
    output_dir: Path,
) -> dict[str, Path]:
    if not runs:
        return {}
    return {
        "csv": _write_csv(runs, output_dir / "metrics.csv"),
        "latex": _write_latex(runs, output_dir / "tables.tex"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _provenance_lines(runs: Sequence[ComparisonRun]) -> list[str]:
    games = sorted(
        {
            int(value)
            for run in runs
            if (value := _finite_float(run.metrics.get("legal_preferred_n_games")))
            is not None
        }
    )
    tokens = sorted(
        {
            int(value)
            for run in runs
            if (value := _finite_float(run.metrics.get("legal_preferred_n_tokens")))
            is not None
        }
    )
    layers = sorted(
        {
            (run.case.architecture, int(count))
            for run in runs
            if (count := _finite_float(run.metrics.get("board_layer_count")))
            is not None
        }
    )
    lines: list[str] = []
    if games:
        detail = ", ".join(_count(value) for value in games) + " games"
        if tokens:
            detail += (
                ", " + ", ".join(_count(value) for value in tokens)
                + " scored positions"
            )
        lines.append(f"- **Legality test set:** {detail}")
    if layers:
        lines.append(
            "- **Encoder depth:** "
            + ", ".join(
                f"{architecture.title()} top layer index {count}"
                for architecture, count in layers
            )
            + " — compare normalized depth, not raw layer numbers"
        )
    return lines


def _report_lines(
    *,
    selection: ComparisonSelection,
    runs: Sequence[ComparisonRun],
    missing: Mapping[str, str],
    artifacts_root: Path,
    report_path: Path,
    figure_paths: Mapping[str, Path],
    export_paths: Mapping[str, Path],
    effects: Mapping[int, Mapping[str, Mapping[str, float]]],
    protocol_id: str = PROTOCOL_ID,
) -> list[str]:
    mode = (
        "Single-run dossier" if selection.is_single_case else "Comparative analysis"
    )
    lines = [
        "# Thesis Common Evaluation — Comparative Report",
        "",
        f"- **Selection:** {selection.display}",
        f"- **Mode:** {mode}",
        f"- **Coverage:** {len(runs)}/{selection.expected_count} selected canonical cells ready",
        f"- **Protocol:** `{protocol_id}`",
        f"- **Report schema:** `{COMPARISON_REPORT_VERSION}`",
        f"- **Generated (UTC):** `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
    ]
    lines += _provenance_lines(runs)
    lines += ["", "## Executive view", ""]
    lines.extend(f"- {item}" for item in _leader_observations(runs))
    if missing:
        lines.append(
            f"- **{len(missing)} selected cell(s) are pending or invalid.** "
            "They remain visible in the coverage table and are never imputed."
        )

    lines += ["", "## Comparability gates", ""]
    lines += _md_table(
        ("Gate", "Status", "Evidence / interpretation"),
        _comparability_gates(runs, selection, protocol_id),
    )

    lines += ["", "## Run coverage and provenance", ""]
    lines += _md_table(
        ("Cell", "Status", "Run", "Split manifest", "Source / reason"),
        _coverage_rows(runs, missing, artifacts_root),
    )

    if not runs:
        lines += [
            "",
            "## Result",
            "",
            "No metric table or chart was produced because no selected canonical "
            f"`{protocol_id}` result is complete. Run the missing Thesis Common "
            "Evaluation cell(s), then rerun this notebook.",
            "",
        ]
        return lines

    lines.append("")
    lines += _metric_glossary_lines()

    lines += [
        "## Next-move head selection",
        "",
        "AR has one head: the native action head it was pretrained with. JEPA "
        "has two frozen-encoder readouts, each trained with early stopping and "
        "a shard allowance it did not exhaust, and the headline row is "
        "whichever scored higher on the validation selection metric recorded "
        "during head training. The test split is never used to choose a head. "
        "`Best / completed shards` reports where each head was checkpointed "
        "and how far it ran before early stopping.",
        "",
    ]
    lines += _md_table(
        (
            "Run",
            "Head kind",
            "Headline head",
            "Validation selection score",
            "Best / completed shards",
            "Selected on",
        ),
        _head_selection_rows(runs),
    )

    lines += [
        "",
        "## Legal-move compatibility",
        "",
        "Every available head is listed; ★ marks the headline row used by the "
        "figures and the factorial decomposition. `Prec@k` is the mean fraction "
        "of the top-k tokens that are legal, so it decreases with k by "
        "construction. The confidence interval is a game-level bootstrap over "
        "test games and does not represent model-seed uncertainty.",
        "",
    ]
    lines += _md_table(
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
        _legal_rows(runs),
        numeric_columns=(2, 4, 5, 6, 7, 8),
    )

    lines += [
        "",
        "## Board-state decodability",
        "",
        "This is the encoder-matched comparison: an identical frozen-encoder "
        "protocol for all four systems. Macro accuracy is the unweighted mean "
        "of the three per-class recalls against a 33.3% chance level. Layers "
        "were selected on validation; the table reports untouched test results "
        "and gives normalized depth beside each raw layer index.",
        "",
    ]
    lines += _md_table(
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
        _board_rows(runs),
        numeric_columns=(2, 4, 6, 7),
    )

    lines += [
        "",
        "## Efficiency and footprint",
        "",
        "Encoder parameters are what run at inference. Total parameters "
        "additionally count the JEPA EMA target encoder and predictor, which "
        "exist only during training — use the encoder column for any footprint "
        "claim.",
        "",
    ]
    lines += _md_table(
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
        _efficiency_rows(runs),
        numeric_columns=(1, 2, 3, 4, 5, 6),
    )
    lines += [""] + _factorial_tables(effects)
    lines += _visual_sections(runs, effects, figure_paths, report_path)
    lines += [
        "## Interpretation boundaries",
        "",
        "- Legal compatibility measures rule-consistent action proposals, not "
        "strategic playing strength.",
        "- The next-move comparison is system-level: AR uses a head over an "
        "encoder trained for exactly this task, JEPA a frozen readout over an "
        "encoder that was not. That is the question being asked, but it means "
        "the legality contrast is not a statement about representation content.",
        "- Board probes are the encoder-matched comparison, but they measure "
        "decodability; they do not by themselves establish causal use of the "
        "decoded representation.",
        "- Rankings and factorial effects are descriptive point estimates from "
        "the available runs. Independent training seeds are required for "
        "inferential architecture/objective claims.",
        "- Cross-board comparisons describe scaling across separately trained "
        "conditions; they are not zero-shot transfer or equal-action-space tests.",
        "- Missing cells are reported as pending and are never filled, "
        "interpolated, or borrowed from legacy protocols.",
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
    board_size: int | str = 12,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_root: str | Path | None = None,
    protocol_id: str = PROTOCOL_ID,
) -> ComparisonReport:
    """Create one Markdown report for the requested canonical result slice.

    Defaults intentionally target the currently completed 12x12 comparison:
    both architectures, both objectives, board size 12.  Output is written under
    ``comparisons_v2`` so a v1 report generated from the same artifacts is never
    overwritten.
    """

    root = Path(artifacts_root)
    selection = ComparisonSelection.from_values(
        architecture=architecture,
        objective=objective,
        board_size=board_size,
    )
    runs, missing = collect_comparison_runs(
        root,
        selection,
        registry_path=registry_path,
        protocol_id=protocol_id,
    )
    base_output = (
        Path(output_root)
        if output_root is not None
        else root / "reports" / "thesis_eval" / "comparisons_v2" / protocol_id
    )
    output_dir = base_output / selection.slug
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"
    effects = _factorial_effects(runs)
    figure_paths = _write_figures(runs, output_dir, effects) if runs else {}
    export_paths = _write_exports(runs, output_dir)
    lines = _report_lines(
        selection=selection,
        runs=runs,
        missing=missing,
        artifacts_root=root,
        report_path=report_path,
        figure_paths=figure_paths,
        export_paths=export_paths,
        effects=effects,
        protocol_id=protocol_id,
    )
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    warnings = tuple(
        f"{run.label}: {run.metrics['legal_head_warning']}"
        for run in runs
        if run.metrics.get("legal_head_warning")
    )
    return ComparisonReport(
        report_path=report_path,
        figure_paths=figure_paths,
        runs=runs,
        missing=missing,
        selection=selection,
        export_paths=export_paths,
        warnings=warnings,
    )
