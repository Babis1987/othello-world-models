"""Common proposal-aligned evaluation entry point.

This module adds experiment discovery, strict checkpoint identity checks,
architecture-matched random controls, proposal-oriented reporting, and 2x2
factorial aggregation around :mod:`othello_research.evaluation.unified`.
Metric implementations remain in the shared unified evaluator so notebooks
cannot drift across architecture, objective, or board size.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import platform
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from othello_thesis.evaluation.unified import (
    PROTOCOL_ID,
    UnifiedEvalConfig,
    UnifiedEvaluator,
    build_evaluation_split,
    file_sha256,
    seed_everything,
)


SUITE_ID = "thesis_eval_suite_v1"
from othello_thesis.evaluation._thesis_registry import (
    ARCHITECTURES,
    BOARD_SIZES,
    DEFAULT_REGISTRY,
    OBJECTIVES,
    EvaluationCase,
    ModelNotReadyError,
    PreparedEvaluation,
    _architecture_from_metadata,
    _checkpoint_architecture,
    _checkpoint_config,
    _metadata_matches_case,
    _nested_mapping,
    _read_run_metadata,
    _train_config,
    load_registry,
    normalize_architecture,
    normalize_board_size,
    normalize_objective,
    resolve_case,
    resolve_data_dir,
    resolve_run_dir,
    validate_checkpoint_identity,
)


def _load_manifest(
    data_dir: Path,
    split: Any,
) -> tuple[dict[str, Any], int, str, str | None]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Authoritative thesis evaluation requires {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = dict(payload.get("shard_counts") or {})
    shard_hashes: dict[str, str] = {}
    for shard in payload.get("shards") or ():
        name = str(shard["filename"])
        count = int(shard["count"])
        if name in counts and int(counts[name]) != count:
            raise ValueError(f"Manifest count disagreement for {name}")
        counts[name] = count
        if shard.get("sha256"):
            shard_hashes[name] = str(shard["sha256"])
    all_paths = (
        tuple(split.pretraining)
        + tuple(split.downstream_train)
        + tuple(split.selection)
        + tuple(split.test)
    )
    missing = [path.name for path in all_paths if path.name not in counts]
    if missing:
        raise ValueError(f"Manifest lacks shard counts: {missing[:3]}")
    expected_pretrain_games = sum(
        int(counts[path.name]) for path in split.pretraining
    )
    content_sha256: str | None = None
    if all(path.name in shard_hashes for path in all_paths):
        digest = hashlib.sha256()
        for path in all_paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(shard_hashes[path.name].encode("ascii"))
        content_sha256 = digest.hexdigest()
    return (
        payload,
        expected_pretrain_games,
        file_sha256(manifest_path),
        content_sha256,
    )


def _project_source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    source_root = project_root / "src" / "othello_thesis"
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_from_checkpoint(
    case: EvaluationCase,
    checkpoint: dict[str, Any],
    *,
    load_weights: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, Any]:
    if case.objective == "jepa":
        from othello_thesis.training.jepa import (
            build_model_from_checkpoint,
        )

        model, model_config = build_model_from_checkpoint(checkpoint)
        if load_weights:
            model.load_state_dict(checkpoint["model"])
        return model, model.context_encoder, model_config

    if case.architecture == "transformer":
        from othello_thesis.models.transformer import GPTConfig, OthelloGPT

        config_data = _checkpoint_config(checkpoint)
        if not config_data:
            raise KeyError("Transformer-AR checkpoint has no model_config")
        model_config = GPTConfig(**config_data)
        model = OthelloGPT(model_config)
        if load_weights:
            model.load_state_dict(checkpoint["model"])
        return model, model, model_config

    from othello_thesis.models.mamba import MambaARConfig, OthelloMambaAR

    config_data = _checkpoint_config(checkpoint)
    if not config_data:
        config_data = _train_config(checkpoint)
    filtered = {
        key: value
        for key, value in config_data.items()
        if key in MambaARConfig.__dataclass_fields__
    }
    model_config = MambaARConfig(**filtered)
    model = OthelloMambaAR(model_config)
    if load_weights:
        model.load_state_dict(checkpoint["model"])
    return model, model, model_config


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _latest_training_attempt(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    if not rows:
        return rows, 0
    start = 0
    previous = rows[0]
    for index, current in enumerate(rows[1:], start=1):
        current_chunk = int(_finite_float(current.get("chunk_count")))
        previous_chunk = int(_finite_float(previous.get("chunk_count")))
        if current_chunk and current_chunk <= previous_chunk:
            start = index
        previous = current
    return rows[start:], start


def summarize_training_metrics(run_dir: str | Path) -> dict[str, Any] | None:
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.is_file():
        return None
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows:
        return None
    rows, discarded = _latest_training_attempt(all_rows)
    timed = [row for row in rows if _finite_float(row.get("dt_seconds")) > 0]
    train_only = [
        row
        for row in timed
        if int(_finite_float(row.get("val_games"))) == 0
    ] or timed
    times = [_finite_float(row["dt_seconds"]) for row in train_only]
    game_rates = [
        _finite_float(row.get("chunk_games"))
        / _finite_float(row.get("dt_seconds"))
        for row in train_only
        if _finite_float(row.get("chunk_games")) > 0
    ]
    unit_rates = [
        _finite_float(row.get("chunk_tokens"))
        / _finite_float(row.get("dt_seconds"))
        for row in train_only
        if _finite_float(row.get("chunk_tokens")) > 0
    ]
    final = rows[-1]
    logical_chunks = int(_finite_float(final.get("chunk_count"), len(rows)))
    median_seconds = statistics.median(times) if times else float("nan")
    return {
        "metrics_path": str(metrics_path),
        "chunks": logical_chunks,
        "discarded_prior_attempt_rows": discarded,
        "games_seen": int(_finite_float(final.get("games_seen"))),
        "supervision_units_seen": int(_finite_float(final.get("tokens_seen"))),
        "median_training_chunk_seconds": median_seconds,
        "median_games_per_second": (
            statistics.median(game_rates) if game_rates else float("nan")
        ),
        "median_supervision_units_per_second": (
            statistics.median(unit_rates) if unit_rates else float("nan")
        ),
        "estimated_training_hours": (
            median_seconds * logical_chunks / 3600.0
            if math.isfinite(median_seconds)
            else float("nan")
        ),
        "observed_train_plus_validation_loop_hours": sum(
            _finite_float(row.get("dt_seconds")) for row in timed
        )
        / 3600.0,
        "timing_scope": (
            "dt_seconds excludes normal checkpoint/Drive-sync time; validation "
            "is included only in observed_train_plus_validation_loop_hours"
        ),
    }


def prepare_evaluation(
    architecture: str,
    objective: str,
    board_size: int | str,
    *,
    artifacts_root: str | Path,
    project_root: str | Path,
    device: str | torch.device = "cuda:0",
    local_checkpoint_root: str | Path = "/content/unified_eval_checkpoints",
    registry_path: str | Path = DEFAULT_REGISTRY,
    overwrite_incompatible: bool = False,
    allow_source_drift: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
    output_subdir: str | Path = Path("thesis_eval") / "final",
) -> PreparedEvaluation:
    case = resolve_case(
        architecture,
        objective,
        board_size,
        registry_path=registry_path,
    )
    artifacts_root = Path(artifacts_root)
    project_root = Path(project_root)
    data_dir = resolve_data_dir(case, artifacts_root)
    run_dir = resolve_run_dir(case, artifacts_root)
    drive_checkpoint = run_dir / "final.pt"

    protocol_config = UnifiedEvalConfig(
        board_size=case.board_size,
        **dict(config_overrides or {}),
    )
    all_chunks = sorted(data_dir.glob("*.pickle"), key=lambda path: path.name)
    split = build_evaluation_split(all_chunks, protocol_config)
    (
        _manifest,
        expected_pretrain_games,
        manifest_sha256,
        split_content_sha256,
    ) = _load_manifest(data_dir, split)

    local_dir = Path(local_checkpoint_root) / case.key
    local_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoint = local_dir / "final.pt"
    shutil.copy2(drive_checkpoint, local_checkpoint)
    checkpoint_sha256 = file_sha256(local_checkpoint)
    checkpoint = torch.load(
        local_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint_identity(case, checkpoint)

    train_config = _train_config(checkpoint)
    games_seen = int(checkpoint.get("games_seen", -1))
    chunk_count = int(checkpoint.get("chunk_count", -1))
    if games_seen != expected_pretrain_games:
        raise ValueError(
            f"Checkpoint games_seen={games_seen:,}, manifest pretraining budget="
            f"{expected_pretrain_games:,}"
        )
    if chunk_count != protocol_config.pretraining_shards:
        raise ValueError(
            f"Checkpoint chunk_count={chunk_count}, expected "
            f"{protocol_config.pretraining_shards}"
        )
    if int(train_config.get("passes_over_data", -1)) != 1:
        raise ValueError("Headline checkpoint must use one pass over the corpus")

    seed_everything(protocol_config.seed)
    primary_model, encoder, model_config = _model_from_checkpoint(
        case,
        checkpoint,
        load_weights=True,
    )
    primary_model.to(device).eval()
    encoder = primary_model if case.objective == "ar" else primary_model.context_encoder

    training_precision = (
        train_config.get("precision")
        or train_config.get("amp_dtype")
        or train_config.get("mixed_precision")
        or checkpoint.get("precision")
    )
    training_metrics = summarize_training_metrics(run_dir)
    metadata = {
        "suite": SUITE_ID,
        "run_name": run_dir.name,
        "architecture": case.architecture,
        "objective": (
            "ar"
            if case.objective == "ar"
            else "jepa_v5_infonce_hard_disjoint_all_position"
        ),
        "position_sampling": (
            None if case.objective == "ar" else "all"
        ),
        "board_size": case.board_size,
        "checkpoint_name": "final.pt",
        "checkpoint_drive_path": str(drive_checkpoint),
        "checkpoint_local_path": str(local_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": local_checkpoint.stat().st_size,
        "data_manifest_path": str(data_dir / "manifest.json"),
        "data_manifest_sha256": manifest_sha256,
        "split_content_sha256": split_content_sha256,
        "project_source_sha256": _project_source_sha256(project_root),
        "step": checkpoint.get("step"),
        "games_seen": games_seen,
        "chunk_count": chunk_count,
        "expected_pretrain_games": expected_pretrain_games,
        "training_precision": training_precision,
        "model_config": (
            asdict(model_config)
            if hasattr(model_config, "__dataclass_fields__")
            else dict(vars(model_config))
        ),
        "train_config": train_config,
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "total_parameters": sum(
            parameter.numel() for parameter in primary_model.parameters()
        ),
        "mamba_backend": getattr(model_config, "mamba_backend", None),
        "training_hardware": checkpoint.get("training_hardware"),
        "training_python": checkpoint.get("python_version"),
        "training_metrics": training_metrics,
    }
    output_subdir = Path(output_subdir)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise ValueError("output_subdir must stay inside the selected run directory")
    output_dir = run_dir / output_subdir
    evaluator = UnifiedEvaluator(
        encoder=encoder,
        native_ar_model=primary_model if case.objective == "ar" else None,
        split=split,
        config=protocol_config,
        device=device,
        output_dir=output_dir,
        run_metadata=metadata,
        overwrite_incompatible=overwrite_incompatible,
        allow_source_drift=allow_source_drift,
    )
    return PreparedEvaluation(
        case=case,
        artifacts_root=artifacts_root,
        data_dir=data_dir,
        run_dir=run_dir,
        checkpoint_path=local_checkpoint,
        checkpoint=checkpoint,
        evaluator=evaluator,
        primary_model=primary_model,
        output_dir=output_dir,
        training_metrics=training_metrics,
    )


def run_random_board_control(
    prepared: PreparedEvaluation,
    *,
    overwrite_incompatible: bool = False,
    allow_source_drift: bool = False,
) -> Path:
    """Evaluate frozen heads and board probes on a matched random encoder."""
    if prepared.case.board_size != 8:
        raise ValueError("Random-encoder controls are restricted to the 8x8 study")
    seed = prepared.evaluator.config.seed
    seed_everything(seed)
    random_model, random_encoder, model_config = _model_from_checkpoint(
        prepared.case,
        prepared.checkpoint,
        load_weights=False,
    )
    digest = hashlib.sha256(
        (
            str(prepared.evaluator.results["metadata"]["checkpoint_sha256"])
            + f":random_encoder:{seed}"
        ).encode("utf-8")
    ).hexdigest()
    primary_metadata = prepared.evaluator.results["metadata"]
    random_metadata = {
        **{
            key: value
            for key, value in primary_metadata.items()
            if key
            not in {
                "protocol",
                "split",
                "split_manifest_hash",
                "source_drift_events",
            }
        },
        "run_name": f"{prepared.run_dir.name}__random_encoder_control",
        "objective": "architecture_matched_random_encoder_control",
        "checkpoint_sha256": digest,
        "random_control_for_checkpoint_sha256": primary_metadata[
            "checkpoint_sha256"
        ],
        "random_seed": seed,
        "step": 0,
        "games_seen": 0,
        "chunk_count": 0,
        "encoder_parameters": sum(
            parameter.numel() for parameter in random_encoder.parameters()
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in random_model.parameters()
        ),
        "model_config": (
            asdict(model_config)
            if hasattr(model_config, "__dataclass_fields__")
            else dict(vars(model_config))
        ),
    }
    output_dir = prepared.output_dir / "random_encoder_control"
    random_evaluator = UnifiedEvaluator(
        encoder=random_encoder,
        split=prepared.evaluator.split,
        config=prepared.evaluator.config,
        device=prepared.evaluator.device,
        output_dir=output_dir,
        run_metadata=random_metadata,
        overwrite_incompatible=overwrite_incompatible,
        allow_source_drift=allow_source_drift,
    )
    random_evaluator.run_frozen_head("linear")
    random_evaluator.run_frozen_head("mlp")
    random_evaluator.run_board_probe("linear")
    random_evaluator.run_board_probe("mlp")
    random_evaluator.write_summary()
    results_path = random_evaluator.results_path
    del random_evaluator, random_model, random_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results_path


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def _topk(metrics: Mapping[str, Any], k: int) -> float:
    values = metrics["topk_legal"]
    return float(values.get(str(k), values.get(k)))


def _selected_metric(
    results: Mapping[str, Any],
    probe_type: str,
    mode: str,
) -> Mapping[str, Any]:
    return results["board_state"][probe_type]["selected"][mode]["test"]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _legal_sources_for_case(
    prepared: PreparedEvaluation,
    results: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Return only the objective-appropriate next-move readouts."""
    if prepared.case.objective == "ar":
        native = results.get("native_ar")
        if not native:
            raise RuntimeError("AR evaluation requires the native AR stage")
        return [("Native AR", native)]
    sources: list[tuple[str, Mapping[str, Any]]] = []
    for head_type in ("linear", "mlp"):
        saved = results.get("frozen_next_move", {}).get(head_type)
        if saved:
            sources.append(
                (f"Frozen {head_type.upper()}", saved["test_legal"])
            )
    if len(sources) != 2:
        raise RuntimeError("JEPA evaluation requires frozen Linear and MLP heads")
    return sources


def write_evaluation_figures(
    prepared: PreparedEvaluation,
    results: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the two compact, position-resolved headline figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_paths: dict[str, Path] = {}
    legal_sources = _legal_sources_for_case(prepared, results)
    legal_path = prepared.output_dir / f"legal_accuracy_by_move__{SUITE_ID}.png"
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    baseline_drawn = False
    vocab_size = int(
        results["metadata"].get("model_config", {}).get(
            "vocab_size", prepared.case.board_size**2 - 3
        )
    )
    for label, metrics in legal_sources:
        curve = [
            row
            for row in metrics.get("position_curve", [])
            if int(row.get("n_tokens", 0)) > 0
            and "top1_legal" in row
        ]
        x = [int(row["position"]) + 1 for row in curve]
        y = [100.0 * float(row["top1_legal"]) for row in curve]
        axis.plot(x, y, linewidth=2.0, label=label)
        if not baseline_drawn:
            baseline = [
                100.0 * float(row["mean_legal_moves"]) / vocab_size
                for row in curve
            ]
            axis.plot(
                x,
                baseline,
                color="0.45",
                linestyle="--",
                linewidth=1.4,
                label="Random-vocabulary legality",
            )
            baseline_drawn = True
    axis.set(
        xlabel="Moves observed before predicting the next move",
        ylabel="Top-1 legal accuracy (%)",
        title=f"{prepared.case.label}: legal accuracy by move",
        ylim=(0.0, 101.0),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(legal_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["legal_accuracy_by_move"] = legal_path

    board_path = prepared.output_dir / f"board_accuracy_by_move__{SUITE_ID}.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True, sharey=True)
    combinations = (
        ("linear", "absolute"),
        ("linear", "relative"),
        ("mlp", "absolute"),
        ("mlp", "relative"),
    )
    missing_position_curves: list[str] = []
    for axis, (probe_type, mode) in zip(axes.flat, combinations):
        probe = results["board_state"][probe_type]
        selected = probe["selected"][mode]
        layer = int(selected["layer"])
        per_position = selected["test"].get("per_position")
        if not per_position:
            missing_position_curves.append(f"{probe_type}/{mode}")
            axis.text(
                0.5,
                0.5,
                "Per-move data unavailable\n(re-run board probe once)",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            rows = sorted(
                (
                    (int(position), metrics)
                    for position, metrics in per_position.items()
                    if int(metrics.get("n_labels", 0)) > 0
                ),
                key=lambda item: item[0],
            )
            axis.plot(
                [position for position, _ in rows],
                [100.0 * float(metrics["accuracy"]) for _, metrics in rows],
                linewidth=1.8,
                color="#2a6fbb" if probe_type == "linear" else "#c44e52",
                label="Overall",
            )
            axis.plot(
                [position for position, _ in rows],
                [
                    100.0 * float(metrics["macro_balanced_accuracy"])
                    for _, metrics in rows
                ],
                linewidth=1.5,
                linestyle="--",
                color="#2a6fbb" if probe_type == "linear" else "#c44e52",
                alpha=0.85,
                label="Macro",
            )
            axis.legend(frameon=False, fontsize=8)
        axis.set_title(f"{probe_type.upper()} {mode} — selected L{layer}")
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Moves observed (probe position t)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Board-square accuracy (%)")
    fig.suptitle(
        f"{prepared.case.label}: board accuracy by move at validation-selected layers",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(board_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["board_accuracy_by_move"] = board_path

    if prepared.case.objective == "jepa":
        saturation_path = (
            prepared.output_dir / f"head_saturation__{SUITE_ID}.png"
        )
        fig, axis = plt.subplots(figsize=(8.0, 4.6))
        for head_type, color in (("linear", "#4c78a8"), ("mlp", "#e45756")):
            history = results["frozen_next_move"][head_type]["train"]["history"]
            curve = [row for row in history if "selection" in row]
            axis.plot(
                [float(row["cumulative_games"]) / 1_000_000 for row in curve],
                [
                    100.0
                    * float(row["selection"]["legal_probability_mass"])
                    for row in curve
                ],
                marker="o",
                markersize=3,
                linewidth=1.7,
                color=color,
                label=head_type.upper(),
            )
        axis.set(
            xlabel="Frozen-head training games (millions)",
            ylabel="Selection legal probability mass (%)",
            title=f"{prepared.case.label}: frozen-readout saturation",
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(saturation_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_paths["head_saturation"] = saturation_path
    return figure_paths


def write_proposal_report(
    prepared: PreparedEvaluation,
    *,
    random_results_path: str | Path | None = None,
) -> Path:
    """Write the complete per-model report promised by the proposal."""
    prepared.evaluator.write_summary()
    results = json.loads(
        prepared.evaluator.results_path.read_text(encoding="utf-8")
    )
    figure_paths = write_evaluation_figures(prepared, results)
    random_results = None
    if random_results_path is not None:
        random_results = json.loads(
            Path(random_results_path).read_text(encoding="utf-8")
        )
    metadata = results["metadata"]
    lines = [
        f"# Proposal evaluation: {prepared.case.label}",
        "",
        f"- **Suite:** `{SUITE_ID}` over `{PROTOCOL_ID}`",
        f"- **Run:** `{metadata['run_name']}`",
        f"- **Checkpoint:** `{metadata['checkpoint_name']}` "
        f"(`{metadata['checkpoint_sha256']}`)",
        f"- **Training budget:** `{int(metadata['games_seen']):,}` games / "
        f"`{int(metadata['chunk_count'])}` shards",
        "- **Next-move readout:** "
        + (
            "native pretrained AR head"
            if prepared.case.objective == "ar"
            else "frozen bf16 encoder with common fp32 Linear/MLP readouts"
        ),
        "- **Exact continuation:** intentionally excluded because corpus "
        "continuations are stochastic legal choices",
        "",
        "## 1. Functional legality",
        "",
        "| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | "
        "Legal probability mass |",
        "|---|---:|---:|---:|---:|",
    ]
    legal_sources = _legal_sources_for_case(prepared, results)
    for label, metrics in legal_sources:
        lines.append(
            f"| {label} | {_pct(metrics['top1_legal_per_token'])} | "
            f"{_pct(_topk(metrics, 3))} | {_pct(_topk(metrics, 5))} | "
            f"{_pct(metrics['legal_prob_mass_per_token'])} |"
        )

    lines += [
        "",
        "### Statistical context",
        "",
        "| Readout | Normalized legal lift | 95% CI for top-1 legal |",
        "|---|---:|---:|",
    ]
    for label, metrics in legal_sources:
        ci = metrics["bootstrap_ci_95_per_game"]["top1_legal"]
        lines.append(
            f"| {label} | {_pct(metrics['top1_legal_normalized_lift'])} | "
            f"{_pct(ci['low'])}–{_pct(ci['high'])} |"
        )

    if "head_saturation" in figure_paths:
        lines += [
            "",
            "### Frozen-head saturation",
            "",
            "There is no minimum shard count. Training stops under the "
            "pre-registered selection legal-mass patience rule and restores "
            "the best checkpoint.",
            "",
            f"![Frozen-head saturation]({figure_paths['head_saturation'].name})",
        ]

    lines += [
        "",
        "### Legal accuracy by move",
        "",
        f"![Legal accuracy by move]({figure_paths['legal_accuracy_by_move'].name})",
        "",
        "### Legality by normalized game phase",
        "",
        "| Readout | Phase | Tokens | Mean legal moves | Top-1 legal | Legal mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in legal_sources:
        for phase in metrics["normalized_phase_bins"]:
            phase_label = (
                f"{float(phase['phase_start']):.2f}–"
                f"{float(phase['phase_end']):.2f}"
            )
            lines.append(
                f"| {label} | {phase_label} | {int(phase['n_tokens']):,} | "
                f"{float(phase['mean_legal_moves']):.2f} | "
                f"{_pct(phase['top1_legal'])} | "
                f"{_pct(phase['legal_probability_mass'])} |"
            )

    lines += [
        "",
        "## 2. Board-state representations",
        "",
        "Layer selection uses only the selection split. Headline test values are "
        "macro-balanced accuracy, so changing empty-square prevalence across "
        "board sizes cannot dominate the comparison.",
        "",
        "**Macro accuracy** is the unweighted mean of the three per-class "
        "recalls. Absolute labels use empty/black/white and relative labels use "
        "empty/mine/opponent. Each class therefore contributes one third even "
        "when empty squares are much more common. Overall accuracy instead "
        "weights every square equally and can be dominated by the empty class.",
        "",
        "| Probe | Labels | Best layer | Normalized depth | Test macro | Overall | "
        "Occupied | Empty |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for probe_type in ("linear", "mlp"):
        probe = results["board_state"][probe_type]
        for mode in ("absolute", "relative"):
            selected = probe["selected"][mode]
            layer = int(selected["layer"])
            test = selected["test"]
            lines.append(
                f"| {probe_type.upper()} | {mode} | L{layer} | "
                f"{float(probe['normalized_depth'][str(layer)]):.3f} | "
                f"{_pct(test['macro_balanced_accuracy'])} | "
                f"{_pct(test['accuracy'])} | "
                f"{_pct(test['occupied_accuracy'])} | "
                f"{_pct(test['empty_accuracy'])} |"
            )

    lines += [
        "",
        "### Probe accuracy by layer",
        "",
        "These are the untouched test-split tables in the same format as the "
        "8x8 unified JEPA notebook. `Selected` marks the layer chosen using "
        "the separate selection split, never the test results.",
    ]
    for probe_type in ("linear", "mlp"):
        probe = results["board_state"][probe_type]
        selected_layers = {
            mode: int(probe["selected"][mode]["layer"])
            for mode in ("absolute", "relative")
        }
        lines += [
            "",
            f"#### {probe_type.upper()} board-state probe",
            "",
            "| Layer | Absolute accuracy | Absolute macro | Relative accuracy | "
            "Relative macro | Selected |",
            "|---:|---:|---:|---:|---:|:---|",
        ]
        for layer in probe["layers"]:
            layer_key = str(layer)
            test = probe["test_metrics"][layer_key]
            selected_modes = " + ".join(
                mode
                for mode in ("absolute", "relative")
                if selected_layers[mode] == int(layer)
            )
            lines.append(
                f"| L{layer} | {_pct(test['absolute']['accuracy'])} | "
                f"{_pct(test['absolute']['macro_balanced_accuracy'])} | "
                f"{_pct(test['relative']['accuracy'])} | "
                f"{_pct(test['relative']['macro_balanced_accuracy'])} | "
                f"{selected_modes} |"
            )

    lines += [
        "",
        "### Board accuracy by move at each selected layer",
        "",
        f"![Board accuracy by move]({figure_paths['board_accuracy_by_move'].name})",
        "",
        "### Selected board probes by game phase",
        "",
        "| Probe | Labels | Phase | Macro accuracy | Occupied | Empty |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for probe_type in ("linear", "mlp"):
        for mode in ("absolute", "relative"):
            selected = results["board_state"][probe_type]["selected"][mode]
            for phase_key, phase in selected["test"]["per_phase_bin"].items():
                lines.append(
                    f"| {probe_type.upper()} | {mode} | {int(phase_key) + 1}/4 | "
                    f"{_pct(phase['macro_balanced_accuracy'])} | "
                    f"{_pct(phase['occupied_accuracy'])} | "
                    f"{_pct(phase['empty_accuracy'])} |"
                )

    section = 3
    intervention = results.get("causal_intervention")
    if intervention:
        lines += [
            "",
            f"## {section}. Nanda-style causal intervention",
            "",
            f"- Readout: `{intervention['readout']}`",
            f"- Selected intervention scale: "
            f"`{float(intervention['selected_alpha']):g}`",
            f"- Interpretation scope: {intervention['interpretation_scope']}",
            "",
            "| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |",
            "|---|---:|---:|---:|",
        ]
        for key, label in (
            ("null", "Null"),
            ("magnitude_matched_random", "Magnitude-matched random"),
            ("probe_direction", "Relative-board direction"),
        ):
            metrics = intervention["test"][key]
            lines.append(
                f"| {label} | {_pct(metrics['target_top1_legal'])} | "
                f"{_pct(metrics['target_legal_probability_mass'])} | "
                f"{float(metrics['mean_topn_false_positive_plus_false_negative']):.3f} |"
            )
        lines += [
            "",
            "For AR this edits the native prediction path. For JEPA it establishes "
            "causal steerability of the composed JEPA encoder plus its post-hoc "
            "frozen MLP readout; it is not evidence of a native JEPA action head.",
        ]
        section += 1

    if random_results is not None:
        lines += [
            "",
            f"## {section}. Architecture-matched random-encoder control",
            "",
            "### Frozen next-move readouts",
            "",
            "| Readout | Trained top-1 legal | Random top-1 legal | "
            "Trained legal mass | Random legal mass |",
            "|---|---:|---:|---:|---:|",
        ]
        for head_type in ("linear", "mlp"):
            trained_head = results.get("frozen_next_move", {}).get(head_type)
            random_head = random_results.get("frozen_next_move", {}).get(head_type)
            if not trained_head or not random_head:
                continue
            trained_legal = trained_head["test_legal"]
            random_legal = random_head["test_legal"]
            lines.append(
                f"| {head_type.upper()} | "
                f"{_pct(trained_legal['top1_legal_per_token'])} | "
                f"{_pct(random_legal['top1_legal_per_token'])} | "
                f"{_pct(trained_legal['legal_prob_mass_per_token'])} | "
                f"{_pct(random_legal['legal_prob_mass_per_token'])} |"
            )
        lines += [
            "",
            "### Board-state probes",
            "",
            "The lift compares independently selection-chosen trained and random "
            "layers under the same probe protocol and position manifest.",
            "",
            "| Probe | Labels | Trained macro | Random macro | Trained − random |",
            "|---|---|---:|---:|---:|",
        ]
        for probe_type in ("linear", "mlp"):
            for mode in ("absolute", "relative"):
                trained = float(
                    _selected_metric(results, probe_type, mode)[
                        "macro_balanced_accuracy"
                    ]
                )
                random_value = float(
                    _selected_metric(random_results, probe_type, mode)[
                        "macro_balanced_accuracy"
                    ]
                )
                lines.append(
                    f"| {probe_type.upper()} | {mode} | {_pct(trained)} | "
                    f"{_pct(random_value)} | {_pct(trained - random_value)} |"
                )
        section += 1

    lines += [
        "",
        f"## {section}. Efficiency and reproducibility",
        "",
        f"- Encoder parameters: `{int(metadata['encoder_parameters']):,}`",
        f"- Total parameters: `{int(metadata['total_parameters']):,}`",
        f"- Checkpoint bytes: `{int(metadata['checkpoint_bytes']):,}`",
        f"- Evaluation GPU: `{metadata.get('gpu') or 'unreported'}`",
        f"- Training hardware: `{metadata.get('training_hardware') or 'unreported'}`",
        f"- Training precision: `{metadata.get('training_precision') or 'unreported'}`",
        f"- Python / PyTorch / CUDA: `{platform.python_version()}` / "
        f"`{torch.__version__}` / `{torch.version.cuda}`",
    ]
    timing = prepared.training_metrics
    if timing:
        lines += [
            f"- Median training chunk: "
            f"`{_number(timing['median_training_chunk_seconds'])} s`",
            f"- Estimated training loop: "
            f"`{_number(timing['estimated_training_hours'])} h`",
            f"- Median games/s: "
            f"`{_number(timing['median_games_per_second'])}`",
            f"- Median supervision units/s: "
            f"`{_number(timing['median_supervision_units_per_second'])}`",
            f"- Timing scope: {timing['timing_scope']}",
        ]
    else:
        lines.append("- Training timing: `metrics.csv unavailable`")

    lines += [
        "",
        f"## {section + 1}. Interpretation boundary",
        "",
        "- Board probes establish decodability, not by themselves causal use.",
        "- Legal behavior measures rule compatibility, not strategic playing strength.",
        "- Bootstrap legality intervals quantify test-game sampling only; they do "
        "not replace independent pretraining seeds.",
        "- Board-probe point estimates use one deterministic probe seed; the "
        "random-encoder control is not a substitute for model-seed replication.",
        "- Cross-board results compare separately trained scaling conditions, not "
        "zero-shot transfer.",
        "",
        "## Artifact index",
        "",
        f"- Unified JSON: `{prepared.evaluator.results_path}`",
        f"- Unified summary: `{prepared.output_dir / ('summary__' + PROTOCOL_ID + '.md')}`",
        f"- Position manifest: "
        f"`{prepared.output_dir / ('position_manifest__' + PROTOCOL_ID + '.json')}`",
        f"- Legal-by-move figure: `{figure_paths['legal_accuracy_by_move']}`",
        f"- Board-by-move figure: `{figure_paths['board_accuracy_by_move']}`",
    ]
    if random_results_path is not None:
        lines.append(f"- Random-control JSON: `{random_results_path}`")
    if intervention:
        lines.append(
            f"- Causal-intervention JSON: `{prepared.evaluator.results_path}` "
            "(`causal_intervention` key)"
        )
    report_path = prepared.output_dir / f"proposal_report__{SUITE_ID}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _headline_metrics(results: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    native = results.get("native_ar")
    for head_type in ("linear", "mlp"):
        if native:
            # In the objective-specific thesis protocol, an AR cell has one
            # native readout. Repeat that value in both comparison columns so
            # it can be compared separately with JEPA Linear and JEPA MLP.
            legal = native
        else:
            saved = results.get("frozen_next_move", {}).get(head_type)
            if not saved:
                continue
            legal = saved["test_legal"]
        metrics[f"{head_type}_legal_top1"] = float(
            legal["top1_legal_per_token"]
        )
        metrics[f"{head_type}_legal_mass"] = float(
            legal["legal_prob_mass_per_token"]
        )
        metrics[f"{head_type}_legal_lift"] = float(
            legal["top1_legal_normalized_lift"]
        )
    for probe_type in ("linear", "mlp"):
        for mode in ("absolute", "relative"):
            metrics[f"{probe_type}_board_{mode}"] = float(
                _selected_metric(results, probe_type, mode)[
                    "macro_balanced_accuracy"
                ]
            )
    return metrics


def write_cross_board_training_time_figure(
    *,
    artifacts_root: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> Path | None:
    """Write one 4-system x 3-board timing figure once all runs are present."""
    artifacts_root = Path(artifacts_root)
    systems = (
        ("transformer", "ar"),
        ("transformer", "jepa"),
        ("mamba", "ar"),
        ("mamba", "jepa"),
    )
    values: dict[tuple[str, str], list[float]] = {system: [] for system in systems}
    for board_size in BOARD_SIZES:
        for system in systems:
            case = resolve_case(
                system[0],
                system[1],
                board_size,
                registry_path=registry_path,
            )
            try:
                run_dir = resolve_run_dir(case, artifacts_root)
            except ModelNotReadyError:
                return None
            result_path = (
                run_dir / "thesis_eval" / "final" / f"results__{PROTOCOL_ID}.json"
            )
            if not result_path.is_file():
                return None
            result = json.loads(result_path.read_text(encoding="utf-8"))
            timing = result["metadata"].get("training_metrics") or {}
            hours = timing.get("estimated_training_hours")
            if hours is None or not math.isfinite(float(hours)):
                return None
            values[system].append(float(hours))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = artifacts_root / "reports" / "thesis_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_time_by_board_size.png"
    x = np.arange(len(BOARD_SIZES), dtype=float)
    width = 0.19
    labels = {
        ("transformer", "ar"): "Transformer-AR",
        ("transformer", "jepa"): "Transformer-JEPA",
        ("mamba", "ar"): "Mamba-AR",
        ("mamba", "jepa"): "Mamba-JEPA",
    }
    colors = ("#4c78a8", "#72b7b2", "#f58518", "#e45756")
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for index, (system, color) in enumerate(zip(systems, colors)):
        offset = (index - 1.5) * width
        axis.bar(
            x + offset,
            values[system],
            width=width,
            label=labels[system],
            color=color,
        )
    axis.set(
        xticks=x,
        xticklabels=[f"{size}x{size}" for size in BOARD_SIZES],
        ylabel="Estimated training-loop time (hours)",
        xlabel="Board size",
        title="Training cost across architecture, objective, and board size",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_factorial_report(
    board_size: int,
    *,
    artifacts_root: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> Path | None:
    """Write the 2x2 architecture×objective report once all four cells exist."""
    board_size = normalize_board_size(board_size)
    artifacts_root = Path(artifacts_root)
    cells: dict[tuple[str, str], tuple[EvaluationCase, Path, dict[str, Any]]] = {}
    for architecture in ARCHITECTURES:
        for objective in OBJECTIVES:
            case = resolve_case(
                architecture,
                objective,
                board_size,
                registry_path=registry_path,
            )
            try:
                run_dir = resolve_run_dir(case, artifacts_root)
            except ModelNotReadyError:
                return None
            results_path = (
                run_dir
                / "thesis_eval"
                / "final"
                / f"results__{PROTOCOL_ID}.json"
            )
            if not results_path.is_file():
                return None
            results = json.loads(results_path.read_text(encoding="utf-8"))
            cells[(architecture, objective)] = (case, results_path, results)

    split_hashes = {
        item[2]["metadata"]["split_manifest_hash"] for item in cells.values()
    }
    if len(split_hashes) != 1:
        raise RuntimeError("The four factorial cells do not share one test split")
    metric_names = sorted(
        set.intersection(
            *(
                set(_headline_metrics(item[2]))
                for item in cells.values()
            )
        )
    )
    training_precisions = {
        f"{architecture}_{objective}": item[2]["metadata"].get(
            "training_precision"
        )
        for (architecture, objective), item in cells.items()
    }
    reported_precisions = {
        str(value).lower()
        for value in training_precisions.values()
        if value not in (None, "")
    }
    precision_comparable = (
        len(reported_precisions) == 1
        and all(value not in (None, "") for value in training_precisions.values())
    )
    derived: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        t_ar = _headline_metrics(cells[("transformer", "ar")][2])[metric]
        t_jepa = _headline_metrics(cells[("transformer", "jepa")][2])[metric]
        m_ar = _headline_metrics(cells[("mamba", "ar")][2])[metric]
        m_jepa = _headline_metrics(cells[("mamba", "jepa")][2])[metric]
        derived[metric] = {
            "transformer_ar": t_ar,
            "transformer_jepa": t_jepa,
            "mamba_ar": m_ar,
            "mamba_jepa": m_jepa,
            "architecture_main_effect_mamba_minus_transformer": (
                ((m_ar + m_jepa) - (t_ar + t_jepa)) / 2.0
            ),
            "objective_main_effect_jepa_minus_ar": (
                ((t_jepa + m_jepa) - (t_ar + m_ar)) / 2.0
            ),
            "interaction_difference_in_differences": (
                (m_jepa - m_ar) - (t_jepa - t_ar)
            ),
        }

    output_dir = (
        artifacts_root
        / "reports"
        / "thesis_eval"
        / f"b{board_size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": SUITE_ID,
        "protocol": PROTOCOL_ID,
        "board_size": board_size,
        "split_manifest_hash": next(iter(split_hashes)),
        "training_precisions": training_precisions,
        "training_precision_comparable": precision_comparable,
        "sources": {
            f"{architecture}_{objective}": str(item[1])
            for (architecture, objective), item in cells.items()
        },
        "metrics": derived,
        "training_time_hours": {
            f"{architecture}_{objective}": (
                item[2]["metadata"].get("training_metrics") or {}
            ).get("estimated_training_hours")
            for (architecture, objective), item in cells.items()
        },
    }
    json_path = output_dir / f"factorial_2x2_b{board_size}.json"
    _write_json(json_path, payload)
    timing_figure = output_dir / f"training_time_2x2_b{board_size}.png"
    timing_values = payload["training_time_hours"]
    if all(value is not None for value in timing_values.values()):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [
            "Transformer\nAR",
            "Transformer\nJEPA",
            "Mamba\nAR",
            "Mamba\nJEPA",
        ]
        keys = (
            "transformer_ar",
            "transformer_jepa",
            "mamba_ar",
            "mamba_jepa",
        )
        values = [float(timing_values[key]) for key in keys]
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        bars = axis.bar(
            labels,
            values,
            color=("#4c78a8", "#72b7b2", "#f58518", "#e45756"),
        )
        axis.bar_label(bars, fmt="%.2f h", padding=3)
        axis.set(
            ylabel="Estimated training-loop time (hours)",
            title=f"{board_size}x{board_size}: model training time",
        )
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(timing_figure, dpi=180, bbox_inches="tight")
        plt.close(fig)
    lines = [
        f"# Thesis 2×2 factorial comparison — {board_size}x{board_size}",
        "",
        f"- **Suite:** `{SUITE_ID}` / `{PROTOCOL_ID}`",
        f"- **Split:** `{next(iter(split_hashes))}`",
        "- **Training precision:** "
        + ", ".join(
            f"`{name}={value or 'unreported'}`"
            for name, value in training_precisions.items()
        ),
        "",
        "| Metric | Transformer AR | Transformer JEPA | Mamba AR | Mamba JEPA | "
        "Architecture effect | Objective effect | Interaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, values in derived.items():
        lines.append(
            f"| {metric} | {_pct(values['transformer_ar'])} | "
            f"{_pct(values['transformer_jepa'])} | "
            f"{_pct(values['mamba_ar'])} | {_pct(values['mamba_jepa'])} | "
            f"{_pct(values['architecture_main_effect_mamba_minus_transformer'])} | "
            f"{_pct(values['objective_main_effect_jepa_minus_ar'])} | "
            f"{_pct(values['interaction_difference_in_differences'])} |"
        )
    lines += [
        "",
        "For legal-move rows, each AR value is its single native-head result; "
        "the Linear and MLP rows compare that same AR result against the "
        "corresponding frozen JEPA readout.",
        "",
        "Interaction is `(Mamba-JEPA − Mamba-AR) − "
        "(Transformer-JEPA − Transformer-AR)`. Positive values mean JEPA gains "
        "more (or loses less) under Mamba for that metric.",
        "",
        (
            "**Comparability gate: PASS.** All four checkpoints report the same "
            "training precision."
            if precision_comparable
            else
            "**Comparability gate: WARNING.** Training precision is mixed or "
            "unreported. Treat the 2x2 effects as provisional until all four "
            "headline checkpoints use the same reported precision."
        ),
        "",
        "## Training efficiency",
        "",
        (
            f"![Training time]({timing_figure.name})"
            if timing_figure.is_file()
            else "Training-time figure pending because one or more runs lack "
            "`metrics.csv` timing data."
        ),
        "",
        "Times use the chunk-level training logs and exclude checkpoint/Drive "
        "synchronization unless it was included inside the recorded chunk time.",
        "",
        f"Machine-readable results: `{json_path}`",
    ]
    report_path = output_dir / f"factorial_2x2_b{board_size}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_cross_board_training_time_figure(
        artifacts_root=artifacts_root,
        registry_path=registry_path,
    )
    return report_path


def run_complete_evaluation(
    prepared: PreparedEvaluation,
    *,
    include_random_board_control: bool = False,
    overwrite_incompatible: bool = False,
    allow_source_drift: bool = False,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Path | None]:
    evaluator = prepared.evaluator
    evaluator.print_protocol()

    stage_number = 0
    total_stages = (
        (5 if prepared.case.objective == "ar" else 6)
        + int(include_random_board_control)
    )

    def run_stage(label: str, action):
        nonlocal stage_number
        stage_number += 1
        print(
            f"\n[{stage_number}/{total_stages}] START {label}",
            flush=True,
        )
        started = time.perf_counter()
        result = action()
        print(
            f"[{stage_number}/{total_stages}] DONE  {label} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
        return result

    if prepared.case.objective == "ar":
        run_stage("native AR legal-move evaluation", evaluator.run_native_ar)
    else:
        run_stage("linear frozen next-move head", lambda: evaluator.run_frozen_head("linear"))
        run_stage("MLP frozen next-move head", lambda: evaluator.run_frozen_head("mlp"))

    def needs_per_position_refresh(probe_type: str) -> bool:
        saved = evaluator.results.get("board_state", {}).get(probe_type)
        if not saved:
            return False
        return any(
            not saved.get("selected", {}).get(mode, {}).get("test", {}).get(
                "per_position"
            )
            for mode in ("absolute", "relative")
        )

    linear_force = needs_per_position_refresh("linear")
    mlp_force = needs_per_position_refresh("mlp")
    if linear_force or mlp_force:
        print(
            "Saved board probes predate per-move curves; refreshing only the "
            "board-probe stages on the identical fixed manifest.",
            flush=True,
        )
    run_stage(
        "linear board-state probe",
        lambda: evaluator.run_board_probe("linear", force=linear_force),
    )
    run_stage(
        "MLP board-state probe",
        lambda: evaluator.run_board_probe("mlp", force=mlp_force),
    )
    run_stage("primary summary", evaluator.write_summary)
    random_path = None
    if include_random_board_control:
        def random_control_stage():
            prepared.primary_model.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return run_random_board_control(
                prepared,
                overwrite_incompatible=overwrite_incompatible,
                allow_source_drift=allow_source_drift,
            )

        random_path = run_stage(
            "architecture-matched random-encoder head/probe control",
            random_control_stage,
        )
    def final_reports_stage():
        proposal = write_proposal_report(
            prepared,
            random_results_path=random_path,
        )
        factorial = write_factorial_report(
            prepared.case.board_size,
            artifacts_root=prepared.artifacts_root,
            registry_path=registry_path,
        )
        return proposal, factorial

    proposal_report, factorial_report = run_stage(
        "proposal and factorial reports",
        final_reports_stage,
    )
    return {
        "results": evaluator.results_path,
        "summary": prepared.output_dir / f"summary__{PROTOCOL_ID}.md",
        "proposal_report": proposal_report,
        "random_control": random_path,
        "factorial_report": factorial_report,
    }
