"""Run the common proposal-aligned evaluation for one thesis condition.

Examples
--------
python -u scripts/run_thesis_evaluation.py \
  --architecture transformer --objective jepa --board-size 8 \
  --artifacts-root /content/drive/MyDrive/Master_Thesis_Artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from othello_thesis.evaluation.thesis import (
    ARCHITECTURES,
    BOARD_SIZES,
    OBJECTIVES,
    DEFAULT_REGISTRY,
    ModelNotReadyError,
    prepare_evaluation,
    resolve_case,
    resolve_data_dir,
    resolve_run_dir,
    run_complete_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Architecture × Objective × Board-size thesis cell "
            "under the shared proposal protocol."
        )
    )
    parser.add_argument("--architecture", required=True, choices=ARCHITECTURES)
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument(
        "--board-size",
        required=True,
        type=int,
        choices=BOARD_SIZES,
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/Master_Thesis_Artifacts"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--local-checkpoint-root",
        type=Path,
        default=Path("/content/unified_eval_checkpoints"),
    )
    parser.add_argument(
        "--include-random-control",
        action="store_true",
        help=(
            "On 8x8, also train frozen heads and board probes on an "
            "architecture-matched random encoder."
        ),
    )
    parser.add_argument(
        "--skip-head-tuning",
        action="store_true",
        help=(
            "Train the frozen readouts at the protocol default hyperparameters "
            "instead of running the grouped-CV search first. Reproduces the v3 "
            "readout behaviour under the v4 identity."
        ),
    )
    parser.add_argument(
        "--head-tuning-games",
        type=int,
        default=None,
        help=(
            "Games drawn from the head-training pool for the readout "
            "hyperparameter search (default 5000). The search validates inside "
            "these games; the final test split is untouched."
        ),
    )
    parser.add_argument(
        "--overwrite-incompatible",
        action="store_true",
        help=(
            "Explicitly replace an incomplete thesis_eval state whose source, "
            "checkpoint, corpus, or protocol identity changed."
        ),
    )
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help=(
            "Resume saved thesis_eval stages when the ONLY identity change is "
            "the othello_research source digest. The drift is recorded in "
            "metadata.source_drift_events; a changed checkpoint, corpus, "
            "split, or protocol still refuses to resume."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate paths without loading the checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case = resolve_case(
        args.architecture,
        args.objective,
        args.board_size,
        registry_path=args.registry,
    )
    print("=" * 72, flush=True)
    print("COMMON THESIS EVALUATION", flush=True)
    print("case          :", case.label, flush=True)
    print("JEPA condition:", "v5 hard-disjoint InfoNCE, all-position" if case.objective == "jepa" else "n/a", flush=True)
    print("artifacts root:", args.artifacts_root, flush=True)
    print("=" * 72, flush=True)
    try:
        data_dir = resolve_data_dir(case, args.artifacts_root)
        run_dir = resolve_run_dir(case, args.artifacts_root)
    except ModelNotReadyError as exc:
        raise SystemExit(str(exc)) from exc
    print("data directory:", data_dir, flush=True)
    print("run directory :", run_dir, flush=True)
    print("checkpoint    :", run_dir / "final.pt", flush=True)
    if args.dry_run:
        print("DRY RUN OK: paths resolved; checkpoint was not loaded.", flush=True)
        return

    print(
        "\n[setup] Copying/checking final.pt and constructing the frozen model...",
        flush=True,
    )
    setup_started = time.perf_counter()
    config_overrides: dict[str, object] = {}
    if args.skip_head_tuning:
        config_overrides["head_tuning_enabled"] = False
    if args.head_tuning_games is not None:
        config_overrides["head_tuning_games"] = args.head_tuning_games
    if config_overrides:
        print("protocol overrides:", config_overrides, flush=True)
    prepared = prepare_evaluation(
        args.architecture,
        args.objective,
        args.board_size,
        artifacts_root=args.artifacts_root,
        project_root=args.project_root,
        device=args.device,
        local_checkpoint_root=args.local_checkpoint_root,
        registry_path=args.registry,
        overwrite_incompatible=args.overwrite_incompatible,
        allow_source_drift=args.allow_source_drift,
        config_overrides=config_overrides,
    )
    print(
        f"[setup] Ready in {time.perf_counter() - setup_started:.1f}s. "
        "Starting resumable evaluation stages.",
        flush=True,
    )
    outputs = run_complete_evaluation(
        prepared,
        include_random_board_control=args.include_random_control,
        overwrite_incompatible=args.overwrite_incompatible,
        allow_source_drift=args.allow_source_drift,
        registry_path=args.registry,
    )
    print("\nEVALUATION COMPLETE", flush=True)
    print(json.dumps({key: None if path is None else str(path) for key, path in outputs.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
