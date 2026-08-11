"""Run the standalone Li/Nanda/adapted causal-intervention comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from othello_thesis.evaluation.causal_suite import (
    CAUSAL_SUITE_ID,
    METHODS,
    PROFILES,
    CausalInterventionSuite,
    causal_claim_scope,
)
from othello_thesis.evaluation.thesis import (
    ARCHITECTURES,
    BOARD_SIZES,
    DEFAULT_REGISTRY,
    OBJECTIVES,
    ModelNotReadyError,
    prepare_evaluation,
    resolve_case,
    resolve_data_dir,
    resolve_run_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Li activation optimization, Nanda linear branch patching, "
            "and the historical adapted residual intervention for one thesis cell."
        )
    )
    parser.add_argument("--architecture", required=True, choices=ARCHITECTURES)
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument("--board-size", required=True, type=int, choices=BOARD_SIZES)
    parser.add_argument(
        "--profile", required=True, choices=tuple(PROFILES), help="smoke or full"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to run; completed methods resume independently.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("/content/drive/MyDrive/Master_Thesis_Artifacts"),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--local-checkpoint-root",
        type=Path,
        default=Path("/content/causal_intervention_checkpoints"),
    )
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="Record and allow source-only drift in resumable setup state.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this profile's causal state and recompute selected methods.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve checkpoint/data/output paths without loading the model.",
    )
    return parser


def _profile_overrides(profile: str) -> dict[str, object]:
    if profile == "full":
        return {"head_tuning_enabled": False}
    # Smoke is an execution gate, not a result.  Bound only prerequisites that
    # cannot be reused from an already-complete common evaluation.
    return {
        "head_tuning_enabled": False,
        "head_max_shards": 1,
        "head_selection_games": 32,
        "head_patience": 1,
        "board_positions_per_game": 4,
        "board_train_games": 32,
        "board_selection_games": 16,
        "board_test_games": 32,
        "board_epochs": 1,
        "intervention_selection_cases": 8,
        "intervention_test_cases": 16,
        "intervention_batch_size": 8,
    }


def main() -> None:
    args = build_parser().parse_args()
    case = resolve_case(
        args.architecture,
        args.objective,
        args.board_size,
        registry_path=args.registry,
    )
    condition_scope = causal_claim_scope(
        case.architecture, case.objective, case.board_size
    )
    claim_scope = (
        "pipeline_smoke_no_thesis_claim"
        if args.profile == "smoke"
        else condition_scope
    )
    try:
        data_dir = resolve_data_dir(case, args.artifacts_root)
        run_dir = resolve_run_dir(case, args.artifacts_root)
    except ModelNotReadyError as exc:
        raise SystemExit(str(exc)) from exc
    output_subdir = (
        Path("causal_intervention") / CAUSAL_SUITE_ID / args.profile
    )
    print("=" * 72, flush=True)
    print("STANDALONE CAUSAL-INTERVENTION SUITE", flush=True)
    print("case       :", case.label, flush=True)
    print("profile    :", args.profile, flush=True)
    print("methods    :", ", ".join(args.methods), flush=True)
    print("condition scope:", condition_scope, flush=True)
    print("claim scope    :", claim_scope, flush=True)
    print("checkpoint :", run_dir / "final.pt", flush=True)
    print("data       :", data_dir, flush=True)
    print("output     :", run_dir / output_subdir, flush=True)
    print("=" * 72, flush=True)
    if args.profile == "smoke":
        print(
            "SMOKE IS A PIPELINE GATE ONLY; do not use it for thesis conclusions.",
            flush=True,
        )
    else:
        print(
            "FULL uses Li's 1,000 activation-optimization steps and is resumable "
            "per case batch. Li runs last and prints a timing projection.",
            flush=True,
        )
    if args.dry_run:
        print("DRY RUN OK: no checkpoint loaded and no result stage executed.")
        return

    prepared = prepare_evaluation(
        case.architecture,
        case.objective,
        case.board_size,
        artifacts_root=args.artifacts_root,
        project_root=args.project_root,
        device=args.device,
        local_checkpoint_root=args.local_checkpoint_root,
        registry_path=args.registry,
        overwrite_incompatible=args.force,
        allow_source_drift=args.allow_source_drift,
        config_overrides=_profile_overrides(args.profile),
        output_subdir=output_subdir,
    )
    outputs = CausalInterventionSuite(
        prepared,
        profile=args.profile,
        methods=args.methods,
        force=args.force,
    ).run()
    print("\nCAUSAL SUITE COMPLETE", flush=True)
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
