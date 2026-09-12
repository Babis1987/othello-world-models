"""Collect the main-body thesis results into ``Main_Experimentation_Results``.

The canonical artifact store lives outside the repository (Google Drive). This
script copies the *reportable* subset of that store into the repository so the
thesis results are readable without Drive access, and records a provenance
manifest for every copied byte.

Only bf16-trained, seed-42 headline runs enter the twelve architecture/board
cells; the seeded replications add seeds 0-3 for the boards that have them.
Checkpoints, probe weights and datasets are never copied.

Usage::

    python tools/organize_main_experimentation_results.py --dry-run
    python tools/organize_main_experimentation_results.py --profile lean
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ARTIFACTS_ROOT = Path(r"G:\My Drive\Master_Thesis_Artifacts")
DESTINATION_ROOT = Path(__file__).resolve().parents[1] / "Main_Experimentation_Results"

# Reportable file types. Checkpoints (.pt), datasets and caches never qualify.
INCLUDED_EXTENSIONS = {".json", ".md", ".png", ".csv", ".tex"}

# Multi-megabyte machine-readable dumps. Every number they carry is also in the
# sibling ``summary__*.md`` / ``proposal_report__*.md``, so the ``lean`` profile
# leaves them in the Drive artifact store.
BULK_PATTERNS = (
    "position_manifest__*.json",  # shared evaluation position set, not a result
    "results__unified_eval_v*.json",  # per-position dump behind the summaries
    "case_manifest__*.json",  # per-case intervention input list
)

# Directories that are never part of the reportable tree.
EXCLUDED_DIR_NAMES = {".recovery_backups", ".ipynb_checkpoints", "__pycache__"}


@dataclass(frozen=True)
class Mapping:
    """One destination directory fed by one Drive source directory."""

    target: str
    source: str


# The twelve headline cells: bf16 training, seed 42, final checkpoint.
MAIN_RUNS = (
    Mapping("Transformer-AR/8x8", "runs/Transformer-AR/transformer_ar_b8_bf16"),
    Mapping("Transformer-AR/12x12", "runs/Transformer-AR/transformer_ar_b12"),
    Mapping("Transformer-AR/16x16", "runs/Transformer-AR/transformer_ar_b16"),
    Mapping(
        "Transformer-JEPA/8x8",
        "runs/Transformer-JEPA/jepa_v5_infonce_b8_run_001_hd_allpos_bf16",
    ),
    Mapping(
        "Transformer-JEPA/12x12",
        "runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12",
    ),
    Mapping(
        "Transformer-JEPA/16x16",
        "runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16",
    ),
    Mapping("Mamba-AR/8x8", "runs/Mamba-AR/mamba_ar_8x8_bf16"),
    Mapping("Mamba-AR/12x12", "runs/Mamba-AR/mamba_ar_b12"),
    Mapping("Mamba-AR/16x16", "runs/Mamba-AR/mamba_ar_b16"),
    Mapping("Mamba-JEPA/8x8", "runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8"),
    Mapping(
        "Mamba-JEPA/12x12",
        "runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b12",
    ),
    Mapping(
        "Mamba-JEPA/16x16",
        "runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16",
    ),
)

# ``unified_eval_v4`` over all three boards is the headline cross-model report.
# The earlier board-12-only comparisons under ``comparisons/``,
# ``comparisons_v2/`` and ``unified_eval_v3/`` are superseded by it.
FULL_COMPARISON = (
    Mapping(
        "Full_Comparison/all_boards",
        "reports/thesis_eval/comparisons_v2/unified_eval_v4/"
        "both_architectures__both_objectives__all_boards",
    ),
    Mapping("Full_Comparison/8x8", "reports/thesis_eval/b8"),
    Mapping("Full_Comparison/12x12", "reports/thesis_eval/b12"),
    Mapping("Full_Comparison/16x16", "reports/thesis_eval/b16"),
)

# Seeds 0-3 per cell, plus the multi-seed factorial aggregates.
SEEDED_RUNS = (
    Mapping(
        "seeded_results/Transformer-AR/8x8",
        "runs/seeded_replications/Transformer-AR/transformer_ar_b8_bf16",
    ),
    Mapping(
        "seeded_results/Transformer-AR/16x16",
        "runs/seeded_replications/Transformer-AR/transformer_ar_b16",
    ),
    Mapping(
        "seeded_results/Transformer-JEPA/8x8",
        "runs/seeded_replications/Transformer-JEPA/"
        "jepa_v5_infonce_b8_run_001_hd_allpos_bf16",
    ),
    Mapping(
        "seeded_results/Transformer-JEPA/16x16",
        "runs/seeded_replications/Transformer-JEPA/"
        "transformer_jepa_v5_hd_infonce_allpos_b16",
    ),
    Mapping(
        "seeded_results/Mamba-AR/8x8",
        "runs/seeded_replications/Mamba-AR/mamba_ar_8x8_bf16",
    ),
    Mapping(
        "seeded_results/Mamba-AR/16x16",
        "runs/seeded_replications/Mamba-AR/mamba_ar_b16",
    ),
    Mapping(
        "seeded_results/Mamba-JEPA/8x8",
        "runs/seeded_replications/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8",
    ),
    Mapping(
        "seeded_results/Mamba-JEPA/16x16",
        "runs/seeded_replications/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16",
    ),
    Mapping("seeded_results/aggregate/8x8", "reports/thesis_eval/multiseed/b8"),
    Mapping("seeded_results/aggregate/16x16", "reports/thesis_eval/multiseed/b16"),
)

EXTRA_FILES = (
    Mapping(
        "Full_Comparison/training_time_by_board_size.png",
        "reports/thesis_eval/training_time_by_board_size.png",
    ),
)

ALL_MAPPINGS = MAIN_RUNS + FULL_COMPARISON + SEEDED_RUNS


def is_bulk(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in BULK_PATTERNS)


def reportable_files(source_dir: Path, include_bulk: bool) -> list[Path]:
    """Every copyable file under ``source_dir``, in stable order."""
    selected: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if EXCLUDED_DIR_NAMES.intersection(path.relative_to(source_dir).parts):
            continue
        if path.suffix.lower() not in INCLUDED_EXTENSIONS:
            continue
        if not include_bulk and is_bulk(path.name):
            continue
        selected.append(path)
    return selected


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("lean", "full"),
        default="lean",
        help=(
            "lean (default) omits the multi-megabyte per-position dumps; "
            "full copies every reportable file"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what each profile would copy without writing anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite a destination tree that already holds files",
    )
    args = parser.parse_args()
    include_bulk = args.profile == "full"

    missing = [
        str(ARTIFACTS_ROOT / mapping.source)
        for mapping in ALL_MAPPINGS
        if not (ARTIFACTS_ROOT / mapping.source).is_dir()
    ] + [
        str(ARTIFACTS_ROOT / mapping.source)
        for mapping in EXTRA_FILES
        if not (ARTIFACTS_ROOT / mapping.source).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing sources:\n" + "\n".join(missing))

    plan: list[tuple[Mapping, list[Path]]] = [
        (mapping, reportable_files(ARTIFACTS_ROOT / mapping.source, include_bulk))
        for mapping in ALL_MAPPINGS
    ]

    # A cell is "unevaluated" when it carries training bookkeeping but no
    # evaluation report, which is how a trained-but-not-yet-scored run looks.
    unevaluated = [
        mapping.target
        for mapping, files in plan
        if not any(
            path.name.startswith(("summary__", "proposal_report__"))
            for path in files
        )
        and mapping.target.count("/") == 1
        and not mapping.target.startswith("Full_Comparison")
    ]
    total_files = sum(len(files) for _, files in plan) + len(EXTRA_FILES)
    total_bytes = sum(
        path.stat().st_size for _, files in plan for path in files
    ) + sum((ARTIFACTS_ROOT / m.source).stat().st_size for m in EXTRA_FILES)

    for mapping, files in plan:
        size = sum(path.stat().st_size for path in files)
        print(f"{mapping.target:40s} {len(files):4d} files  {size / 1048576:8.1f} MiB")
    print(f"{'TOTAL':40s} {total_files:4d} files  {total_bytes / 1048576:8.1f} MiB")
    if unevaluated:
        print("\nTrained but not evaluated: " + ", ".join(unevaluated))

    if args.dry_run:
        return

    if (
        DESTINATION_ROOT.exists()
        and any(path.is_file() for path in DESTINATION_ROOT.rglob("*"))
        and not args.force
    ):
        raise RuntimeError(
            f"Destination already contains files (pass --force): {DESTINATION_ROOT}"
        )

    manifest: list[dict[str, object]] = []
    for mapping, files in plan:
        source_dir = ARTIFACTS_ROOT / mapping.source
        target_dir = DESTINATION_ROOT / mapping.target
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_file in files:
            relative = source_file.relative_to(source_dir)
            target_file = target_dir / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            manifest.append(
                {
                    "target": f"{mapping.target}/{relative.as_posix()}",
                    "source": source_file.relative_to(ARTIFACTS_ROOT).as_posix(),
                    "bytes": source_file.stat().st_size,
                    "sha256": sha256_of(target_file),
                }
            )
        print(f"copied {mapping.target}", flush=True)

    for mapping in EXTRA_FILES:
        source_file = ARTIFACTS_ROOT / mapping.source
        target_file = DESTINATION_ROOT / mapping.target
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        manifest.append(
            {
                "target": mapping.target,
                "source": mapping.source,
                "bytes": source_file.stat().st_size,
                "sha256": sha256_of(target_file),
            }
        )

    manifest_path = DESTINATION_ROOT / "provenance" / "results_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts_root": str(ARTIFACTS_ROOT),
                "profile": args.profile,
                "excluded_patterns": (
                    [] if include_bulk else sorted(BULK_PATTERNS)
                ),
                "included_extensions": sorted(INCLUDED_EXTENSIONS),
                "cells_trained_but_not_evaluated": unevaluated,
                "file_count": len(manifest),
                "total_bytes": sum(int(entry["bytes"]) for entry in manifest),
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nmanifest: {manifest_path}")
    print(f"TOTAL {len(manifest)} files, {total_bytes / 1048576:.1f} MiB")


if __name__ == "__main__":
    main()
