"""Internal implementation module extracted from the canonical JEPA trainer."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from othello_thesis.data.chunk_dataset import OthelloChunkDataset, load_chunk
from othello_thesis.models.transformer import GPTConfig
from othello_thesis.objectives.jepa import OthelloJEPA
from othello_thesis.training._jepa_config import TrainConfig, objective_class

def list_chunks(data_dir: str) -> list[Path]:
    """List all .pickle chunk files in a directory, sorted by name."""
    chunks = sorted(Path(data_dir).glob("*.pickle"))
    if not chunks:
        raise FileNotFoundError(f"No .pickle chunks found in {data_dir}")
    return chunks

def resolve_chunks(cfg: TrainConfig) -> tuple[list[Path], list[Path]]:
    """Resolve train/val chunk lists, validating no overlap.

    Two modes:
        --data_dir : Li-style split into first li_split_train_chunks
                     for train, remaining for val.
        --train_dir / --val_dir : explicit directories.
    """
    if cfg.data_dir:
        all_chunks = list_chunks(cfg.data_dir)
        n_train = cfg.li_split_train_chunks
        if len(all_chunks) <= n_train:
            raise ValueError(
                f"data_dir has {len(all_chunks)} chunks, but li_split_train_chunks="
                f"{n_train}; no validation chunks would remain. Either lower "
                f"li_split_train_chunks or use --train_dir/--val_dir explicitly."
            )
        train_chunks = all_chunks[:n_train]
        val_chunks = all_chunks[n_train:]
    else:
        if not cfg.train_dir:
            raise ValueError("Provide either --data_dir or --train_dir.")
        train_chunks = list_chunks(cfg.train_dir)
        val_chunks = list_chunks(cfg.val_dir) if cfg.val_dir else []

    # Resolve to absolute paths for accurate overlap detection (handles
    # symlinks and relative-vs-absolute path mismatches).
    train_set = {p.resolve() for p in train_chunks}
    val_set = {p.resolve() for p in val_chunks}
    overlap = train_set & val_set
    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            f"Train/validation chunks overlap. Example: {example}. "
            f"Train and val must be disjoint."
        )
    return train_chunks, val_chunks

def parse_milestones(spec: str) -> list[int]:
    """Parse a comma-separated list of integer game-count milestones."""
    if not spec.strip():
        return []
    return sorted({int(x.strip()) for x in spec.split(",") if x.strip()})

def parse_fraction_checkpoints(spec: str) -> list[float]:
    """Parse comma-separated checkpoint fractions in ``(0, 1)``."""
    if not spec.strip():
        return []
    fractions = sorted({float(x.strip()) for x in spec.split(",") if x.strip()})
    invalid = [value for value in fractions if not (0.0 < value < 1.0)]
    if invalid:
        raise ValueError(
            "intermediate_checkpoint_fractions must be in (0, 1), "
            f"got {invalid}"
        )
    return fractions

def fraction_checkpoint_name(fraction: float) -> str:
    """Return checkpoint name for a fractional training milestone."""
    return f"checkpoint_frac_{round(fraction * 100):03d}.pt"

def read_manifest_counts(chunk_dir: Path) -> dict[str, int]:
    """Read shard counts from a manifest.json if present.

    Returns an empty dict if the manifest is missing or malformed.
    """
    manifest = chunk_dir / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    counts = data.get("shard_counts", data)
    if not isinstance(counts, dict):
        return {}
    return {str(k): int(v) for k, v in counts.items()}

def estimate_total_steps(chunks: list[Path], cfg: TrainConfig) -> tuple[int, int]:
    """Estimate the total number of optimiser steps for the cosine schedule.

    Returns:
        (total_steps, total_games_estimate).

    Order of preference:
        1. cfg.total_steps if explicitly set.
        2. Sum of manifest.json shard_counts if all chunks are listed.
        3. len(chunks) * cfg.games_per_chunk fallback.
    """
    counts: dict[str, int] = {}
    for parent in sorted({p.parent for p in chunks}):
        counts.update(read_manifest_counts(parent))

    if counts and all(p.name in counts for p in chunks):
        total_games = sum(counts[p.name] for p in chunks)
        source = "manifest"
    else:
        total_games = len(chunks) * cfg.games_per_chunk
        source = f"games_per_chunk={cfg.games_per_chunk}"

    if cfg.max_chunks:
        per_chunk = counts.get(chunks[0].name, cfg.games_per_chunk) if counts else cfg.games_per_chunk
        total_games = min(total_games, cfg.max_chunks * per_chunk)

    if cfg.max_batches_per_chunk:
        limited_chunks = min(len(chunks), cfg.max_chunks or len(chunks))
        total_games = min(
            total_games,
            limited_chunks * cfg.max_batches_per_chunk * cfg.batch_size,
        )

    if cfg.total_steps > 0:
        total_steps = cfg.total_steps
        print(f"using explicit total_steps={total_steps}", flush=True)
    else:
        total_steps = math.ceil(total_games / cfg.batch_size) * cfg.passes_over_data
        print(f"estimated total_steps={total_steps} ({source}, "
              f"~{total_games:,} games over {cfg.passes_over_data} pass(es))", flush=True)

    return total_steps, total_games

def sync_to_drive(local_dir: Path, drive_dir: Path) -> tuple[int, float]:
    """Copy any new or updated files from local_dir to drive_dir.

    Idempotent: only copies files whose mtime is newer than the destination.
    Failures (e.g. transient Drive errors) are caught and logged so they
    cannot crash the training loop.

    Returns:
        (n_files_copied, total_seconds).
    """
    if not local_dir.exists():
        return 0, 0.0

    drive_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    t0 = time.time()

    try:
        for f in sorted(local_dir.iterdir()):
            if not f.is_file():
                continue
            target = drive_dir / f.name
            needs_copy = (
                not target.exists()
                or f.stat().st_mtime > target.stat().st_mtime
            )
            if needs_copy:
                # Atomic copy via tmp + rename so a partial write never
                # leaves a corrupt file on Drive.
                tmp = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(f, tmp)
                tmp.replace(target)
                n_copied += 1
    except Exception as e:
        print(f"  ⚠ drive sync failed ({type(e).__name__}: {e}); "
              f"continuing training", flush=True)
        return n_copied, time.time() - t0

    return n_copied, time.time() - t0

def make_loader(
    chunk_path: Path,
    cfg: TrainConfig,
    model_cfg: GPTConfig,
    shuffle: bool,
) -> tuple[DataLoader, list[list[int]], OthelloChunkDataset]:
    """Load a chunk and wrap it in a fresh DataLoader."""
    games = load_chunk(str(chunk_path))
    dataset = OthelloChunkDataset(
        games=games,
        block_size=model_cfg.block_size,
        board_size=cfg.board_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, games, dataset

def get_lr(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """Cosine schedule with linear warmup, decaying to min_lr_ratio * lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + (cfg.learning_rate - min_lr) * cosine

def autocast_dtype(cfg: TrainConfig) -> torch.dtype:
    """Resolve configured CUDA autocast dtype."""
    if cfg.precision == "bf16":
        return torch.bfloat16
    if cfg.precision == "fp16":
        return torch.float16
    if cfg.precision == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {cfg.precision!r}")

def use_autocast(cfg: TrainConfig, device: torch.device) -> bool:
    """Return whether CUDA autocast should be enabled."""
    return device.type == "cuda" and cfg.precision in {"bf16", "fp16"}

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap torch.compile modules when calling custom JEPA methods."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model

@torch.no_grad()
def update_target_encoder(model: torch.nn.Module) -> None:
    """Apply target update on the underlying objective module, if present."""
    raw_model = unwrap_model(model)
    if hasattr(raw_model, "update_target_encoder"):
        raw_model.update_target_encoder()
        return
    if hasattr(raw_model, "update_targets"):
        raw_model.update_targets()
        return

def model_state_dict(model: torch.nn.Module) -> dict:
    """Get the state_dict from a model, unwrapping torch.compile if used."""
    return unwrap_model(model).state_dict()

def load_model_state(model: torch.nn.Module, state: dict) -> None:
    """Load state into a model, unwrapping torch.compile if used."""
    raw_model = unwrap_model(model)
    try:
        raw_model.load_state_dict(state)
    except RuntimeError:
        # Older VICReg checkpoints used ``encoder.*`` before the unified JEPA
        # wrapper standardized on ``context_encoder.*`` for both variants.
        if (
            isinstance(raw_model, OthelloJEPA)
            and raw_model.jepa_config.variant == "jepa_v1"
            and any(key.startswith("encoder.") for key in state)
        ):
            remapped = {
                key.replace("encoder.", "context_encoder.", 1)
                if key.startswith("encoder.")
                else key: value
                for key, value in state.items()
            }
            raw_model.load_state_dict(remapped)
            return
        raise

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: TrainConfig,
    jepa_cfg: Any,
    step: int,
    pass_num: int,
    chunk_count: int,
    next_chunk_idx: int,
    games_seen: int,
    tokens_seen: int,
) -> None:
    """Atomically save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "train_config": asdict(cfg),
            "jepa_config": asdict(jepa_cfg),
            "model_config": {
                "objective_class": objective_class(cfg),
                **asdict(jepa_cfg),
            },
            "step": step,
            "pass_num": pass_num,
            "chunk_count": chunk_count,
            "next_chunk_idx": next_chunk_idx,
            "games_seen": games_seen,
            "tokens_seen": tokens_seen,
        },
        tmp_path,
    )
    tmp_path.replace(path)  # atomic rename

def append_csv(path: Path, row: dict) -> None:
    """Append a row to a CSV file, writing the header on first write.

    Floats that are NaN are written as empty strings for portability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    sanitized = {
        k: ("" if isinstance(v, float) and math.isnan(v) else v)
        for k, v in row.items()
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(sanitized)

def init_wandb(cfg: TrainConfig, jepa_cfg: Any):
    """Initialise WandB if requested. Returns the run object or None."""
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb_project was set, but wandb is not installed; "
              "continuing without WandB.", flush=True)
        return None

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config={"train": asdict(cfg), "jepa": asdict(jepa_cfg)},
    )
