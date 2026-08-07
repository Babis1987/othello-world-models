"""Colab-friendly chunked training loop for OthelloGPT.

The script follows the Go-GPT pattern: load one pickle chunk into RAM, train
through it with normal DataLoader shuffle, free it, then move to the next
chunk. It can either consume explicit train/val directories or reproduce the
Li et al. synthetic split from one data directory: first N gen10e5 chunks
for train, remaining chunks for validation.

Features:
    - Explicit fp32/fp16/bf16 precision control on CUDA.
    - Cosine learning-rate schedule with linear warmup.
    - Chunk-by-chunk training with deterministic shuffle ordering.
    - Game-count tracking (games_seen, tokens_seen) and milestone
      checkpoints (e.g. checkpoint_games_005M.pt at 5M games).
    - latest.pt updated after every chunk for Colab session resume.
    - CSV metrics log + optional WandB integration.
    - Outer progress bar across all chunks for overall ETA.
    - Optional auto-sync of the local output dir to a Drive backup dir
      after every N chunks, so an overnight Colab disconnect cannot lose
      more than N chunks of work.

Resume notes:
    Resume requires identical config (especially seed, board_size, and the
    chunk file list) to the original run. The chunk shuffling is keyed on
    seed + pass_num, so changing either will produce a different visit
    order from this checkpoint onward.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader

from othello_thesis.data.chunk_dataset import OthelloChunkDataset
from othello_thesis.models.transformer import GPTConfig, OthelloGPT
from othello_thesis.training.ar_checkpoints import save_training_checkpoint
from othello_thesis.training.ar_data import (
    estimate_total_steps as _estimate_total_steps,
    list_chunks as _list_chunks,
    make_chunk_loader,
    parse_milestones as _parse_milestones,
    read_manifest_counts as _read_manifest_counts,
    resolve_chunks as _resolve_chunks,
)
from othello_thesis.training.ar_runtime import (
    append_csv as _append_csv,
    autocast_context,
    cosine_learning_rate,
    load_model_state as _load_model_state,
    model_state_dict as _model_state_dict,
    sync_to_drive as _sync_to_drive,
    validate_resume_precision as _validate_resume_precision,
)
from othello_thesis.training.ar_steps import (
    evaluate_chunks as _evaluate_ar_chunks,
    train_one_chunk as _train_ar_chunk,
)
from othello_thesis.training.transformer_ar_runner import run_transformer_ar


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    out_dir: str
    train_dir: str | None = None
    val_dir: str | None = None
    data_dir: str | None = None
    li_split_train_chunks: int = 200

    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    batch_size: int = 256
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    passes_over_data: int = 1
    total_steps: int = 0
    games_per_chunk: int = 100_000
    precision: str = "fp16"

    num_workers: int = 0
    seed: int = 42
    eval_chunks: int = 1
    eval_every_chunks: int = 5
    checkpoint_every_chunks: int = 0
    milestone_games: str = "1000000,5000000,10000000,15000000,20000000"
    max_chunks: int = 0
    resume: str | None = None
    compile_model: bool = False

    # Drive auto-sync (for Colab overnight runs).
    drive_sync_dir: str | None = None
    drive_sync_every_chunks: int = 1

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"


# ============================================================================
# Chunk discovery and split
# ============================================================================

def list_chunks(data_dir: str) -> list[Path]:
    """List all .pickle chunk files in a directory, sorted by name."""
    return _list_chunks(data_dir)


def resolve_chunks(cfg: TrainConfig) -> tuple[list[Path], list[Path]]:
    """Resolve train/val chunk lists, validating no overlap.

    Two modes:
        --data_dir : Li-style split into first li_split_train_chunks
                     for train, remaining for val.
        --train_dir / --val_dir : explicit directories.
    """
    return _resolve_chunks(cfg)


def parse_milestones(spec: str) -> list[int]:
    """Parse a comma-separated list of integer game-count milestones."""
    return _parse_milestones(spec)


# ============================================================================
# Total-step estimation for LR schedule
# ============================================================================

def read_manifest_counts(chunk_dir: Path) -> dict[str, int]:
    """Read shard counts from a manifest.json if present.

    Returns an empty dict if the manifest is missing or malformed.
    """
    return _read_manifest_counts(chunk_dir)


def estimate_total_steps(chunks: list[Path], cfg: TrainConfig) -> tuple[int, int]:
    """Estimate the total number of optimiser steps for the cosine schedule.

    Returns:
        (total_steps, total_games_estimate).

    Order of preference:
        1. cfg.total_steps if explicitly set.
        2. Sum of manifest.json shard_counts if all chunks are listed.
        3. len(chunks) * cfg.games_per_chunk fallback.
    """
    return _estimate_total_steps(chunks, cfg)


# ============================================================================
# Drive auto-sync
# ============================================================================

def sync_to_drive(local_dir: Path, drive_dir: Path) -> tuple[int, float]:
    """Copy any new or updated files from local_dir to drive_dir.

    Idempotent: only copies files whose mtime is newer than the destination.
    Failures (e.g. transient Drive errors) are caught and logged so they
    cannot crash the training loop.

    Returns:
        (n_files_copied, total_seconds).
    """
    return _sync_to_drive(local_dir, drive_dir)


# ============================================================================
# Training utilities
# ============================================================================

def make_loader(
    chunk_path: Path,
    cfg: TrainConfig,
    model_cfg: GPTConfig,
    shuffle: bool,
) -> tuple[DataLoader, list[list[int]], OthelloChunkDataset]:
    """Load a chunk and wrap it in a fresh DataLoader."""
    return make_chunk_loader(
        chunk_path,
        block_size=model_cfg.block_size,
        board_size=cfg.board_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
    )


def get_lr(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """Cosine schedule with linear warmup, decaying to min_lr_ratio * lr."""
    return cosine_learning_rate(step, cfg, total_steps)


def _autocast_context(device: torch.device, precision: str):
    return autocast_context(device, precision)


def validate_resume_precision(
    checkpoint: Mapping[str, Any],
    requested_precision: str,
) -> None:
    """Reject mixed-precision resume runs before loading optimizer state."""
    _validate_resume_precision(checkpoint, requested_precision)


def train_one_chunk(
    model: OthelloGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    chunk_path: Path,
    cfg: TrainConfig,
    model_cfg: GPTConfig,
    device: torch.device,
    step: int,
    total_steps: int,
    progress_desc: str | None = None,
) -> tuple[float, float, int, int, int]:
    """Train one full pass over a single chunk.

    Returns:
        (avg_loss, avg_token_accuracy, new_step, n_games, n_tokens_trained_on)
    """
    return _train_ar_chunk(
        model,
        optimizer,
        scaler,
        chunk_path,
        cfg,
        model_cfg,
        device,
        step,
        total_steps,
        style="transformer",
        progress_desc=progress_desc,
    )


@torch.no_grad()
def eval_chunks(
    model: OthelloGPT,
    chunks: list[Path],
    cfg: TrainConfig,
    model_cfg: GPTConfig,
    device: torch.device,
) -> tuple[float, float, int, int]:
    """Run evaluation on the first cfg.eval_chunks val chunks.

    Returns:
        (avg_loss, avg_token_accuracy, n_games, n_tokens). Returns NaN
        for loss/acc and 0 for counts if no eval is requested.
    """
    return _evaluate_ar_chunks(
        model,
        chunks,
        cfg,
        model_cfg,
        device,
        style="transformer",
    )


# ============================================================================
# Checkpoint helpers
# ============================================================================

def model_state_dict(model: torch.nn.Module) -> dict:
    """Get the state_dict from a model, unwrapping torch.compile if used."""
    return _model_state_dict(model)


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    """Load state into a model, unwrapping torch.compile if used."""
    _load_model_state(model, state)


def save_checkpoint(
    path: Path,
    model: OthelloGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: TrainConfig,
    model_cfg: GPTConfig,
    step: int,
    pass_num: int,
    chunk_count: int,
    next_chunk_idx: int,
    games_seen: int,
    tokens_seen: int,
) -> None:
    """Atomically save a training checkpoint."""
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        cfg,
        model_cfg,
        step,
        pass_num,
        chunk_count,
        next_chunk_idx,
        games_seen,
        tokens_seen,
    )


# ============================================================================
# Logging helpers
# ============================================================================

def append_csv(path: Path, row: dict) -> None:
    """Append a row to a CSV file, writing the header on first write.

    Floats that are NaN are written as empty strings for portability.
    """
    _append_csv(path, row)


def init_wandb(cfg: TrainConfig, model_cfg: GPTConfig):
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
        config={"train": asdict(cfg), "model": asdict(model_cfg)},
    )


# ============================================================================
# Argument parsing
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Paths and split
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Single directory; first N chunks become train, rest val.")
    parser.add_argument("--li_split_train_chunks", type=int, default=200)

    # Model architecture
    parser.add_argument("--board_size", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Optimisation
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--passes_over_data", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=0,
                        help="Override schedule denominator. 0 = auto.")
    parser.add_argument("--games_per_chunk", type=int, default=100_000,
                        help="Used when manifest.json is unavailable.")
    parser.add_argument(
        "--precision", type=str, default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )

    # Run control
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_chunks", type=int, default=1,
                        help="How many val chunks to use per eval. 0 = no eval.")
    parser.add_argument("--eval_every_chunks", type=int, default=5,
                        help="Run eval every N training chunks.")
    parser.add_argument("--checkpoint_every_chunks", type=int, default=0,
                        help="Save checkpoint_chunk_NNNNNN.pt every N chunks. 0 = off.")
    parser.add_argument("--milestone_games", type=str,
                        default="1000000,5000000,10000000,15000000,20000000")
    parser.add_argument("--max_chunks", type=int, default=0,
                        help="Stop after N total chunks. 0 = no limit.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile_model", action="store_true",
                        help="Use torch.compile. Often slower on T4; try only after first run.")

    # Drive auto-sync
    parser.add_argument("--drive_sync_dir", type=str, default=None,
                        help="If set, copy local out_dir contents to this Drive "
                             "directory after every drive_sync_every_chunks chunks.")
    parser.add_argument("--drive_sync_every_chunks", type=int, default=1,
                        help="How often to sync to Drive (default 1 = every chunk).")

    # WandB
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    return parser


# ============================================================================
# Main training loop
# ============================================================================

def run_training(
    cfg: TrainConfig,
    *,
    model_factory: Callable[[GPTConfig], torch.nn.Module] | None = None,
) -> None:
    """Train one canonical run, constructing the supplied model exactly once."""
    run_transformer_ar(
        cfg,
        model_factory=model_factory,
        resolve_chunks=resolve_chunks,
        estimate_total_steps=estimate_total_steps,
        parse_milestones=parse_milestones,
        validate_resume_precision=validate_resume_precision,
        load_model_state=load_model_state,
        train_one_chunk=train_one_chunk,
        eval_chunks=eval_chunks,
        save_checkpoint=save_checkpoint,
        append_csv=append_csv,
        sync_to_drive=sync_to_drive,
        init_wandb=init_wandb,
    )

def main() -> None:
    """Parse the historical CLI and run the canonical training loop."""
    parser = build_parser()
    args = parser.parse_args()
    run_training(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
