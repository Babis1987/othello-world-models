"""Runtime, logging, and state helpers shared by canonical AR trainers."""

from __future__ import annotations

import csv
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch


def autocast_context(device: torch.device, precision: str):
    """Return the exact historical CUDA autocast context."""
    precision = precision.lower()
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )
    if precision == "fp16":
        return torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        )
    raise ValueError("precision must be one of: fp32, fp16, bf16")


def cosine_learning_rate(step: int, cfg: Any, total_steps: int) -> float:
    """Cosine decay with the historical linear-warmup indexing."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + (cfg.learning_rate - min_lr) * cosine


def validate_resume_precision(
    checkpoint: Mapping[str, Any],
    requested_precision: str,
) -> None:
    """Reject mixed-precision resume runs before optimizer restoration."""
    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, Mapping):
        raise RuntimeError(
            "Resume checkpoint has no train_config; its training precision "
            "cannot be verified. Start a new run directory."
        )
    saved_precision = train_config.get("precision")
    if saved_precision is None:
        raise RuntimeError(
            "Resume checkpoint predates explicit precision metadata; refusing "
            "to create an fp16/bf16 hybrid run. Start a new run directory."
        )
    saved = str(saved_precision).lower()
    requested = str(requested_precision).lower()
    if saved != requested:
        raise RuntimeError(
            f"Resume precision mismatch: checkpoint={saved!r}, "
            f"requested={requested!r}. Start a new run directory."
        )


def model_state_dict(model: torch.nn.Module) -> dict:
    """Return state from either a plain or torch.compile-wrapped model."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    """Load state into either a plain or torch.compile-wrapped model."""
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(state)


def append_csv(path: Path, row: dict) -> None:
    """Append one row with historical key order and NaN sanitization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    sanitized = {
        key: ("" if isinstance(value, float) and math.isnan(value) else value)
        for key, value in row.items()
    }
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(sanitized)


def sync_to_drive(local_dir: Path, drive_dir: Path) -> tuple[int, float]:
    """Copy changed files atomically while preserving non-fatal failures."""
    if not local_dir.exists():
        return 0, 0.0

    drive_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    started = time.time()
    try:
        for source in sorted(local_dir.iterdir()):
            if not source.is_file():
                continue
            target = drive_dir / source.name
            needs_copy = (
                not target.exists()
                or source.stat().st_mtime > target.stat().st_mtime
            )
            if needs_copy:
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
                n_copied += 1
    except Exception as error:
        print(
            f"  ⚠ drive sync failed ({type(error).__name__}: {error}); "
            f"continuing training",
            flush=True,
        )
        return n_copied, time.time() - started
    return n_copied, time.time() - started
