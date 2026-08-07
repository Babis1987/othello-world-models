"""Checkpoint payload construction shared by canonical AR trainers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from othello_thesis.training.ar_runtime import model_state_dict


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: Any,
    model_cfg: Any,
    step: int,
    pass_num: int,
    chunk_count: int,
    next_chunk_idx: int,
    games_seen: int,
    tokens_seen: int,
    *,
    format_id: str | None = None,
) -> dict[str, Any]:
    """Build the exact ordered Transformer-AR or Mamba-AR payload."""
    payload: dict[str, Any] = {}
    if format_id is not None:
        payload["format"] = format_id
    payload.update(
        {
            "model": model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "train_config": asdict(cfg),
            "model_config": asdict(model_cfg),
            "step": step,
            "pass_num": pass_num,
            "chunk_count": chunk_count,
            "next_chunk_idx": next_chunk_idx,
            "games_seen": games_seen,
            "tokens_seen": tokens_seen,
        }
    )
    return payload


def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: Any,
    model_cfg: Any,
    step: int,
    pass_num: int,
    chunk_count: int,
    next_chunk_idx: int,
    games_seen: int,
    tokens_seen: int,
    *,
    format_id: str | None = None,
) -> None:
    """Atomically persist an AR checkpoint with the historical filename flow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(
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
            format_id=format_id,
        ),
        temporary,
    )
    temporary.replace(path)
