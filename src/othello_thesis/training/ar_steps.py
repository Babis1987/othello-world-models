"""Numerically locked single-shard train/eval steps for canonical AR runs."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Literal

import torch

from othello_thesis.data.chunk_dataset import TARGET_PAD
from othello_thesis.training.ar_data import make_chunk_loader
from othello_thesis.training.ar_runtime import autocast_context, cosine_learning_rate
from othello_thesis.training.progress import make_chunk_progress


ARStyle = Literal["transformer", "mamba"]


def train_one_chunk(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    chunk_path: Path,
    cfg: Any,
    model_cfg: Any,
    device: torch.device,
    step: int,
    total_steps: int,
    *,
    style: ARStyle,
    progress_desc: str | None = None,
) -> tuple[float, float, int, int, int]:
    """Run the historical per-batch operations in their original order."""
    loader, games, dataset = make_chunk_loader(
        chunk_path,
        block_size=model_cfg.block_size,
        board_size=cfg.board_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    n_games = len(games)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    progress = make_chunk_progress(
        loader, desc=progress_desc or f"train {chunk_path.name}"
    )
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        learning_rate = cosine_learning_rate(step, cfg, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.precision):
            logits, loss = model(x, y)
        if style == "mamba" and loss is None:
            raise RuntimeError("Model returned loss=None during training")

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            mask = y != TARGET_PAD
            n_tokens = int(mask.sum().item())
            predictions = logits.argmax(dim=-1)
            correct = int(((predictions == y) & mask).sum().item())
            if style == "mamba":
                total_loss += float(loss.detach().item()) * n_tokens
            else:
                total_loss += loss.item() * n_tokens
            total_correct += correct
            total_tokens += n_tokens

        step += 1
        if hasattr(progress, "set_postfix"):
            if style == "mamba":
                progress.set_postfix(
                    loss=f"{total_loss / max(1, total_tokens):.4f}",
                    acc=f"{total_correct / max(1, total_tokens):.4f}",
                    lr=f"{learning_rate:.2e}",
                )
            else:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{correct / max(1, n_tokens):.2%}",
                    lr=f"{learning_rate:.2e}",
                )

    if style == "mamba" and hasattr(progress, "close"):
        progress.close()
    del loader, dataset, games
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        total_loss / max(1, total_tokens),
        total_correct / max(1, total_tokens),
        step,
        n_games,
        total_tokens,
    )


@torch.no_grad()
def evaluate_chunks(
    model: torch.nn.Module,
    chunks: list[Path],
    cfg: Any,
    model_cfg: Any,
    device: torch.device,
    *,
    style: ARStyle,
) -> tuple[float, float, int, int]:
    """Evaluate the same leading held-out chunks and token reductions."""
    if not chunks or cfg.eval_chunks <= 0:
        return float("nan"), float("nan"), 0, 0

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_games = 0
    for chunk_path in chunks[: cfg.eval_chunks]:
        loader, games, dataset = make_chunk_loader(
            chunk_path,
            block_size=model_cfg.block_size,
            board_size=cfg.board_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )
        total_games += len(games)
        for x, y in make_chunk_progress(loader, desc=f"eval {chunk_path.name}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(device, cfg.precision):
                logits, loss = model(x, y)
            if style == "mamba" and loss is None:
                raise RuntimeError("Model returned loss=None during eval")
            mask = y != TARGET_PAD
            n_tokens = int(mask.sum().item())
            predictions = logits.argmax(dim=-1)
            if style == "mamba":
                total_loss += float(loss.detach().item()) * n_tokens
            else:
                total_loss += loss.item() * n_tokens
            total_correct += int(((predictions == y) & mask).sum().item())
            total_tokens += n_tokens
        del loader, dataset, games
        gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        total_loss / max(1, total_tokens),
        total_correct / max(1, total_tokens),
        total_games,
        total_tokens,
    )
