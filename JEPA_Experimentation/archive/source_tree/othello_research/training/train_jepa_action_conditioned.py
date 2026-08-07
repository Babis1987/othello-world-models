"""Grouped train/eval path for v7 grouped InfoNCE and hybrid objectives."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import torch

from othello_research.training.progress import (
    make_manual_progress,
    tqdm_disabled,
)


def make_action_grouped_batches(chunk_path: Path, cfg, seed_offset: int = 0):
    from othello_research.datasets.action_grouped_dataset import (
        ActionGroupedBatchConfig,
        ActionGroupedChunkBatches,
    )

    grouped_batch_size = cfg.action_groups_per_batch * cfg.action_samples_per_group
    if cfg.batch_size != grouped_batch_size:
        raise ValueError(
            "For v7 grouped losses, batch_size must equal "
            "action_groups_per_batch * action_samples_per_group "
            f"({grouped_batch_size}), got {cfg.batch_size}"
        )
    batch_cfg = ActionGroupedBatchConfig(
        board_size=cfg.board_size,
        t_min=cfg.action_t_min,
        t_max=cfg.action_t_max,
        groups_per_batch=cfg.action_groups_per_batch,
        samples_per_group=cfg.action_samples_per_group,
        chunk_sample_multiplier=cfg.action_chunk_sample_multiplier,
        seed=cfg.seed + seed_offset,
    )
    return ActionGroupedChunkBatches(chunk_path, batch_cfg)


def train_one_chunk_action_grouped(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    chunk_path: Path,
    cfg,
    model_cfg,
    device: torch.device,
    step: int,
    total_steps: int,
    step_callback: Callable[[int, int], None] | None = None,
    progress_desc: str | None = None,
):
    del model_cfg
    from othello_research.training import train_jepa as common

    batches = make_action_grouped_batches(chunk_path, cfg, seed_offset=step)
    if not tqdm_disabled():
        # Streamed runs keep one permanent row per chunk; this diagnostic would
        # add a second. The metrics row and metrics.csv still carry the run.
        print(f"  v7 action-bucket coverage: {batches.diagnostics}", flush=True)
    n_games = len(batches.games)
    model.train()
    raw_model = common.unwrap_model(model)

    metric_keys = common.objective_metric_keys(cfg)
    metric_sums = {key: 0.0 for key in metric_keys}
    total_loss_sum = 0.0
    total_samples = 0
    seen_batches = 0
    seen_samples = 0
    amp_enabled = common.use_autocast(cfg, device)
    amp_dtype = common.autocast_dtype(cfg)

    pbar = make_manual_progress(
        progress_desc or f"train({cfg.variant}) {chunk_path.name}"
    )
    for batch in batches:
        x_context = batch["x_context"].to(device, non_blocking=True)
        x_target = batch["x_target"].to(device, non_blocking=True)
        group_ids = batch["group_ids"].to(device, non_blocking=True)
        batch_size = x_context.size(0)
        seen_batches += 1
        seen_samples += batch_size

        lr = common.get_lr(step, cfg, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            out = raw_model(x_context, x_target, group_ids=group_ids)
            loss = out["loss"]

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        common.update_target_encoder(model)

        total_loss_sum += float(loss.detach()) * batch_size
        for key in metric_keys:
            metric_sums[key] += float(common.output_metric(out, key)) * batch_size
        total_samples += batch_size
        step += 1

        if step_callback is not None:
            step_callback(step, total_samples)
        common.maybe_warn_collapse(cfg, out, step)
        if hasattr(pbar, "update"):
            pbar.update(1)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                main=f"{float(out['main_loss']):.4f}",
                acc=f"{float(out['positive_accuracy']):.3f}",
                utility=f"{float(out['context_utility']):.3f}",
                lr=f"{lr:.2e}",
            )
    if hasattr(pbar, "close"):
        pbar.close()

    del batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    denom = max(1, total_samples)
    metric_avgs = {key: value / denom for key, value in metric_sums.items()}
    return (
        total_loss_sum / denom,
        metric_avgs,
        step,
        n_games,
        total_samples,
        seen_batches,
        0,
        seen_samples,
        0,
    )


@torch.no_grad()
def eval_chunks_action_grouped(
    model: torch.nn.Module,
    chunks: list[Path],
    cfg,
    model_cfg,
    device: torch.device,
):
    del model_cfg
    from othello_research.training import train_jepa as common

    if not chunks or cfg.eval_chunks <= 0:
        nan = float("nan")
        return nan, {}, 0, 0, 0, 0, 0, 0

    model.eval()
    raw_model = common.unwrap_model(model)
    metric_keys = common.objective_metric_keys(cfg)
    metric_sums = {key: 0.0 for key in metric_keys}
    total_loss_sum = 0.0
    total_samples = 0
    total_games = 0
    seen_batches = 0
    seen_samples = 0
    amp_enabled = common.use_autocast(cfg, device)
    amp_dtype = common.autocast_dtype(cfg)

    for chunk_path in chunks[: cfg.eval_chunks]:
        batches = make_action_grouped_batches(chunk_path, cfg, seed_offset=1_000_003)
        total_games += len(batches.games)
        for batch in batches:
            x_context = batch["x_context"].to(device, non_blocking=True)
            x_target = batch["x_target"].to(device, non_blocking=True)
            group_ids = batch["group_ids"].to(device, non_blocking=True)
            batch_size = x_context.size(0)
            seen_batches += 1
            seen_samples += batch_size

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                out = raw_model(x_context, x_target, group_ids=group_ids)
                loss = out["loss"]
            total_loss_sum += float(loss.detach()) * batch_size
            for key in metric_keys:
                metric_sums[key] += float(common.output_metric(out, key)) * batch_size
            total_samples += batch_size
        del batches
        gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    denom = max(1, total_samples)
    metric_avgs = {key: value / denom for key, value in metric_sums.items()}
    return (
        total_loss_sum / denom,
        metric_avgs,
        total_games,
        total_samples,
        seen_batches,
        0,
        seen_samples,
        0,
    )
