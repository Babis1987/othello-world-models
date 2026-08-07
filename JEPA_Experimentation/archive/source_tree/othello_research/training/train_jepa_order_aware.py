"""Grouped train/eval path for v8 order-aware JEPA."""

from __future__ import annotations

from collections import deque
import gc
from pathlib import Path
from typing import Callable

import torch

from othello_research.training.progress import make_manual_progress


def make_order_aware_batches(chunk_path: Path, cfg, seed_offset: int = 0):
    from othello_research.datasets.order_aware_dataset import (
        OrderAwareBatchConfig,
        OrderAwareChunkBatches,
    )

    effective_batch = cfg.order_groups_per_batch * cfg.order_samples_per_group
    if cfg.batch_size != effective_batch:
        raise ValueError(
            "For v8, batch_size must equal "
            "order_groups_per_batch * order_samples_per_group "
            f"({effective_batch}), got {cfg.batch_size}"
        )
    if not cfg.pair_index_path:
        raise ValueError("v8 requires pair_index_path")
    if not cfg.data_dir:
        raise ValueError("v8 currently requires data_dir to resolve pair references")
    batch_cfg = OrderAwareBatchConfig(
        board_size=cfg.board_size,
        t_min=cfg.order_t_min,
        t_max=cfg.order_t_max,
        groups_per_batch=cfg.order_groups_per_batch,
        samples_per_group=cfg.order_samples_per_group,
        hard_negatives_per_anchor=cfg.order_hard_negatives_per_anchor,
        chunk_sample_multiplier=cfg.order_chunk_sample_multiplier,
        pair_chunk_cache_size=cfg.order_pair_chunk_cache_size,
        min_pair_anchors_per_group=cfg.order_min_pair_anchors_per_group,
        expected_use_constructive_pairs=cfg.order_index_use_constructive_pairs,
        expected_use_surface_hard_negatives=cfg.order_index_use_surface_hard_negatives,
        expected_surface_jaccard_threshold=cfg.order_surface_jaccard_threshold,
        expected_positions_per_game=cfg.order_index_positions_per_game,
        seed=cfg.seed + seed_offset,
    )
    return OrderAwareChunkBatches(
        chunk_path, cfg.pair_index_path, cfg.data_dir, batch_cfg
    )


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in (
            "x_context",
            "x_positive_target",
            "x_hard_negative_targets",
            "hard_negative_mask",
            "group_ids",
            "positive_fallback_mask",
        )
    }


def train_one_chunk_order_aware(
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

    batches = make_order_aware_batches(chunk_path, cfg, seed_offset=step)
    n_games = len(batches.games)
    model.train()
    raw_model = common.unwrap_model(model)
    metric_keys = common.objective_metric_keys(cfg)
    metric_sums = {key: 0.0 for key in metric_keys}
    total_loss_sum = 0.0
    total_samples = 0
    seen_batches = 0
    amp_enabled = common.use_autocast(cfg, device)
    amp_dtype = common.autocast_dtype(cfg)

    pbar = make_manual_progress(
        progress_desc or f"train(v8) {chunk_path.name}"
    )
    for batch in batches:
        tensors = _move_batch(batch, device)
        batch_size = tensors["x_context"].size(0)
        seen_batches += 1
        lr = common.get_lr(step, cfg, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            out = raw_model(**tensors)
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
        if cfg.order_utility_early_stop_threshold >= 0:
            history = getattr(raw_model, "_order_utility_history", None)
            if history is None or history.maxlen != cfg.order_utility_early_stop_window:
                history = deque(maxlen=cfg.order_utility_early_stop_window)
                raw_model._order_utility_history = history
            history.append(float(out["context_utility"].detach()))
            if (
                step >= cfg.order_utility_early_stop_min_steps
                and len(history) == history.maxlen
                and sum(history) / len(history)
                <= cfg.order_utility_early_stop_threshold
            ):
                raw_model._order_early_stop_requested = True
                break
        if hasattr(pbar, "update"):
            pbar.update(1)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                acc=f"{float(out['positive_accuracy']):.3f}",
                hard=f"{float(out['mean_hard_negatives']):.2f}",
                fallback=f"{float(out['positive_fallback_rate']):.2%}",
                lr=f"{lr:.2e}",
            )
    if hasattr(pbar, "close"):
        pbar.close()
    batches.close()
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
        total_samples,
        0,
    )


@torch.no_grad()
def eval_chunks_order_aware(
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
    amp_enabled = common.use_autocast(cfg, device)
    amp_dtype = common.autocast_dtype(cfg)

    for chunk_path in chunks[: cfg.eval_chunks]:
        batches = make_order_aware_batches(chunk_path, cfg, seed_offset=1_000_003)
        total_games += len(batches.games)
        for batch in batches:
            tensors = _move_batch(batch, device)
            batch_size = tensors["x_context"].size(0)
            seen_batches += 1
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                out = raw_model(**tensors)
                loss = out["loss"]
            total_loss_sum += float(loss.detach()) * batch_size
            for key in metric_keys:
                metric_sums[key] += float(common.output_metric(out, key)) * batch_size
            total_samples += batch_size
        batches.close()
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
        total_samples,
        0,
    )
