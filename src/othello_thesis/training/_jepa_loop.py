"""Internal implementation module extracted from the canonical JEPA trainer.

The JEPA counterpart of ``training/ar_steps.py``: one shard in, one pass over
it, shard freed. The surrounding runner, checkpointing and config handling live
in the sibling ``_jepa_*`` modules; this file holds only the per-batch train and
eval loops.

Two supervision regimes live side by side, selected by
``cfg.position_sampling``:

*all-position* (``"all"``, the thesis setting). Every valid prefix boundary in
every game of the batch becomes a training pair, so one batch of 256 games
yields tens of thousands of (context, target) pairs. This is what makes the
JEPA arm sample-efficient enough to be trained on comparable compute to the AR
arm, where every position already produces a gradient signal for free.

*window sampling* (anything else, retained from the development runs). One
random context length is drawn per batch and only rows long enough for it are
kept, giving one pair per game. Cheaper per step but far weaker supervision.

Both regimes have to cope with the fact that games end at different lengths.
A prefix boundary is only usable if the context *and* the target are real
moves rather than padding, so batches and individual rows get skipped. The
skip counters threaded through both loops are not bookkeeping for its own sake:
a high skip ratio silently biases training toward long games, which are
systematically different positions (endgames, few legal moves). Hence the
warnings printed at the end of every shard.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Callable

import torch

from othello_thesis.models.transformer import GPTConfig
from othello_thesis.training._jepa_all_positions import (
    build_jepa_views,
    forward_all_position_v1,
    maybe_warn_collapse,
    output_metric,
    sample_valid_window,
    valid_window_mask,
)
from othello_thesis.training._jepa_config import (
    TrainConfig,
    jepa_loss_type,
    jepa_view_mode,
    normalize_multiaction_loss_mode,
    objective_class,
    objective_horizon,
)
from othello_thesis.training._jepa_reporting import objective_metric_keys
from othello_thesis.training._jepa_runtime import (
    autocast_dtype,
    get_lr,
    make_loader,
    update_target_encoder,
    use_autocast,
)
from othello_thesis.training.progress import (
    make_chunk_progress,
    tqdm_disabled as _tqdm_disabled,
)

def train_one_chunk(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    chunk_path: Path,
    cfg: TrainConfig,
    model_cfg: GPTConfig,
    device: torch.device,
    step: int,
    total_steps: int,
    step_callback: Callable[[int, int], None] | None = None,
    progress_desc: str | None = None,
) -> tuple[float, dict[str, float], int, int, int, int, int, int, int]:
    """Train one full JEPA pass over a single chunk.

    Returns:
        ``(loss, metric_avgs, new_step, n_games, n_samples, seen_batches,
        skipped_batches, seen_samples, skipped_samples)``. The sample count
        reuses the AR trainer's ``tokens_seen`` plumbing but represents
        batched JEPA windows.

    Losses and metrics are accumulated weighted by the number of pairs actually
    contributed, not per batch, because the effective batch size varies with how
    many rows survived the validity filter.

    ``step`` counts optimiser steps and is threaded in and out so the learning
    rate schedule is continuous across shard boundaries; the second element of
    the target-encoder update (``update_target_encoder``) is likewise tied to
    optimiser steps, not to shards.
    """
    loader, games, dataset = make_loader(chunk_path, cfg, model_cfg, shuffle=True)
    n_games = len(games)
    model.train()

    total_loss_sum = 0.0
    metric_keys = objective_metric_keys(cfg)
    metric_sums = {key: 0.0 for key in metric_keys}
    total_samples = 0
    seen_batches = 0
    skipped_batches = 0
    seen_samples = 0
    skipped_samples = 0

    pbar = make_chunk_progress(
        loader, desc=progress_desc or f"train {chunk_path.name}"
    )
    amp_enabled = use_autocast(cfg, device)
    amp_dtype = autocast_dtype(cfg)
    pad_token = cfg.board_size * cfg.board_size - 4
    horizon = objective_horizon(cfg)
    view_mode = jepa_view_mode(cfg)

    for batch_idx, (x, _) in enumerate(pbar):
        if cfg.max_batches_per_chunk and batch_idx >= cfg.max_batches_per_chunk:
            break
        x = x.to(device, non_blocking=True)
        batch_size, seq_len = x.shape
        seen_batches += 1
        seen_samples += batch_size
        if seq_len <= horizon:
            raise ValueError(
                f"Sequence length {seq_len} must exceed prediction_horizon={horizon}."
            )

        # ---- all-position supervision (the thesis path) --------------------
        # Every usable prefix boundary in the batch becomes one training pair.
        if cfg.position_sampling == "all":
            # Profile only the first batch of a capped (smoke-test) run; the
            # timing calls below force GPU syncs and would distort a real run.
            profile_step = bool(cfg.max_batches_per_chunk and seen_batches == 1)
            lr = get_lr(step, cfg, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                out, effective_batch_size = forward_all_position_v1(
                    model,
                    x,
                    pad_token=pad_token,
                    target_chunk_size=cfg.all_position_target_chunk_size,
                    stats_chunk_size=cfg.all_position_stats_chunk_size,
                    profile=profile_step,
                )
                # No usable prefix boundary anywhere in the batch: every game
                # was too short. Nothing to learn from, so skip without an
                # optimiser step (which would otherwise apply a stale gradient).
                if effective_batch_size == 0:
                    skipped_samples += batch_size
                    skipped_batches += 1
                    continue
                loss = out["loss"]

            # The mamba loss-finiteness check runs after the optimizer step,
            # piggybacking on the metric sync below: an eager pre-backward
            # isfinite() forces a full GPU sync that serializes the CPU launch
            # stream against the forward pass on every step.
            if profile_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_started_at = time.perf_counter()
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if profile_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            optimizer_started_at = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            # EMA update immediately after the optimiser step, so the target
            # encoder always trails the weights that were just written. Doing
            # it before the step would mix in the previous iteration's state.
            update_target_encoder(model)
            if profile_step:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                optimizer_finished_at = time.perf_counter()
                print(
                    "  all-position step profile: "
                    f"backward+clip={optimizer_started_at - backward_started_at:.3f}s "
                    f"optimizer={optimizer_finished_at - optimizer_started_at:.3f}s",
                    flush=True,
                )

            # This float() is the first host read of the step, so the
            # finiteness check rides on a sync that has to happen anyway.
            # Mamba-JEPA is the configuration that actually produced NaNs
            # during development (the selective scan can overflow in bf16),
            # hence the architecture-specific guard: failing loudly beats
            # discovering hours later that the checkpoint is poisoned.
            loss_value = float(loss.detach())
            if (
                getattr(cfg, "encoder_architecture", "transformer") == "mamba"
                and not math.isfinite(loss_value)
            ):
                raise FloatingPointError(
                    f"Non-finite JEPA loss at optimizer step {step + 1}"
                )
            total_loss_sum += loss_value * effective_batch_size
            for key in metric_keys:
                metric = output_metric(out, key)
                if isinstance(metric, torch.Tensor):
                    metric = metric.detach()
                metric_sums[key] += float(metric) * effective_batch_size
            total_samples += effective_batch_size
            step += 1
            if step_callback is not None:
                step_callback(step, total_samples)
            maybe_warn_collapse(cfg, out, step)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    pairs=effective_batch_size,
                    pos_per_game=f"{float(out['positions_per_game']):.1f}",
                    lr=f"{lr:.2e}",
                )
            continue

        # ---- window sampling (development path) ----------------------------
        # Use one random context length for the batch, then keep only rows
        # whose context plus K target tokens are real moves. This avoids
        # dropping an entire batch because one short game has padding.
        #
        # A single shared context length per batch is what lets the whole batch
        # go through the encoder as one rectangular tensor; per-row lengths
        # would require padding or a loop, both far slower.
        t, valid_mask = sample_valid_window(
            x,
            horizon=horizon,
            pad_token=pad_token,
            attempts=cfg.window_sample_attempts,
        )
        valid_count = int(valid_mask.sum().item())
        # Contrastive objectives need at least one negative, so a batch of one
        # surviving row is useless to them even though a distance loss could
        # still use it.
        needs_pair_batch = (
            view_mode in {"disjoint_future", "hard_disjoint_future"}
            or objective_class(cfg) == "jepa_contrastive"
            or (
                objective_class(cfg) == "jepa_multiaction"
                and normalize_multiaction_loss_mode(cfg.rollout_loss_mode) == "infonce"
            )
        )
        if valid_count == 0 or (needs_pair_batch and valid_count < 2):
            skipped_samples += batch_size
            skipped_batches += 1
            continue
        # Rows dropped by the validity filter still count as skipped, so the
        # ratio reported at the end reflects games lost, not just whole batches.
        skipped_samples += batch_size - valid_count

        x_valid = x[valid_mask]
        effective_batch_size = x_valid.size(0)
        x_ctx, x_tgt, target_positions = build_jepa_views(
            cfg,
            x_valid,
            context_length=t,
            horizon=horizon,
        )

        # Set learning rate for this step.
        lr = get_lr(step, cfg, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(x_ctx, x_tgt, target_positions)
            loss = out["loss"]

        # Mamba loss finiteness is checked after the optimizer step, at the
        # metric sync that already reads the loss back to the host.
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            # Only unscale if the scaler is actually active (CUDA + fp16).
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        update_target_encoder(model)

        loss_value = float(loss.detach())
        if (
            getattr(cfg, "encoder_architecture", "transformer") == "mamba"
            and not math.isfinite(loss_value)
        ):
            raise FloatingPointError(
                f"Non-finite JEPA loss at optimizer step {step + 1}"
            )
        total_loss_sum += loss_value * effective_batch_size
        for key in metric_keys:
            metric_sums[key] += float(output_metric(out, key)) * effective_batch_size
        total_samples += effective_batch_size

        step += 1
        if step_callback is not None:
            step_callback(step, total_samples)
        maybe_warn_collapse(cfg, out, step)
        if hasattr(pbar, "set_postfix"):
            postfix = {
                "loss": f"{loss_value:.4f}",
                "eff_bs": effective_batch_size,
                "lr": f"{lr:.2e}",
            }
            if objective_class(cfg) == "jepa_action_conditioned":
                postfix.update({
                    "main": f"{float(output_metric(out, 'main_loss')):.3f}",
                    "utility": f"{float(output_metric(out, 'context_utility')):.3f}",
                })
            elif objective_class(cfg) == "jepa_multiaction":
                postfix.update({
                    "rollout": f"{float(output_metric(out, 'rollout_loss')):.3f}",
                    "z_std": f"{float(output_metric(out, 'z_std')):.3f}",
                    "p_std": f"{float(output_metric(out, 'p_std')):.3f}",
                })
                if normalize_multiaction_loss_mode(cfg.rollout_loss_mode) == "infonce":
                    postfix["acc"] = f"{float(output_metric(out, 'positive_accuracy')):.3f}"
            elif objective_class(cfg) == "jepa_contrastive":
                postfix.update({
                    "pos_acc": f"{float(output_metric(out, 'positive_accuracy')):.3f}",
                    "diag_acc": f"{float(output_metric(out, 'diag_accuracy')):.3f}",
                    "pos": f"{float(output_metric(out, 'n_positives_mean')):.1f}",
                })
            elif jepa_loss_type(cfg) == "vicreg":
                postfix.update({
                    "inv": f"{float(out['inv_loss']):.3f}",
                    "var": f"{float(out['var_loss']):.3f}",
                    "cov": f"{float(out['cov_loss']):.3f}",
                })
            elif jepa_loss_type(cfg) == "mse":
                postfix.update({
                    "mse": f"{float(out['mse_loss']):.3f}",
                    "pred_std": f"{float(out['z_pred_std']):.3f}",
                    "tgt_std": f"{float(out['z_tgt_std']):.3f}",
                })
            else:
                postfix.update({
                    "pred_std": f"{float(out['z_pred_std']):.3f}",
                    "tgt_std": f"{float(out['z_tgt_std']):.3f}",
                })
            pbar.set_postfix(**postfix)

    # Clear the in-place progress line (also on early break) so the permanent
    # per-chunk metrics line prints on a clean row.
    if hasattr(pbar, "close"):
        pbar.close()

    # Free chunk memory before moving on.
    del loader, dataset, games
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    batch_skip_ratio = skipped_batches / max(1, seen_batches)
    sample_skip_ratio = skipped_samples / max(1, seen_samples)
    sampling_label = (
        "all-position supervision" if cfg.position_sampling == "all"
        else "valid-window sampling"
    )
    if not _tqdm_disabled():
        print(
            f"  {sampling_label}: skipped_batches={skipped_batches}/{seen_batches} "
            f"({batch_skip_ratio:.2%}) skipped_samples={skipped_samples}/{seen_samples} "
            f"({sample_skip_ratio:.2%})",
            flush=True,
        )
    # Skipped samples are not random: short games are dropped preferentially,
    # so a high ratio means the model is being trained on a biased slice of the
    # position distribution. These thresholds were chosen as the point beyond
    # which that bias becomes large enough to affect the reported metrics.
    if sample_skip_ratio > 0.10:
        print(
            "  warning: train_sample_skip_ratio is above 10%; inspect game lengths "
            "before trusting a full run.",
            flush=True,
        )
    elif sample_skip_ratio > 0.05:
        print(
            "  warning: train_sample_skip_ratio is above 5%; monitor this run closely.",
            flush=True,
        )

    denom = max(1, total_samples)
    metric_avgs = {key: value / denom for key, value in metric_sums.items()}
    return (
        total_loss_sum / denom,
        metric_avgs,
        step,
        min(n_games, seen_samples),
        total_samples,
        seen_batches,
        skipped_batches,
        seen_samples,
        skipped_samples,
    )

@torch.no_grad()
def eval_chunks(
    model: torch.nn.Module,
    chunks: list[Path],
    cfg: TrainConfig,
    model_cfg: GPTConfig,
    device: torch.device,
) -> tuple[float, dict[str, float], int, int, int, int, int, int]:
    """Run JEPA evaluation on the first cfg.eval_chunks val chunks.

    Mirrors ``train_one_chunk`` without the optimiser, and with every source of
    randomness removed so the figure is comparable across checkpoints: always
    the same leading shards, shuffling off, and a deterministic context length
    per batch instead of a sampled one.
    """
    if not chunks or cfg.eval_chunks <= 0:
        nan = float("nan")
        return nan, {}, 0, 0, 0, 0, 0, 0

    model.eval()
    total_loss_sum = 0.0
    metric_keys = objective_metric_keys(cfg)
    metric_sums = {key: 0.0 for key in metric_keys}
    total_samples = 0
    total_games = 0
    seen_batches = 0
    skipped_batches = 0
    seen_samples = 0
    skipped_samples = 0
    amp_enabled = use_autocast(cfg, device)
    amp_dtype = autocast_dtype(cfg)
    pad_token = cfg.board_size * cfg.board_size - 4
    horizon = objective_horizon(cfg)
    view_mode = jepa_view_mode(cfg)

    for chunk_path in chunks[: cfg.eval_chunks]:
        loader, games, dataset = make_loader(chunk_path, cfg, model_cfg, shuffle=False)
        total_games += len(games)
        for batch_idx, (x, _) in enumerate(
            make_chunk_progress(loader, desc=f"eval {chunk_path.name}")
        ):
            x = x.to(device, non_blocking=True)
            batch_size, seq_len = x.shape
            seen_batches += 1
            seen_samples += batch_size
            if seq_len <= horizon:
                raise ValueError(
                    f"Sequence length {seq_len} must exceed prediction_horizon={horizon}."
                )

            if cfg.position_sampling == "all":
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    out, effective_batch_size = forward_all_position_v1(
                        model,
                        x,
                        pad_token=pad_token,
                        target_chunk_size=cfg.all_position_target_chunk_size,
                        stats_chunk_size=cfg.all_position_stats_chunk_size,
                    )
                    if effective_batch_size == 0:
                        skipped_samples += batch_size
                        skipped_batches += 1
                        continue
                    loss = out["loss"]
                total_loss_sum += float(loss.detach()) * effective_batch_size
                for key in metric_keys:
                    metric_sums[key] += (
                        float(output_metric(out, key)) * effective_batch_size
                    )
                total_samples += effective_batch_size
                continue

            # Deterministic sweep over possible context lengths keeps eval
            # comparable across runs without introducing variable-length batches.
            # Cycling t with the batch index means the evaluation still covers
            # the full range of game phases, but always in the same order.
            t = 1 + (batch_idx % max(1, seq_len - horizon))
            valid_mask = valid_window_mask(
                x,
                context_length=t,
                horizon=horizon,
                pad_token=pad_token,
            )
            valid_count = int(valid_mask.sum().item())
            needs_pair_batch = (
                view_mode in {"disjoint_future", "hard_disjoint_future"}
                or objective_class(cfg) == "jepa_contrastive"
                or (
                    objective_class(cfg) == "jepa_multiaction"
                    and normalize_multiaction_loss_mode(cfg.rollout_loss_mode) == "infonce"
                )
            )
            if valid_count == 0 or (needs_pair_batch and valid_count < 2):
                skipped_samples += batch_size
                skipped_batches += 1
                continue
            skipped_samples += batch_size - valid_count
            x_valid = x[valid_mask]
            effective_batch_size = x_valid.size(0)
            x_ctx, x_tgt, target_positions = build_jepa_views(
                cfg,
                x_valid,
                context_length=t,
                horizon=horizon,
            )

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(x_ctx, x_tgt, target_positions)
                loss = out["loss"]

            total_loss_sum += float(loss.detach()) * effective_batch_size
            for key in metric_keys:
                metric_sums[key] += float(output_metric(out, key)) * effective_batch_size
            total_samples += effective_batch_size
        del loader, dataset, games
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
        skipped_batches,
        seen_samples,
        skipped_samples,
    )


# The two dispatchers below are the same guardrail as the normalisers in
# objectives/jepa.py, applied at the trainer level: the development tree
# supported several objective classes, and a config that selected one of them
# would silently produce numbers that are not comparable with the reported
# cells. Only the final 'jepa' path is allowed to run.

def dispatch_train_one_chunk(*args, **kwargs):
    """Dispatch the canonical final JEPA training path."""
    cfg = kwargs.get("cfg") if "cfg" in kwargs else args[4]
    resolved = objective_class(cfg)
    if resolved != "jepa":
        raise ValueError(
            "The final trainer supports only objective_class='jepa'; "
            f"got {resolved!r}."
        )
    return train_one_chunk(*args, **kwargs)


def dispatch_eval_chunks(*args, **kwargs):
    """Dispatch the canonical final JEPA evaluation path."""
    cfg = kwargs.get("cfg") if "cfg" in kwargs else args[2]
    resolved = objective_class(cfg)
    if resolved != "jepa":
        raise ValueError(
            "The final trainer supports only objective_class='jepa'; "
            f"got {resolved!r}."
        )
    return eval_chunks(*args, **kwargs)
