"""Internal implementation module extracted from the canonical JEPA trainer."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from othello_thesis.training._jepa_config import (
    TrainConfig,
    objective_class,
    write_config_records,
)
from othello_thesis.training._jepa_cli import (
    namespace_to_train_config,
    parse_configured_args,
    validate_train_config,
)
from othello_thesis.training._jepa_loop import (
    dispatch_eval_chunks,
    dispatch_train_one_chunk,
)
from othello_thesis.training._jepa_models import (
    ModelFactory,
    build_model_from_objective_config,
    build_objective_config,
)
from othello_thesis.training._jepa_reporting import (
    format_objective_header,
    format_objective_metrics,
    objective_metric_keys,
)
from othello_thesis.training._jepa_runtime import (
    append_csv,
    estimate_total_steps,
    fraction_checkpoint_name,
    init_wandb,
    load_model_state,
    parse_fraction_checkpoints,
    parse_milestones,
    resolve_chunks,
    save_checkpoint,
    sync_to_drive,
)
from othello_thesis.training.progress import tqdm, tqdm_disabled as _tqdm_disabled


TrainChunkDispatcher = Callable[..., tuple]
EvalChunkDispatcher = Callable[..., tuple]
StateLoader = Callable[[torch.nn.Module, dict], None]


def _run_training(
    cfg: TrainConfig,
    model_factory: ModelFactory | None = None,
    *,
    _source_config: Path | None = None,
    _source_data: dict[str, Any] | None = None,
    _objective_config_factory: Callable[[TrainConfig], Any] = build_objective_config,
    _train_dispatch: TrainChunkDispatcher = dispatch_train_one_chunk,
    _eval_dispatch: EvalChunkDispatcher = dispatch_eval_chunks,
    _state_loader: StateLoader = load_model_state,
    _validated: bool = False,
) -> None:
    """Train one resolved canonical JEPA configuration.

    ``model_factory`` receives the resolved objective config and is invoked
    exactly once, after all global RNGs have been seeded.  This lets notebooks
    show model construction explicitly without a preview model that would
    consume RNG state and alter the experiment.

    Underscore-prefixed collaborators are internal seams used by the Mamba
    facade; ordinary callers should pass only ``cfg`` and ``model_factory``.
    """
    if not _validated:
        validate_train_config(cfg)
    resolved_objective = objective_class(cfg)
    if resolved_objective != "jepa":
        raise ValueError(
            "Master_Thesis_Code_Final trains only objective_class='jepa'; "
            f"got {resolved_objective!r}."
        )

    # Reproducibility.
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _source_config is not None:
        write_config_records(
            source_config=_source_config,
            source_data=_source_data or {},
            resolved_args=asdict(cfg),
        )
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    drive_sync_dir = Path(cfg.drive_sync_dir) if cfg.drive_sync_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_chunks, val_chunks = resolve_chunks(cfg)

    # Resolve the objective without touching RNG, then construct exactly once.
    jepa_cfg = _objective_config_factory(cfg)
    factory = model_factory or build_model_from_objective_config
    model = factory(jepa_cfg)
    model = model.to(device)
    model_cfg = model.config
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and cfg.precision == "fp16"
    )

    # State that survives a resume.
    step = 0
    start_pass = 1
    start_chunk_idx = 0
    chunk_count = 0
    games_seen = 0
    tokens_seen = 0
    completed_milestones: set[int] = set()

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        _state_loader(model, ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt.get("step", 0)
        start_pass = ckpt.get("pass_num", 1)
        start_chunk_idx = ckpt.get("next_chunk_idx", 0)
        chunk_count = ckpt.get("chunk_count", 0)
        games_seen = ckpt.get("games_seen", 0)
        tokens_seen = ckpt.get("tokens_seen", 0)
        completed_milestones = {
            m for m in parse_milestones(cfg.milestone_games) if m <= games_seen
        }
        if start_chunk_idx >= len(train_chunks):
            start_pass += 1
            start_chunk_idx = 0
        print(f"resumed from {cfg.resume}: step={step} games_seen={games_seen:,} "
              f"pass={start_pass} chunk_idx={start_chunk_idx}", flush=True)

    total_steps, total_games_estimate = estimate_total_steps(train_chunks, cfg)
    milestones = parse_milestones(cfg.milestone_games)
    fraction_checkpoints = parse_fraction_checkpoints(cfg.intermediate_checkpoint_fractions)
    completed_fraction_checkpoints: set[float] = {
        fraction
        for fraction in fraction_checkpoints
        if step >= max(1, math.ceil(total_steps * fraction))
    }
    wandb_run = init_wandb(cfg, jepa_cfg)
    metrics_path = out_dir / "metrics.csv"
    print(f"device: {device}", flush=True)
    print(f"train chunks: {len(train_chunks)}", flush=True)
    print(f"val chunks: {len(val_chunks)}", flush=True)
    print(f"vocab_size: {model_cfg.vocab_size}, block_size: {model_cfg.block_size}", flush=True)
    print(format_objective_header(cfg, model), flush=True)
    print(f"milestone games: {milestones}", flush=True)
    print(f"fraction checkpoints: {fraction_checkpoints}", flush=True)
    if drive_sync_dir is not None:
        print(f"drive auto-sync: every {cfg.drive_sync_every_chunks} chunk(s) -> {drive_sync_dir}", flush=True)

    # Outer progress bar across all chunks of the run.
    total_chunks_to_train = (
        cfg.max_chunks
        if cfg.max_chunks
        else len(train_chunks) * cfg.passes_over_data
    )
    outer_bar = tqdm(
        total=total_chunks_to_train,
        initial=chunk_count,
        desc="run",
        unit="chunk",
        leave=True,
        disable=_tqdm_disabled(),
    )

    for pass_num in range(start_pass, cfg.passes_over_data + 1):
        # Deterministic per-pass shuffle. With a fixed seed this is
        # reproducible across resumes.
        chunk_order = train_chunks[:]
        random.Random(cfg.seed + pass_num).shuffle(chunk_order)
        first_idx = start_chunk_idx if pass_num == start_pass else 0

        for ci in range(first_idx, len(chunk_order)):
            chunk_path = chunk_order[ci]
            t0 = time.time()
            chunk_progress_desc = (
                f"chunk {chunk_count + 1}/{total_chunks_to_train} {chunk_path.name}"
            )
            if not _tqdm_disabled():
                # The in-place bar already carries this header on its own line.
                # Printing it again would cost a second permanent row per chunk.
                print(
                    f"chunk {chunk_count + 1}/{total_chunks_to_train} starting: "
                    f"{chunk_path.name} (pass {pass_num}, "
                    f"order {ci + 1}/{len(chunk_order)})",
                    flush=True,
                )
            pre_chunk_games_seen = games_seen
            pre_chunk_tokens_seen = tokens_seen

            def save_step_fraction_checkpoint(current_step: int, chunk_samples_seen: int) -> None:
                for fraction in fraction_checkpoints:
                    threshold_step = max(1, math.ceil(total_steps * fraction))
                    if (
                        current_step >= threshold_step
                        and fraction not in completed_fraction_checkpoints
                    ):
                        completed_fraction_checkpoints.add(fraction)
                        checkpoint_path = out_dir / fraction_checkpoint_name(fraction)
                        save_checkpoint(
                            checkpoint_path,
                            model,
                            optimizer,
                            scaler,
                            cfg,
                            jepa_cfg,
                            current_step,
                            pass_num,
                            chunk_count,
                            ci,
                            pre_chunk_games_seen,
                            pre_chunk_tokens_seen + chunk_samples_seen,
                        )
                        print(
                            f"  saved fraction checkpoint {checkpoint_path.name} "
                            f"at step={current_step}/{total_steps}",
                            flush=True,
                        )
            (
                train_loss,
                train_metrics,
                step,
                chunk_games,
                chunk_tokens,
                train_seen_batches,
                train_skipped_batches,
                train_seen_samples,
                train_skipped_samples,
            ) = _train_dispatch(
                model,
                optimizer,
                scaler,
                chunk_path,
                cfg,
                model_cfg,
                device,
                step,
                total_steps,
                step_callback=save_step_fraction_checkpoint,
                progress_desc=chunk_progress_desc,
            )
            chunk_count += 1
            games_seen += chunk_games
            tokens_seen += chunk_tokens

            # Periodic evaluation on the held-out val chunks.
            val_loss = float("nan")
            val_metrics: dict[str, float] = {}
            val_games = 0
            val_tokens = 0
            val_seen_batches = 0
            val_skipped_batches = 0
            val_seen_samples = 0
            val_skipped_samples = 0
            should_eval = (
                bool(val_chunks)
                and cfg.eval_chunks > 0
                and chunk_count % max(1, cfg.eval_every_chunks) == 0
            )
            if should_eval:
                (
                    val_loss,
                    val_metrics,
                    val_games,
                    val_tokens,
                    val_seen_batches,
                    val_skipped_batches,
                    val_seen_samples,
                    val_skipped_samples,
                ) = _eval_dispatch(
                    model, val_chunks, cfg, model_cfg, device
                )

            lr = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0
            train_batch_skip_ratio = train_skipped_batches / max(1, train_seen_batches)
            train_sample_skip_ratio = train_skipped_samples / max(1, train_seen_samples)
            val_batch_skip_ratio = val_skipped_batches / max(1, val_seen_batches)
            val_sample_skip_ratio = val_skipped_samples / max(1, val_seen_samples)

            row = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pass": pass_num,
                "chunk_count": chunk_count,
                "chunk_index_in_pass": ci,
                "step": step,
                "games_seen": games_seen,
                "tokens_seen": tokens_seen,
                "chunk_games": chunk_games,
                "chunk_tokens": chunk_tokens,
                "train_seen_batches": train_seen_batches,
                "train_skipped_batches": train_skipped_batches,
                "train_batch_skip_ratio": train_batch_skip_ratio,
                "train_seen_samples": train_seen_samples,
                "train_skipped_samples": train_skipped_samples,
                "train_sample_skip_ratio": train_sample_skip_ratio,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_games": val_games,
                "val_tokens": val_tokens,
                "val_seen_batches": val_seen_batches,
                "val_skipped_batches": val_skipped_batches,
                "val_batch_skip_ratio": val_batch_skip_ratio,
                "val_seen_samples": val_seen_samples,
                "val_skipped_samples": val_skipped_samples,
                "val_sample_skip_ratio": val_sample_skip_ratio,
                "lr": lr,
                "dt_seconds": dt,
                "chunk_file": chunk_path.name,
            }
            for key in objective_metric_keys(cfg):
                row[key] = train_metrics.get(key, float("nan"))
                row[f"val_{key}"] = val_metrics.get(key, float("nan"))
            append_csv(metrics_path, row)
            if wandb_run is not None:
                wandb_run.log(
                    {k: v for k, v in row.items()
                     if k not in {"time", "chunk_file"}
                     and not (isinstance(v, float) and math.isnan(v))},
                    step=step,
                )

            val_text = "" if math.isnan(val_loss) else (
                f" | val: loss={val_loss:.4f} "
                f"{format_objective_metrics(cfg, val_metrics)} "
                f"sample_skip={val_sample_skip_ratio:.2%}"
            )
            print(
                f"jepa pass={pass_num} "
                f"chunk={chunk_count}/{total_chunks_to_train} step={step} "
                f"games={games_seen:,} samples={tokens_seen:,} "
                f"{format_objective_metrics(cfg, train_metrics)} "
                f"loss={train_loss:.4f} "
                f"sample_skip={train_sample_skip_ratio:.2%}"
                f"{val_text} dt={dt:.0f}s",
                flush=True,
            )

            outer_bar.update(1)
            if hasattr(outer_bar, "set_postfix"):
                outer_bar.set_postfix(
                    games=f"{games_seen / 1e6:.1f}M",
                    loss=f"{train_loss:.3f}",
                )

            # Always save latest.pt for Colab session resume.
            save_checkpoint(
                out_dir / "latest.pt",
                model, optimizer, scaler, cfg, jepa_cfg,
                step, pass_num, chunk_count, ci + 1, games_seen, tokens_seen,
            )

            # Milestone checkpoints (1M, 5M, 10M, ... games).
            for milestone in milestones:
                if games_seen >= milestone and milestone not in completed_milestones:
                    completed_milestones.add(milestone)
                    save_checkpoint(
                        out_dir / f"checkpoint_games_{milestone // 1_000_000:03d}M.pt",
                        model, optimizer, scaler, cfg, jepa_cfg,
                        step, pass_num, chunk_count, ci + 1, games_seen, tokens_seen,
                    )

            # Optional periodic chunk-indexed checkpoints.
            if cfg.checkpoint_every_chunks and chunk_count % cfg.checkpoint_every_chunks == 0:
                save_checkpoint(
                    out_dir / f"checkpoint_chunk_{chunk_count:06d}.pt",
                    model, optimizer, scaler, cfg, jepa_cfg,
                    step, pass_num, chunk_count, ci + 1, games_seen, tokens_seen,
                )

            # Fallback for thresholds that were crossed by a skipped/no-step chunk.
            for fraction in fraction_checkpoints:
                threshold_step = max(1, math.ceil(total_steps * fraction))
                if step >= threshold_step and fraction not in completed_fraction_checkpoints:
                    completed_fraction_checkpoints.add(fraction)
                    checkpoint_path = out_dir / fraction_checkpoint_name(fraction)
                    save_checkpoint(
                        checkpoint_path,
                        model, optimizer, scaler, cfg, jepa_cfg,
                        step, pass_num, chunk_count, ci + 1, games_seen, tokens_seen,
                    )
                    print(
                        f"  saved fraction checkpoint {checkpoint_path.name} "
                        f"at step={step}/{total_steps}",
                        flush=True,
                    )

            # Auto-sync to Drive after every drive_sync_every_chunks chunks.
            if drive_sync_dir is not None and chunk_count % max(1, cfg.drive_sync_every_chunks) == 0:
                n_synced, sync_dt = sync_to_drive(out_dir, drive_sync_dir)
                if n_synced > 0:
                    print(f"  drive sync: {n_synced} file(s) in {sync_dt:.1f}s", flush=True)

            # Stop after max_chunks, if requested.
            if cfg.max_chunks and chunk_count >= cfg.max_chunks:
                save_checkpoint(
                    out_dir / "final.pt",
                    model, optimizer, scaler, cfg, jepa_cfg,
                    step, pass_num, chunk_count, ci + 1, games_seen, tokens_seen,
                )
                # Final sync to Drive before exiting.
                if drive_sync_dir is not None:
                    n_synced, sync_dt = sync_to_drive(out_dir, drive_sync_dir)
                    print(f"  final drive sync: {n_synced} file(s) in {sync_dt:.1f}s", flush=True)
                outer_bar.close()
                if wandb_run is not None:
                    wandb_run.finish()
                return

    # Reached the end of the run normally.
    save_checkpoint(
        out_dir / "final.pt",
        model, optimizer, scaler, cfg, jepa_cfg,
        step, cfg.passes_over_data, chunk_count, len(train_chunks),
        games_seen, tokens_seen,
    )
    # Final sync to Drive.
    if drive_sync_dir is not None:
        n_synced, sync_dt = sync_to_drive(out_dir, drive_sync_dir)
        print(f"  final drive sync: {n_synced} file(s) in {sync_dt:.1f}s", flush=True)
    outer_bar.close()
    if wandb_run is not None:
        wandb_run.finish()


def run_training(
    cfg: TrainConfig,
    model_factory: ModelFactory | None = None,
) -> None:
    """Train one resolved canonical config through the notebook-facing API."""
    _run_training(cfg, model_factory=model_factory)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint with the source trainer's parse/validation contract."""
    cli_argv = sys.argv[1:] if argv is None else argv
    args, config_path, config_data = parse_configured_args(cli_argv)
    cfg, print_resolved_config = namespace_to_train_config(args)
    validate_train_config(cfg)
    if print_resolved_config:
        print(yaml.safe_dump({"resolved_args": asdict(cfg)}, sort_keys=False))
        return
    _run_training(
        cfg,
        _source_config=config_path,
        _source_data=config_data,
        _validated=True,
    )
