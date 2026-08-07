"""Cohesive orchestration phases for one canonical Mamba-AR run."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from othello_thesis.models.mamba import MambaARConfig, OthelloMambaAR


class MambaARRunner:
    """Stateful orchestration preserving Mamba-AR's diagnostic stdout."""

    def __init__(
        self,
        cfg: Any,
        *,
        model_factory: Callable[[MambaARConfig], torch.nn.Module] | None,
        resolve_chunks: Callable,
        estimate_total_steps: Callable,
        parse_milestones: Callable,
        validate_resume_precision: Callable,
        load_model_state: Callable,
        train_one_chunk: Callable,
        eval_chunks: Callable,
        save_checkpoint: Callable,
        append_csv: Callable,
        sync_to_drive: Callable,
        init_wandb: Callable,
        format_chunk_progress: Callable,
    ) -> None:
        self.cfg = cfg
        self.model_factory = model_factory
        self.resolve_chunks = resolve_chunks
        self.estimate_total_steps = estimate_total_steps
        self.parse_milestones = parse_milestones
        self.validate_resume_precision = validate_resume_precision
        self.load_model_state = load_model_state
        self.train_one_chunk = train_one_chunk
        self.eval_chunks = eval_chunks
        self.save_checkpoint = save_checkpoint
        self.append_csv = append_csv
        self.sync_to_drive = sync_to_drive
        self.init_wandb = init_wandb
        self.format_chunk_progress = format_chunk_progress

    def run(self) -> None:
        self._validate_and_seed()
        self._prepare_paths_and_split()
        self._build_model_and_optimizer()
        self._initialize_and_restore_state()
        self._prepare_schedule_and_logging()
        if self._train_all_chunks():
            return
        self._finish_normal_run()

    def _validate_and_seed(self) -> None:
        cfg = self.cfg
        print(
            f"[mamba-ar] args parsed: experiment={cfg.experiment_name} "
            f"backend={cfg.mamba_backend} layers={cfg.n_layers} "
            f"d_model={cfg.d_model}",
            flush=True,
        )
        if (
            cfg.precision == "bf16"
            and torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "precision=bf16 was requested but this CUDA device does not support bf16"
            )
        if (
            cfg.mamba_backend in {"mamba_ssm", "official", "auto"}
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "Mamba-AR with mamba_ssm requires a CUDA GPU runtime. "
                "Use GPU for smoke/full runs, or explicitly pass "
                "--mamba_backend torch only for local debugging tests."
            )
        print("[mamba-ar] seeding and configuring torch", flush=True)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _prepare_paths_and_split(self) -> None:
        cfg = self.cfg
        print("[mamba-ar] preparing output directory", flush=True)
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "train_config.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(cfg), handle, indent=2)
        self.drive_sync_dir = (
            Path(cfg.drive_sync_dir) if cfg.drive_sync_dir else None
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[mamba-ar] device resolved: {self.device}", flush=True)
        print("[mamba-ar] resolving chunk split", flush=True)
        self.train_chunks, self.val_chunks = self.resolve_chunks(cfg)
        print(
            f"[mamba-ar] chunks resolved: train={len(self.train_chunks)} "
            f"val={len(self.val_chunks)}",
            flush=True,
        )

    def _build_model_and_optimizer(self) -> None:
        cfg = self.cfg
        print("[mamba-ar] building model config", flush=True)
        self.model_cfg = MambaARConfig(
            board_size=cfg.board_size,
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dropout=cfg.dropout,
            mlp_hidden_mult=cfg.mlp_hidden_mult,
            mamba_backend=cfg.mamba_backend,
        )
        print("[mamba-ar] constructing OthelloMambaAR", flush=True)
        started = time.time()
        construct_model = self.model_factory or OthelloMambaAR
        self.model = construct_model(self.model_cfg)
        print(
            f"[mamba-ar] model constructed on CPU in "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
        print("[mamba-ar] moving model to device", flush=True)
        started = time.time()
        self.model = self.model.to(self.device)
        print(
            f"[mamba-ar] model moved to {self.device} in "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
        if cfg.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        print("[mamba-ar] creating optimizer/scaler", flush=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type == "cuda" and cfg.precision == "fp16")
        )

    def _initialize_and_restore_state(self) -> None:
        cfg = self.cfg
        self.step = 0
        self.start_pass = 1
        self.start_chunk_idx = 0
        self.chunk_count = 0
        self.games_seen = 0
        self.tokens_seen = 0
        self.completed_milestones: set[int] = set()
        if not cfg.resume:
            return

        checkpoint = torch.load(
            cfg.resume,
            map_location=self.device,
            weights_only=False,
        )
        self.validate_resume_precision(checkpoint, cfg.precision)
        self.load_model_state(self.model, checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = checkpoint.get("step", 0)
        self.start_pass = checkpoint.get("pass_num", 1)
        self.start_chunk_idx = checkpoint.get("next_chunk_idx", 0)
        self.chunk_count = checkpoint.get("chunk_count", 0)
        self.games_seen = checkpoint.get("games_seen", 0)
        self.tokens_seen = checkpoint.get("tokens_seen", 0)
        self.completed_milestones = {
            milestone
            for milestone in self.parse_milestones(cfg.milestone_games)
            if milestone <= self.games_seen
        }
        if self.start_chunk_idx >= len(self.train_chunks):
            self.start_pass += 1
            self.start_chunk_idx = 0
        print(
            f"resumed from {cfg.resume}: step={self.step} "
            f"games_seen={self.games_seen:,} pass={self.start_pass} "
            f"chunk_idx={self.start_chunk_idx}",
            flush=True,
        )

    def _prepare_schedule_and_logging(self) -> None:
        cfg = self.cfg
        print("[mamba-ar] estimating total steps", flush=True)
        self.total_steps, _ = self.estimate_total_steps(self.train_chunks, cfg)
        print("[mamba-ar] parsing milestones and initializing logging", flush=True)
        self.milestones = self.parse_milestones(cfg.milestone_games)
        self.wandb_run = self.init_wandb(cfg, self.model_cfg)
        self.metrics_path = self.out_dir / "metrics.csv"
        print(f"experiment: {cfg.experiment_name}", flush=True)
        print(f"device: {self.device}", flush=True)
        print(f"train chunks: {len(self.train_chunks)}", flush=True)
        print(f"val chunks: {len(self.val_chunks)}", flush=True)
        print(
            f"vocab_size: {self.model_cfg.vocab_size}, "
            f"block_size: {self.model_cfg.block_size}",
            flush=True,
        )
        print(f"mamba_backend: {self.model_cfg.mamba_backend}", flush=True)
        print(f"parameters: {self.model.num_parameters():,}", flush=True)
        if self.drive_sync_dir is not None:
            print(
                f"drive auto-sync: every {cfg.drive_sync_every_chunks} "
                f"chunk(s) -> {self.drive_sync_dir}",
                flush=True,
            )

    def _train_all_chunks(self) -> bool:
        cfg = self.cfg
        self.total_chunks_to_train = (
            cfg.max_chunks
            if cfg.max_chunks
            else len(self.train_chunks) * cfg.passes_over_data
        )
        print(
            f"mamba-ar chunk progress: 0/{self.total_chunks_to_train} chunks",
            flush=True,
        )
        for pass_num in range(self.start_pass, cfg.passes_over_data + 1):
            chunk_order = self.train_chunks[:]
            random.Random(cfg.seed + pass_num).shuffle(chunk_order)
            first_idx = self.start_chunk_idx if pass_num == self.start_pass else 0
            for chunk_index in range(first_idx, len(chunk_order)):
                if self._run_one_chunk(
                    pass_num,
                    chunk_index,
                    chunk_order[chunk_index],
                ):
                    return True
        return False

    def _run_one_chunk(
        self,
        pass_num: int,
        chunk_index: int,
        chunk_path: Path,
    ) -> bool:
        cfg = self.cfg
        started = time.time()
        train_loss, train_acc, self.step, chunk_games, chunk_tokens = (
            self.train_one_chunk(
                self.model,
                self.optimizer,
                self.scaler,
                chunk_path,
                cfg,
                self.model_cfg,
                self.device,
                self.step,
                self.total_steps,
                progress_desc=(
                    f"chunk {self.chunk_count + 1}/{self.total_chunks_to_train} "
                    f"{chunk_path.name}"
                ),
            )
        )
        self.chunk_count += 1
        self.games_seen += chunk_games
        self.tokens_seen += chunk_tokens
        val_loss, val_acc, val_games, val_tokens = self._maybe_evaluate()
        learning_rate = self.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - started
        row = self._metrics_row(
            pass_num=pass_num,
            chunk_index=chunk_index,
            chunk_path=chunk_path,
            chunk_games=chunk_games,
            chunk_tokens=chunk_tokens,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_games=val_games,
            val_tokens=val_tokens,
            learning_rate=learning_rate,
            elapsed=elapsed,
        )
        self.append_csv(self.metrics_path, row)
        self._log_wandb(row)
        print(
            self.format_chunk_progress(
                chunk_count=self.chunk_count,
                total_chunks=self.total_chunks_to_train,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                games_seen=self.games_seen,
                dt=elapsed,
                chunk_name=chunk_path.name,
            ),
            flush=True,
        )
        self._save_progress_checkpoints(pass_num, chunk_index)
        self._sync_if_due()
        if cfg.max_chunks and self.chunk_count >= cfg.max_chunks:
            self._finish_early(pass_num, chunk_index)
            return True
        return False

    def _maybe_evaluate(self) -> tuple[float, float, int, int]:
        cfg = self.cfg
        should_eval = (
            bool(self.val_chunks)
            and cfg.eval_chunks > 0
            and self.chunk_count % max(1, cfg.eval_every_chunks) == 0
        )
        if should_eval:
            return self.eval_chunks(
                self.model,
                self.val_chunks,
                cfg,
                self.model_cfg,
                self.device,
            )
        return float("nan"), float("nan"), 0, 0

    def _metrics_row(
        self,
        *,
        pass_num: int,
        chunk_index: int,
        chunk_path: Path,
        chunk_games: int,
        chunk_tokens: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_games: int,
        val_tokens: int,
        learning_rate: float,
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": self.cfg.experiment_name,
            "pass": pass_num,
            "chunk_count": self.chunk_count,
            "chunk_index_in_pass": chunk_index,
            "step": self.step,
            "games_seen": self.games_seen,
            "tokens_seen": self.tokens_seen,
            "chunk_games": chunk_games,
            "chunk_tokens": chunk_tokens,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_games": val_games,
            "val_tokens": val_tokens,
            "lr": learning_rate,
            "dt_seconds": elapsed,
            "chunk_file": chunk_path.name,
        }

    def _log_wandb(self, row: dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(
            {
                key: value
                for key, value in row.items()
                if key not in {"time", "chunk_file", "experiment_name"}
                and not (isinstance(value, float) and math.isnan(value))
            },
            step=self.step,
        )

    def _save_progress_checkpoints(self, pass_num: int, chunk_index: int) -> None:
        cfg = self.cfg
        self._save(self.out_dir / "latest.pt", pass_num, chunk_index + 1)
        for milestone in self.milestones:
            if (
                self.games_seen >= milestone
                and milestone not in self.completed_milestones
            ):
                self.completed_milestones.add(milestone)
                self._save(
                    self.out_dir
                    / f"checkpoint_games_{milestone // 1_000_000:03d}M.pt",
                    pass_num,
                    chunk_index + 1,
                )
        if (
            cfg.checkpoint_every_chunks
            and self.chunk_count % cfg.checkpoint_every_chunks == 0
        ):
            self._save(
                self.out_dir / f"checkpoint_chunk_{self.chunk_count:06d}.pt",
                pass_num,
                chunk_index + 1,
            )

    def _save(self, path: Path, pass_num: int, next_chunk_idx: int) -> None:
        self.save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scaler,
            self.cfg,
            self.model_cfg,
            self.step,
            pass_num,
            self.chunk_count,
            next_chunk_idx,
            self.games_seen,
            self.tokens_seen,
        )

    def _sync_if_due(self) -> None:
        cfg = self.cfg
        if self.drive_sync_dir is None or (
            self.chunk_count % max(1, cfg.drive_sync_every_chunks) != 0
        ):
            return
        n_synced, elapsed = self.sync_to_drive(
            self.out_dir, self.drive_sync_dir
        )
        if n_synced > 0:
            print(f"  drive sync: {n_synced} file(s) in {elapsed:.1f}s", flush=True)

    def _finish_early(self, pass_num: int, chunk_index: int) -> None:
        self._save(self.out_dir / "final.pt", pass_num, chunk_index + 1)
        self._final_sync()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _finish_normal_run(self) -> None:
        self._save(
            self.out_dir / "final.pt",
            self.cfg.passes_over_data,
            len(self.train_chunks),
        )
        self._final_sync()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _final_sync(self) -> None:
        if self.drive_sync_dir is None:
            return
        n_synced, elapsed = self.sync_to_drive(
            self.out_dir, self.drive_sync_dir
        )
        print(
            f"  final drive sync: {n_synced} file(s) in {elapsed:.1f}s",
            flush=True,
        )


def run_mamba_ar(cfg: Any, **kwargs: Any) -> None:
    """Construct the runner without adding any stochastic operation."""
    MambaARRunner(cfg, **kwargs).run()
