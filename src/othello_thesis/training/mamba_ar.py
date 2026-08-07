"""Colab-friendly chunked training loop for Othello Mamba-AR.

This mirrors ``training/train_gpt.py`` so Transformer-AR and Mamba-AR can
be compared under the same data split, tokenization, optimizer schedule,
checkpoint style, and downstream evaluation utilities. Configuration is
intentionally CLI/notebook driven, matching the existing Transformer-AR
notebooks rather than the JEPA YAML workflow.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from othello_thesis.data.chunk_dataset import OthelloChunkDataset
from othello_thesis.models.mamba import MambaARConfig, OthelloMambaAR
from othello_thesis.training.ar_checkpoints import save_training_checkpoint
from othello_thesis.training.ar_data import (
    estimate_total_steps,
    make_chunk_loader,
    parse_milestones,
    resolve_chunks,
)
from othello_thesis.training.ar_runtime import (
    append_csv,
    autocast_context,
    cosine_learning_rate,
    load_model_state as _load_model_state,
    model_state_dict as _model_state_dict,
    sync_to_drive,
    validate_resume_precision,
)
from othello_thesis.training.ar_steps import (
    evaluate_chunks as _evaluate_ar_chunks,
    train_one_chunk as _train_ar_chunk,
)
from othello_thesis.training.mamba_ar_runner import run_mamba_ar



@dataclass
class MambaARTrainConfig:
    out_dir: str
    train_dir: str | None = None
    val_dir: str | None = None
    data_dir: str | None = None
    li_split_train_chunks: int = 200
    experiment_name: str = "mamba_ar"

    board_size: int = 8
    n_layers: int = 15
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    mlp_hidden_mult: int = 0
    mamba_backend: str = "mamba_ssm"

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

    drive_sync_dir: str | None = None
    drive_sync_every_chunks: int = 1

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"


def _autocast_context(device: torch.device, precision: str):
    return autocast_context(device, precision)


def make_loader(
    chunk_path: Path,
    cfg: MambaARTrainConfig,
    model_cfg: MambaARConfig,
    shuffle: bool,
) -> tuple[DataLoader, list[list[int]], OthelloChunkDataset]:
    return make_chunk_loader(
        chunk_path,
        block_size=model_cfg.block_size,
        board_size=cfg.board_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
    )


def format_chunk_progress(
    chunk_count: int,
    total_chunks: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    games_seen: int,
    dt: float,
    chunk_name: str,
) -> str:
    width = 30
    fraction = min(1.0, chunk_count / max(1, total_chunks))
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * fraction
    val_text = "" if math.isnan(val_loss) else (
        f" | val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
    )
    return (
        f"mamba-ar [{bar}] {chunk_count}/{total_chunks} ({percent:5.1f}%) "
        f"{chunk_name} | games={games_seen:,} "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.2%}"
        f"{val_text} | dt={dt:.0f}s"
    )


def get_lr(step: int, cfg: MambaARTrainConfig, total_steps: int) -> float:
    return cosine_learning_rate(step, cfg, total_steps)


def train_one_chunk(
    model: OthelloMambaAR,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    chunk_path: Path,
    cfg: MambaARTrainConfig,
    model_cfg: MambaARConfig,
    device: torch.device,
    step: int,
    total_steps: int,
    progress_desc: str | None = None,
) -> tuple[float, float, int, int, int]:
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
        style="mamba",
        progress_desc=progress_desc,
    )


@torch.no_grad()
def eval_chunks(
    model: OthelloMambaAR,
    chunks: list[Path],
    cfg: MambaARTrainConfig,
    model_cfg: MambaARConfig,
    device: torch.device,
) -> tuple[float, float, int, int]:
    return _evaluate_ar_chunks(
        model,
        chunks,
        cfg,
        model_cfg,
        device,
        style="mamba",
    )


def model_state_dict(model: torch.nn.Module) -> dict:
    return _model_state_dict(model)


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    _load_model_state(model, state)


def save_checkpoint(
    path: Path,
    model: OthelloMambaAR,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: MambaARTrainConfig,
    model_cfg: MambaARConfig,
    step: int,
    pass_num: int,
    chunk_count: int,
    next_chunk_idx: int,
    games_seen: int,
    tokens_seen: int,
) -> None:
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
        format_id="mamba_ar_checkpoint_v1",
    )


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> tuple[OthelloMambaAR, MambaARConfig]:
    config_data = dict(ckpt.get("model_config") or {})
    if not config_data:
        train_cfg = dict(ckpt.get("train_config") or {})
        config_data = {
            key: train_cfg[key]
            for key in (
                "board_size",
                "n_layers",
                "d_model",
                "d_state",
                "d_conv",
                "expand",
                "dropout",
                "mlp_hidden_mult",
                "mamba_backend",
            )
            if key in train_cfg
        }
    model_cfg = MambaARConfig(**{
        key: value
        for key, value in config_data.items()
        if key in MambaARConfig.__dataclass_fields__
    })
    model = OthelloMambaAR(model_cfg)
    model.load_state_dict(ckpt["model"])
    return model, model_cfg


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[OthelloMambaAR, MambaARConfig, dict[str, Any]]:
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, model_cfg = build_model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()
    return model, model_cfg, ckpt


def init_wandb(cfg: MambaARTrainConfig, model_cfg: MambaARConfig):
    if not cfg.wandb_project or cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb_project was set, but wandb is not installed; continuing without WandB.", flush=True)
        return None
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name or cfg.experiment_name,
        mode=cfg.wandb_mode,
        config={"train": asdict(cfg), "model": asdict(model_cfg)},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--li_split_train_chunks", type=int, default=200)
    parser.add_argument("--experiment_name", type=str, default="mamba_ar")

    parser.add_argument("--board_size", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_hidden_mult", type=int, default=0)
    parser.add_argument("--mamba_backend", type=str, default="mamba_ssm",
                        choices=["mamba_ssm", "official", "auto", "torch", "fallback", "mamba_py"])

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--passes_over_data", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=0)
    parser.add_argument("--games_per_chunk", type=int, default=100_000)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_chunks", type=int, default=1)
    parser.add_argument("--eval_every_chunks", type=int, default=5)
    parser.add_argument("--checkpoint_every_chunks", type=int, default=0)
    parser.add_argument("--milestone_games", type=str, default="1000000,5000000,10000000,15000000,20000000")
    parser.add_argument("--max_chunks", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile_model", action="store_true")

    parser.add_argument("--drive_sync_dir", type=str, default=None)
    parser.add_argument("--drive_sync_every_chunks", type=int, default=1)

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    return parser


def parse_args(argv: list[str] | None = None) -> MambaARTrainConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return MambaARTrainConfig(**vars(args))


def run_training(
    cfg: MambaARTrainConfig,
    *,
    model_factory: Callable[[MambaARConfig], torch.nn.Module] | None = None,
) -> None:
    """Train one canonical run, constructing the supplied model exactly once."""
    run_mamba_ar(
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
        format_chunk_progress=format_chunk_progress,
    )

def main(argv: list[str] | None = None) -> None:
    """Parse the historical CLI and run the canonical training loop."""
    print("[mamba-ar] parsing args", flush=True)
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
