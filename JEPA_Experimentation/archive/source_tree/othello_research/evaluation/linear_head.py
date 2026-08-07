"""Linear next-move head evaluation utilities for frozen JEPA context encoders."""

from __future__ import annotations

import csv
import gc
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from othello_research.datasets.dataset import (
    TARGET_PAD,
    OthelloChunkDataset,
    load_chunk,
)
from othello_research.models.gpt import GPTConfig
from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is available on Colab by default
    def tqdm(x=None, **_):
        return x if x is not None else _NoOpBar()


class _NoOpBar:
    """Minimal stand-in for tqdm when it is not installed."""

    def update(self, _n: int = 1) -> None: ...
    def set_postfix(self, **_: object) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *_: object) -> None: ...


# ============================================================================
# Frozen-encoder linear head model
# ============================================================================

class JEPALinearHead(nn.Module):
    """Frozen JEPA context encoder plus trainable next-token linear head.

    Only the JEPA context encoder is used for next-move prediction. The JEPA
    predictor and EMA target encoder are ignored. The context encoder is frozen
    and kept in eval mode; only ``self.head`` receives gradients.
    """

    def __init__(self, jepa_model: OthelloJEPA) -> None:
        super().__init__()
        self.encoder = jepa_model.context_encoder
        self.encoder_config = jepa_model.config
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

        d_model = self.encoder_config.d_model
        vocab_size = self.encoder_config.vocab_size
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def config(self) -> GPTConfig:
        """Expose the underlying GPT config for existing evaluators."""
        return self.encoder_config

    def train(self, mode: bool = True) -> "JEPALinearHead":
        """Set the head train/eval mode while keeping the encoder frozen."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, loss)`` with the same signature as ``OthelloGPT``.

        Args:
            idx: Token ids with shape ``(B, T)``.
            targets: Optional next-token labels with ``TARGET_PAD`` ignored.
        """
        if idx.ndim != 2:
            raise ValueError(f"Expected idx with shape (B, T), got {tuple(idx.shape)}")
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input length {seq_len} exceeds block_size {self.config.block_size}."
            )

        with torch.no_grad():
            enc = self.encoder
            pos = torch.arange(0, seq_len, device=idx.device)
            x = enc.wte(idx) + enc.wpe(pos)
            x = enc.drop(x)
            for block in enc.blocks:
                x = block(x)
            x = enc.ln_f(x)

        logits = self.head(x)
        if targets is None:
            return logits, None

        flat_targets = targets.reshape(-1)
        valid_count = (flat_targets != TARGET_PAD).sum()
        if int(valid_count.item()) == 0:
            # PyTorch cross_entropy returns NaN when every target is ignored.
            # Return a connected zero loss so callers can safely backward()
            # without updating the head on all-pad batches.
            return logits, logits.sum() * 0.0

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            flat_targets,
            ignore_index=TARGET_PAD,
            reduction="sum",
        ) / valid_count.clamp_min(1)
        return logits, loss


# ============================================================================
# Training the linear head
# ============================================================================

@dataclass
class LinearHeadTrainConfig:
    """Hyperparameters for training the linear head on a frozen encoder."""

    train_chunks: list[str | Path]
    out_dir: str
    board_size: int = 8
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    n_chunks: int = 0
    eval_every_chunks: int = 0
    precision: str = "bf16"


def _amp_context(device: torch.device, precision: str = "bf16"):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    return nullcontext()


def _make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _config_jsonable(cfg: LinearHeadTrainConfig) -> dict[str, object]:
    data = asdict(cfg)
    data["train_chunks"] = [str(path) for path in cfg.train_chunks]
    return data


def _objective_config_jsonable(jepa_model: nn.Module) -> dict[str, object]:
    """Serialize objective metadata across predictive, contrastive, and v6 JEPA."""
    objective_cfg = getattr(jepa_model, "jepa_config", None)
    if objective_cfg is None:
        objective_cfg = getattr(jepa_model, "cfg", None)
    return asdict(objective_cfg) if is_dataclass(objective_cfg) else {}


def _append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    sanitized = {
        key: ("" if isinstance(value, float) and math.isnan(value) else value)
        for key, value in row.items()
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(sanitized)


def _make_loader(
    chunk_path: str | Path,
    model: JEPALinearHead,
    cfg: LinearHeadTrainConfig,
    *,
    shuffle: bool = True,
) -> tuple[DataLoader, list[list[int]], OthelloChunkDataset]:
    games = load_chunk(str(chunk_path))
    dataset = OthelloChunkDataset(
        games=games,
        block_size=model.config.block_size,
        board_size=cfg.board_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, games, dataset


def train_linear_head(
    jepa_model: OthelloJEPA,
    cfg: LinearHeadTrainConfig,
    device: torch.device,
) -> JEPALinearHead:
    """Train a fresh linear next-token head on a frozen JEPA encoder.

    Every valid target position in the chunk dataset contributes to the
    cross-entropy loss; target PAD positions are ignored with ``TARGET_PAD``.
    The function saves ``head.pt``, ``linear_head.pt``, ``train_config.json``,
    and ``metrics.csv`` to ``cfg.out_dir`` and returns the trained wrapper.
    """
    device = torch.device(device)
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {cfg.batch_size}")
    if cfg.board_size != jepa_model.config.board_size:
        raise ValueError(
            f"cfg.board_size={cfg.board_size} does not match "
            f"JEPA board_size={jepa_model.config.board_size}"
        )

    selected_chunks = list(cfg.train_chunks)
    if cfg.n_chunks > 0:
        selected_chunks = selected_chunks[: cfg.n_chunks]
    if not selected_chunks:
        raise ValueError("No training chunks were provided")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(
        json.dumps(_config_jsonable(cfg), indent=2),
        encoding="utf-8",
    )

    model = JEPALinearHead(jepa_model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scaler = _make_grad_scaler(device, cfg.precision)
    metrics_path = out_dir / "metrics.csv"

    step = 0
    games_seen = 0
    tokens_seen = 0
    outer = tqdm(selected_chunks, desc="linear head chunks", unit="chunk")

    for chunk_idx, chunk_path in enumerate(outer, start=1):
        t0 = time.time()
        loader, games, dataset = _make_loader(chunk_path, model, cfg, shuffle=True)
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        chunk_games = len(games)

        pbar = tqdm(loader, desc=f"head {Path(chunk_path).name}", leave=False)
        for x_cpu, y_cpu in pbar:
            x = x_cpu.to(device, non_blocking=device.type == "cuda")
            y = y_cpu.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, cfg.precision):
                logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Linear-head training expected a loss")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                mask = y != TARGET_PAD
                n_tokens = int(mask.sum().item())
                preds = logits.argmax(dim=-1)
                correct = int(((preds == y) & mask).sum().item())
                total_loss += float(loss.detach().cpu().item()) * n_tokens
                total_correct += correct
                total_tokens += n_tokens

            step += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{float(loss.detach().cpu().item()):.4f}",
                    acc=f"{correct / max(1, n_tokens):.2%}",
                    lr=f"{cfg.learning_rate:.2e}",
                )

        del loader, dataset, games
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        games_seen += chunk_games
        tokens_seen += total_tokens
        avg_loss = total_loss / max(1, total_tokens)
        avg_acc = total_correct / max(1, total_tokens)
        dt = time.time() - t0
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunk": chunk_idx,
            "step": step,
            "games_seen": games_seen,
            "tokens_seen": tokens_seen,
            "chunk_games": chunk_games,
            "chunk_tokens": total_tokens,
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "lr": cfg.learning_rate,
            "dt_seconds": dt,
            "chunk_file": Path(chunk_path).name,
        }
        _append_csv(metrics_path, row)
        print(
            f"chunk={chunk_idx}/{len(selected_chunks)} step={step} "
            f"games={games_seen:,} tokens={tokens_seen:,} "
            f"train_loss={avg_loss:.4f} train_acc={avg_acc:.2%} "
            f"lr={cfg.learning_rate:.2e} dt={dt:.0f}s",
            flush=True,
        )
        if hasattr(outer, "set_postfix"):
            outer.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2%}")

    head_state = model.head.state_dict()
    torch.save(head_state, out_dir / "head.pt")
    torch.save(
        {
            "head": head_state,
            "train_config": _config_jsonable(cfg),
            "jepa_config": _objective_config_jsonable(jepa_model),
            "model_config": asdict(jepa_model.config),
            "format": "jepa_linear_next_move_head_v1",
        },
        out_dir / "linear_head.pt",
    )
    return model


def load_linear_head(
    jepa_model: OthelloJEPA,
    head_path: str | Path,
    device: torch.device | str,
) -> JEPALinearHead:
    """Load a saved linear head on top of a frozen JEPA context encoder."""
    device = torch.device(device)
    model = JEPALinearHead(jepa_model).to(device)
    payload = torch.load(head_path, map_location=device)
    state = payload["head"] if isinstance(payload, dict) and "head" in payload else payload
    model.head.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_next_move_head(
    model: JEPALinearHead,
    chunks: Sequence[str | Path],
    board_size: int,
    device: torch.device | str,
    *,
    n_chunks: int = 0,
    batch_size: int = 512,
    precision: str = "bf16",
) -> dict[str, float | int]:
    """Evaluate validation loss and exact next-token top-k accuracy."""
    device = torch.device(device)
    selected_chunks = list(chunks)
    if n_chunks > 0:
        selected_chunks = selected_chunks[:n_chunks]
    if not selected_chunks:
        raise ValueError("No chunks were provided for next-move evaluation")

    if board_size != model.config.board_size:
        raise ValueError(
            f"board_size={board_size} does not match model board_size={model.config.board_size}"
        )

    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_tokens = 0
    total_games = 0

    eval_cfg = LinearHeadTrainConfig(
        train_chunks=[],
        out_dir=".",
        board_size=board_size,
        batch_size=batch_size,
        precision=precision,
    )
    for chunk_path in tqdm(selected_chunks, desc="next-move eval", unit="chunk"):
        loader, games, dataset = _make_loader(chunk_path, model, eval_cfg, shuffle=False)
        total_games += len(games)
        for x_cpu, y_cpu in tqdm(loader, desc=f"eval {Path(chunk_path).name}", leave=False):
            x = x_cpu.to(device, non_blocking=device.type == "cuda")
            y = y_cpu.to(device, non_blocking=device.type == "cuda")
            with _amp_context(device, precision):
                logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Next-move evaluation expected a loss")

            mask = y != TARGET_PAD
            n_tokens = int(mask.sum().item())
            preds = logits.argmax(dim=-1)
            valid_logits = logits[mask]
            valid_targets = y[mask]
            actual_k3 = min(3, logits.size(-1))
            actual_k5 = min(5, logits.size(-1))
            top3 = torch.topk(valid_logits, k=actual_k3, dim=-1).indices
            top5 = torch.topk(valid_logits, k=actual_k5, dim=-1).indices
            total_loss += float(loss.detach().cpu().item()) * n_tokens
            total_top1 += int(((preds == y) & mask).sum().item())
            total_top3 += int((top3 == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item())
            total_top5 += int((top5 == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item())
            total_tokens += n_tokens

        del loader, dataset, games
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "n_chunks": len(selected_chunks),
        "n_games": total_games,
        "n_tokens": total_tokens,
        "loss": total_loss / max(1, total_tokens),
        "top1_accuracy": total_top1 / max(1, total_tokens),
        "top3_accuracy": total_top3 / max(1, total_tokens),
        "top5_accuracy": total_top5 / max(1, total_tokens),
        "accuracy": total_top1 / max(1, total_tokens),
    }


def evaluate_linear_head_legal_moves(
    model: JEPALinearHead,
    val_chunks: Sequence[str | Path],
    board_size: int,
    n_games: int,
    device: torch.device | str,
    k_values: Sequence[int] = (1, 2, 3, 5),
    batch_size: int = 64,
) -> dict[str, object]:
    """Evaluate a trained JEPA linear head with the AR legal-move metrics."""
    from othello_research.evaluation.legal_moves import evaluate_legal_moves

    return evaluate_legal_moves(
        model=model,
        val_chunks=val_chunks,
        board_size=board_size,
        n_games=n_games,
        device=device,
        k_values=k_values,
        batch_size=batch_size,
    )


def _smoke_test() -> None:
    torch.manual_seed(42)
    cfg = JEPAConfig(board_size=8, n_layers=2, n_heads=4, d_model=64, dropout=0.0)
    jepa = OthelloJEPA(cfg)
    model = JEPALinearHead(jepa)

    batch_size = 4
    seq_len = min(8, model.config.block_size)
    idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    targets[0, -1] = TARGET_PAD

    logits, loss = model(idx, targets)
    assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
    assert loss is not None and torch.isfinite(loss).all()

    loss.backward()
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    assert not encoder_grads, "Frozen JEPA encoder received gradients"
    assert len(head_grads) == 2, "Linear head did not receive gradients"
    assert all(torch.isfinite(grad).all() for grad in head_grads)
    print("OK")


if __name__ == "__main__":
    _smoke_test()
