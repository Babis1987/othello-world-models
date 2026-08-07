"""Periodic board-state probing callback for v7 training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from othello_research.probes.board_state import train_board_state_probes


def _parse_layers(value: str) -> tuple[int, ...] | None:
    value = value.strip()
    if not value:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _flatten_probe_accuracies(result: dict[str, object]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return flattened
    for layer, layer_metrics in metrics.items():
        if not isinstance(layer_metrics, dict):
            continue
        for mode, mode_metrics in layer_metrics.items():
            if not isinstance(mode_metrics, dict):
                continue
            accuracy = mode_metrics.get("accuracy")
            if accuracy is not None:
                flattened[f"probe/layer_{layer}/{mode}_accuracy"] = float(accuracy)
    return flattened


class PeriodicActionConditionedProbe:
    """Run fresh linear probes on a deterministic fixed subset every N steps."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_chunks: Sequence[Path],
        val_chunks: Sequence[Path],
        cfg,
        device: torch.device,
        out_dir: Path,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.train_chunks = tuple(train_chunks)
        self.val_chunks = tuple(val_chunks)
        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir / "periodic_probes"
        self.wandb_run = wandb_run
        self.last_step = -1

    def __call__(self, step: int, _chunk_samples_seen: int) -> None:
        interval = int(self.cfg.probe_interval_steps)
        if interval <= 0 or step == self.last_step or step % interval != 0:
            return
        if not self.train_chunks or not self.val_chunks:
            return
        self.last_step = step

        raw_model = (
            self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        )
        encoder = raw_model.context_encoder
        was_training = encoder.training
        requires_grad = [parameter.requires_grad for parameter in encoder.parameters()]
        try:
            _, result = train_board_state_probes(
                model=encoder,
                train_chunks=self.train_chunks,
                val_chunks=self.val_chunks,
                board_size=self.cfg.board_size,
                n_train_games=self.cfg.probe_train_games,
                n_val_games=self.cfg.probe_val_games,
                device=self.device,
                layers=_parse_layers(self.cfg.probe_layers),
                batch_size=self.cfg.probe_batch_size,
                epochs=self.cfg.probe_epochs,
                lr=self.cfg.probe_learning_rate,
            )
        finally:
            for parameter, original in zip(encoder.parameters(), requires_grad):
                parameter.requires_grad_(original)
            encoder.train(was_training)
            raw_model.target_encoder.eval()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.out_dir / f"probe_step_{step:08d}.json"
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        metrics = _flatten_probe_accuracies(result)
        if self.wandb_run is not None and metrics:
            self.wandb_run.log(metrics, step=step)
        print(f"  periodic board probe step={step}: {metrics}", flush=True)
