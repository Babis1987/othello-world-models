"""Configuration, split and runtime contracts for unified evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

PROTOCOL_ID = "unified_eval_v4"
V3_PROTOCOL_ID = "unified_eval_v3"
V2_PROTOCOL_ID = "unified_eval_v2"
SUPPORTED_PROTOCOL_IDS = frozenset(
    {V2_PROTOCOL_ID, V3_PROTOCOL_ID, PROTOCOL_ID}
)


@dataclass(frozen=True)
class UnifiedEvalConfig:
    """Every tunable choice in the common 8x8/12x12/16x16 evaluation."""

    board_size: int
    protocol_id: str = PROTOCOL_ID
    seed: int = 42

    # Clean split after the 200 pretraining shards.
    pretraining_shards: int = 200
    downstream_train_shards: int = 20
    selection_shards: int = 9
    test_shards: int = 9
    strict_total_shards: bool = True

    # Frozen next-move readouts.
    encoder_precision: str = "bf16"
    readout_precision: str = "fp32"
    head_batch_size: int = 512
    head_learning_rate: float = 1e-3
    head_weight_decay: float = 0.0
    head_gradient_clip: float = 1.0
    head_max_shards: int = 200
    head_selection_games: int = 2000
    head_eval_every_shards: int = 1
    head_patience: int = 5
    head_min_delta: float = 1e-3
    mlp_hidden_dim: int = 512
    mlp_dropout: float = 0.1

    # Frozen-readout hyperparameter selection (v4).  A small grouped-CV grid
    # search runs on cached encoder features before the streaming refit, so the
    # JEPA readouts are not compared at one arbitrary hyperparameter point.  The
    # search never sees the selection or test shards: it draws its own games
    # from the head-training pool and validates on held-out shards inside it.
    head_tuning_enabled: bool = True
    head_tuning_games: int = 5_000
    head_tuning_folds: int = 3
    head_tuning_epochs: int = 3
    head_tuning_token_batch: int = 8_192

    # Final legal-move test.
    legal_eval_games: int = 50_000
    legal_eval_batch_size: int = 64
    legal_k_values: tuple[int, ...] = (1, 2, 3, 5)

    # Resource-bounded Nanda-inspired probes.  Sixteen normalized positions
    # from 3,125 games give exactly 50,000 training positions independent of
    # board size, keeping the 16x16 suite within the Colab runtime budget.
    board_positions_per_game: int = 16
    board_train_games: int = 3125
    board_selection_games: int = 512
    board_test_games: int = 1000
    board_batch_size: int = 64
    board_probe_batch_size: int = 256
    board_epochs: int = 5
    board_patience: int = 2
    board_min_delta: float = 5e-4
    board_linear_lr: float = 1e-3
    board_mlp_lr: float = 3e-4
    board_weight_decay: float = 0.0
    board_gradient_clip: float = 1.0
    board_input_layernorm: bool = True

    # Optional 8x8 Nanda-style causal test.
    intervention_selection_cases: int = 200
    intervention_test_cases: int = 1000
    intervention_batch_size: int = 64
    intervention_alpha_values: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

    require_cuda_bf16: bool = True

    def __post_init__(self) -> None:
        if self.protocol_id not in SUPPORTED_PROTOCOL_IDS:
            raise ValueError(
                f"protocol_id must be one of {sorted(SUPPORTED_PROTOCOL_IDS)!r}, "
                f"got {self.protocol_id!r}"
            )
        if self.board_size < 4 or self.board_size % 2:
            raise ValueError("board_size must be an even integer >= 4")
        if self.encoder_precision != "bf16":
            raise ValueError("The headline unified protocol requires bf16 encoders")
        if self.readout_precision != "fp32":
            raise ValueError("The headline unified protocol requires fp32 readouts")
        positive = {
            "pretraining_shards": self.pretraining_shards,
            "downstream_train_shards": self.downstream_train_shards,
            "selection_shards": self.selection_shards,
            "test_shards": self.test_shards,
            "head_batch_size": self.head_batch_size,
            "head_max_shards": self.head_max_shards,
            "head_selection_games": self.head_selection_games,
            "head_eval_every_shards": self.head_eval_every_shards,
            "head_patience": self.head_patience,
            "legal_eval_games": self.legal_eval_games,
            "legal_eval_batch_size": self.legal_eval_batch_size,
            "board_positions_per_game": self.board_positions_per_game,
            "board_train_games": self.board_train_games,
            "board_selection_games": self.board_selection_games,
            "board_test_games": self.board_test_games,
            "board_batch_size": self.board_batch_size,
            "board_probe_batch_size": self.board_probe_batch_size,
            "board_epochs": self.board_epochs,
            "intervention_selection_cases": self.intervention_selection_cases,
            "intervention_test_cases": self.intervention_test_cases,
            "intervention_batch_size": self.intervention_batch_size,
        }
        if self.head_tuning_enabled:
            positive.update(
                {
                    "head_tuning_games": self.head_tuning_games,
                    "head_tuning_epochs": self.head_tuning_epochs,
                    "head_tuning_token_batch": self.head_tuning_token_batch,
                }
            )
            if self.head_tuning_folds < 2:
                raise ValueError("head_tuning_folds must be at least 2")
            if self.head_tuning_games < self.head_tuning_folds:
                raise ValueError(
                    "head_tuning_games must cover at least one game per fold"
                )
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Protocol counts must be positive: {invalid}")
        if self.head_min_delta < 0 or self.board_min_delta < 0:
            raise ValueError("Early-stopping min_delta values must be non-negative")
        if self.head_max_shards > (
            self.pretraining_shards + self.downstream_train_shards
        ):
            raise ValueError("head_max_shards exceeds the available head-training pool")

    @classmethod
    def v3(cls, board_size: int, *, seed: int = 42) -> "UnifiedEvalConfig":
        """Reproduce v3: identical to v4 minus the readout hyperparameter search."""
        return cls(
            board_size=board_size,
            protocol_id=V3_PROTOCOL_ID,
            seed=seed,
            head_tuning_enabled=False,
        )

    @classmethod
    def v2(cls, board_size: int, *, seed: int = 42) -> "UnifiedEvalConfig":
        """Reproduce the historical v2 model-selection protocol."""
        return cls(
            board_size=board_size,
            protocol_id=V2_PROTOCOL_ID,
            seed=seed,
            head_tuning_enabled=False,
            head_max_shards=20,
            # v2 trains exactly once on shards 200:220; selection is used for
            # layer/model selection, not to stop the one-epoch head training.
            head_eval_every_shards=20,
            head_patience=2,
            legal_eval_games=5_000,
            board_positions_per_game=4,
            board_train_games=5_000,
            board_selection_games=512,
            board_test_games=2_048,
        )


@dataclass(frozen=True)
class EvaluationSplit:
    """Disjoint pretraining, readout-training, selection and test shards."""

    pretraining: tuple[Path, ...]
    downstream_train: tuple[Path, ...]
    selection: tuple[Path, ...]
    test: tuple[Path, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "pretraining": [str(path) for path in self.pretraining],
            "downstream_train": [str(path) for path in self.downstream_train],
            "selection": [str(path) for path in self.selection],
            "test": [str(path) for path in self.test],
        }

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def head_training_pool(self, max_shards: int) -> tuple[Path, ...]:
        """Return held-out downstream shards first, then pretraining shards.

        This preserves the old clean downstream readout data as the first
        budget while allowing an unconverged head to continue on the SSL
        pretraining corpus without ever touching selection or test games.
        """
        if max_shards <= 0:
            raise ValueError("max_shards must be positive")
        pool = self.downstream_train + self.pretraining
        if max_shards > len(pool):
            raise ValueError(
                f"Requested {max_shards} head shards, only {len(pool)} available"
            )
        return pool[:max_shards]


def build_evaluation_split(
    chunks: Sequence[str | Path],
    config: UnifiedEvalConfig,
) -> EvaluationSplit:
    """Build and validate the exact common shard split."""
    ordered = tuple(sorted((Path(path) for path in chunks), key=lambda p: p.name))
    required = (
        config.pretraining_shards
        + config.downstream_train_shards
        + config.selection_shards
        + config.test_shards
    )
    if len(ordered) < required:
        raise ValueError(f"Need {required} shards, found {len(ordered)}")
    if config.strict_total_shards and len(ordered) != required:
        raise ValueError(
            f"{config.protocol_id} expects exactly {required} shards, found {len(ordered)}"
        )
    if len({str(path) for path in ordered}) != len(ordered):
        raise ValueError("Duplicate chunk paths are not allowed")

    p0 = config.pretraining_shards
    p1 = p0 + config.downstream_train_shards
    p2 = p1 + config.selection_shards
    p3 = p2 + config.test_shards
    split = EvaluationSplit(
        pretraining=ordered[:p0],
        downstream_train=ordered[p0:p1],
        selection=ordered[p1:p2],
        test=ordered[p2:p3],
    )
    groups = [
        set(map(str, split.pretraining)),
        set(map(str, split.downstream_train)),
        set(map(str, split.selection)),
        set(map(str, split.test)),
    ]
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            if left & right:
                raise AssertionError("Evaluation split groups overlap")
    return split


def validate_runtime(device: torch.device | str, config: UnifiedEvalConfig) -> None:
    """Refuse to label a fallback runtime as the headline protocol."""
    device = torch.device(device)
    if not config.require_cuda_bf16:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"{config.protocol_id} requires a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{config.protocol_id} requires CUDA bf16 support")
