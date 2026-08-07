"""Shared thesis evaluation protocol for Othello representation models.

The authoritative 8x8 and 12x12 notebooks intentionally contain only model
loading code.  Everything that can affect a reported metric lives here so AR
and JEPA, Transformer and Mamba cannot silently drift onto different splits,
precisions, readout budgets, or seeds.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from othello_thesis.data.chunk_dataset import (
    TARGET_PAD,
    OthelloChunkDataset,
    load_chunk,
)
from othello_thesis.data.move_vocabulary import build_mappings
from othello_thesis.evaluation.legal_moves import (
    _pad_token_batch,
    _tokenize_game,
    evaluate_legal_moves,
    replay_game_legality,
)
from othello_thesis.evaluation.position_probe import (
    PositionFeatureSet,
    PositionProbeBank,
    PositionProbeSplits,
    build_position_probe_splits,
    extract_position_features,
    train_position_probe_bank,
)


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoder_amp_context(device: torch.device, precision: str = "bf16"):
    if device.type == "cuda":
        if precision != "bf16":
            raise ValueError("Unified encoder precision must be bf16")
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def _base_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def validate_encoder_surface(encoder: nn.Module, board_size: int) -> None:
    encoder = _base_model(encoder)
    missing = [
        name
        for name in ("wte", "wpe", "drop", "blocks", "ln_f", "config")
        if not hasattr(encoder, name)
    ]
    if missing:
        raise TypeError(f"Encoder is missing shared surface fields: {missing}")
    model_board_size = int(getattr(encoder.config, "board_size", -1))
    if model_board_size != board_size:
        raise ValueError(
            f"Encoder board_size={model_board_size}, protocol board_size={board_size}"
        )
    expected_block_size = board_size * board_size - 5
    model_block_size = int(getattr(encoder.config, "block_size", -1))
    if model_block_size != expected_block_size:
        raise ValueError(
            f"Encoder block_size={model_block_size}, expected {expected_block_size} "
            f"for board_size={board_size}"
        )
    expected_vocab_size = board_size * board_size - 3
    model_vocab_size = int(getattr(encoder.config, "vocab_size", -1))
    if model_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Encoder vocab_size={model_vocab_size}, expected {expected_vocab_size} "
            f"for board_size={board_size}"
        )


def assert_finite_module(module: nn.Module, label: str = "model") -> None:
    """Scan a loaded checkpoint before any metric can hide corruption."""
    for name, tensor in module.state_dict().items():
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            count = int((~torch.isfinite(tensor)).sum().item())
            raise RuntimeError(f"{label}.{name} contains {count} non-finite values")


class _MLPReadout(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_readout(
    d_model: int,
    vocab_size: int,
    head_type: str,
    *,
    hidden_dim: int = 512,
    dropout: float = 0.1,
) -> nn.Module:
    """Build the fp32 readout used both by the tuning search and the refit."""
    if head_type == "linear":
        return nn.Linear(d_model, vocab_size)
    if head_type == "mlp":
        return _MLPReadout(d_model, vocab_size, hidden_dim, dropout)
    raise ValueError("head_type must be 'linear' or 'mlp'")


class FrozenEncoderNextMoveHead(nn.Module):
    """Architecture-neutral frozen encoder plus a supervised fp32 readout."""

    def __init__(
        self,
        encoder: nn.Module,
        head_type: str,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        encoder_precision: str = "bf16",
    ) -> None:
        super().__init__()
        validate_encoder_surface(encoder, int(encoder.config.board_size))
        self.encoder = _base_model(encoder)
        self.encoder_config = self.encoder.config
        self.head_type = str(head_type)
        self.encoder_precision = encoder_precision
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

        self.head: nn.Module = build_readout(
            int(self.encoder_config.d_model),
            int(self.encoder_config.vocab_size),
            head_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head.float()

    @property
    def config(self):
        return self.encoder_config

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (B, T), got {tuple(idx.shape)}")
        seq_len = int(idx.shape[1])
        if seq_len > int(self.config.block_size):
            raise ValueError(
                f"Input length {seq_len} exceeds block_size {self.config.block_size}"
            )
        encoder = self.encoder
        positions = torch.arange(seq_len, device=idx.device)
        with torch.no_grad():
            with encoder_amp_context(idx.device, self.encoder_precision):
                hidden = encoder.drop(encoder.wte(idx) + encoder.wpe(positions))
                for block in encoder.blocks:
                    hidden = block(hidden)
                hidden = encoder.ln_f(hidden)
        hidden = hidden.float()
        if not torch.isfinite(hidden).all():
            raise RuntimeError("Frozen encoder produced non-finite hidden states")
        return hidden

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self._encode(idx)
        # ``evaluate_legal_moves`` also uses bf16 autocast for native AR
        # models.  Explicitly disable any enclosing autocast here: the shared
        # frozen encoder is bf16, but the post-hoc readout is a protocol-level
        # fp32 component in both training and evaluation.
        if idx.device.type in {"cuda", "cpu"}:
            readout_context = torch.autocast(
                device_type=idx.device.type,
                enabled=False,
            )
        else:  # pragma: no cover - thesis runs use CUDA; keeps other devices valid
            readout_context = nullcontext()
        with readout_context:
            logits = self.head(hidden.float())
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{self.head_type} readout produced non-finite logits")
        if targets is None:
            return logits, None
        flat_targets = targets.reshape(-1)
        valid_count = int((flat_targets != TARGET_PAD).sum().item())
        if valid_count == 0:
            return logits, logits.sum() * 0.0
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            flat_targets,
            ignore_index=TARGET_PAD,
            reduction="sum",
        ) / valid_count
        if not torch.isfinite(loss):
            raise RuntimeError(f"{self.head_type} readout produced non-finite loss")
        return logits, loss


def _head_seed(base_seed: int, head_type: str) -> int:
    return base_seed + (0 if head_type == "linear" else 10_000)


@dataclass
class _HeadSelectionBatch:
    hidden: torch.Tensor
    legal_mask: torch.Tensor
    valid_mask: torch.Tensor


def _first_raw_games(
    chunks: Sequence[str | Path],
    n_games: int,
) -> list[list[int]]:
    games: list[list[int]] = []
    for chunk_path in chunks:
        for game in load_chunk(str(chunk_path)):
            if len(game) >= 2:
                games.append(list(game))
            if len(games) >= n_games:
                return games
    if len(games) != n_games:
        raise ValueError(f"Loaded {len(games)} games, expected {n_games}")
    return games


def _build_head_selection_cache(
    model: FrozenEncoderNextMoveHead,
    chunks: Sequence[str | Path],
    *,
    config: UnifiedEvalConfig,
    device: torch.device,
) -> tuple[list[_HeadSelectionBatch], float]:
    """Encode the fixed stopping split once and retain it on the GPU."""
    started = time.perf_counter()
    raw_games = _first_raw_games(chunks, config.head_selection_games)
    raw_to_token, _ = build_mappings(config.board_size)
    vocab_size = int(model.config.vocab_size)
    pad_token = config.board_size * config.board_size - 4
    cache: list[_HeadSelectionBatch] = []
    for start in range(0, len(raw_games), config.head_batch_size):
        batch_raw = raw_games[start : start + config.head_batch_size]
        batch_tokens = [
            _tokenize_game(game, raw_to_token, config.board_size)
            for game in batch_raw
        ]
        x_cpu, input_lengths = _pad_token_batch(batch_tokens, pad_token)
        x = x_cpu.to(device, non_blocking=device.type == "cuda")
        hidden = model._encode(x).to(torch.bfloat16)
        max_len = int(hidden.shape[1])
        legal_mask = torch.zeros(
            (len(batch_raw), max_len, vocab_size),
            dtype=torch.bool,
            device=device,
        )
        valid_mask = torch.zeros(
            (len(batch_raw), max_len),
            dtype=torch.bool,
            device=device,
        )
        for row, (game, input_len) in enumerate(zip(batch_raw, input_lengths)):
            legal_sets = replay_game_legality(game, config.board_size)
            for position in range(input_len):
                legal = legal_sets[position + 1]
                if not legal:
                    continue
                indices = torch.tensor(
                    sorted(legal),
                    dtype=torch.long,
                    device=device,
                )
                legal_mask[row, position, indices] = True
                valid_mask[row, position] = True
        cache.append(
            _HeadSelectionBatch(
                hidden=hidden,
                legal_mask=legal_mask,
                valid_mask=valid_mask,
            )
        )
        del x
    return cache, time.perf_counter() - started


def _evaluate_cached_head_selection(
    model: FrozenEncoderNextMoveHead,
    cache: Sequence[_HeadSelectionBatch],
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    mass_sum = 0.0
    top1_sum = 0
    token_count = 0
    with torch.inference_mode():
        for batch in cache:
            with torch.autocast(
                device_type=batch.hidden.device.type,
                enabled=False,
            ):
                logits = model.head(batch.hidden.float())
            probs = torch.softmax(logits, dim=-1)
            legal_mass = (probs * batch.legal_mask).sum(dim=-1)
            top1 = logits.argmax(dim=-1, keepdim=True)
            top1_legal = batch.legal_mask.gather(-1, top1).squeeze(-1)
            valid = batch.valid_mask
            mass_sum += float(legal_mass[valid].sum().cpu())
            top1_sum += int(top1_legal[valid].sum().cpu())
            token_count += int(valid.sum().cpu())
    model.train(was_training)
    if token_count <= 0:
        raise RuntimeError("Frozen-head selection cache has no scorable positions")
    return {
        "n_tokens": token_count,
        "legal_probability_mass": mass_sum / token_count,
        "top1_legal": top1_sum / token_count,
    }


def default_head_grid(head_type: str) -> list[dict[str, Any]]:
    """The deliberately small v4 search space for one readout family.

    Both families include the v3 fixed point (``lr=1e-3`` with no explicit
    regularization, MLP ``hidden=512`` / ``dropout=0.1``), so the tuning gain is
    readable as ``best - v3 default`` from the recorded search table without a
    separate run.
    """
    if head_type == "linear":
        return [
            {
                "label": f"linear__lr={lr:g}__{name}",
                "learning_rate": lr,
                "regularization": name,
                "weight_decay": weight_decay,
                "l1_strength": l1_strength,
            }
            for lr in (3e-4, 1e-3)
            for name, weight_decay, l1_strength in (
                ("none", 0.0, 0.0),
                ("l2=1e-3", 1e-3, 0.0),
                ("l1=1e-5", 0.0, 1e-5),
            )
        ]
    if head_type == "mlp":
        return [
            {
                "label": f"mlp__lr={lr:g}__drop={dropout:g}",
                "learning_rate": lr,
                "regularization": "none",
                "weight_decay": 0.0,
                "l1_strength": 0.0,
                "dropout": dropout,
            }
            for lr in (3e-4, 1e-3)
            for dropout in (0.0, 0.1)
        ]
    raise ValueError("head_type must be 'linear' or 'mlp'")


@dataclass
class HeadTuningCache:
    """Flat, GPU-resident encoder features for the readout hyperparameter search.

    Positions are flattened across games so the search can take real minibatch
    gradient steps; ``fold_ids`` carries the source shard, which is the grouping
    unit for cross-validation.  A game never straddles two folds.
    """

    features: torch.Tensor
    targets: torch.Tensor
    legal_mask: torch.Tensor
    fold_ids: torch.Tensor
    games_per_fold: tuple[int, ...]
    build_seconds: float

    @property
    def n_positions(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_folds(self) -> int:
        return len(self.games_per_fold)

    def fold_index(self, fold: int, *, held_out: bool) -> torch.Tensor:
        mask = self.fold_ids == fold
        if not held_out:
            mask = ~mask
        return mask.nonzero(as_tuple=True)[0]


def _shard_games(chunk_path: str | Path, n_games: int) -> list[list[int]]:
    games = [game for game in load_chunk(str(chunk_path)) if len(game) >= 2]
    if len(games) < n_games:
        raise ValueError(
            f"{Path(chunk_path).name} holds {len(games)} usable games, "
            f"needed {n_games} for the readout search"
        )
    return [list(game) for game in games[:n_games]]


def build_head_tuning_cache(
    model: FrozenEncoderNextMoveHead,
    chunks: Sequence[str | Path],
    *,
    config: UnifiedEvalConfig,
    device: torch.device,
) -> HeadTuningCache:
    """Encode the search corpus once; every grid point then reuses these features.

    The encoder forward is the expensive part of a readout experiment, so it is
    paid once for the whole grid rather than once per configuration.
    """
    started = time.perf_counter()
    folds = config.head_tuning_folds
    if len(chunks) < folds:
        raise ValueError(
            f"Readout search needs {folds} shards to group by, got {len(chunks)}"
        )
    base, remainder = divmod(config.head_tuning_games, folds)
    quotas = [base + (index < remainder) for index in range(folds)]

    raw_to_token, _ = build_mappings(config.board_size)
    vocab_size = int(model.config.vocab_size)
    pad_token = config.board_size * config.board_size - 4

    feature_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    fold_parts: list[torch.Tensor] = []

    for fold, (chunk_path, quota) in enumerate(zip(chunks[:folds], quotas)):
        raw_games = _shard_games(chunk_path, quota)
        for start in range(0, len(raw_games), config.head_batch_size):
            batch_raw = raw_games[start : start + config.head_batch_size]
            batch_tokens = [
                _tokenize_game(game, raw_to_token, config.board_size)
                for game in batch_raw
            ]
            x_cpu, input_lengths = _pad_token_batch(batch_tokens, pad_token)
            x = x_cpu.to(device, non_blocking=device.type == "cuda")
            hidden = model._encode(x).to(torch.bfloat16)

            rows: list[int] = []
            columns: list[int] = []
            keep_rows: list[int] = []
            keep_positions: list[int] = []
            targets: list[int] = []
            for row, (game, tokens, input_len) in enumerate(
                zip(batch_raw, batch_tokens, input_lengths)
            ):
                legal_sets = replay_game_legality(game, config.board_size)
                for position in range(input_len):
                    legal = legal_sets[position + 1]
                    if not legal:
                        continue
                    flat_index = len(targets)
                    for token in legal:
                        rows.append(flat_index)
                        columns.append(token)
                    keep_rows.append(row)
                    keep_positions.append(position)
                    targets.append(tokens[position + 1])
            if not targets:
                del x, hidden
                continue

            mask = torch.zeros((len(targets), vocab_size), dtype=torch.bool)
            mask[
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(columns, dtype=torch.long),
            ] = True
            selected = hidden[
                torch.tensor(keep_rows, dtype=torch.long, device=device),
                torch.tensor(keep_positions, dtype=torch.long, device=device),
            ]
            feature_parts.append(selected)
            target_parts.append(
                torch.tensor(targets, dtype=torch.long, device=device)
            )
            mask_parts.append(mask.to(device, non_blocking=device.type == "cuda"))
            fold_parts.append(
                torch.full((len(targets),), fold, dtype=torch.int8, device=device)
            )
            del x, hidden, selected

    if not feature_parts:
        raise RuntimeError("Readout search corpus produced no scorable positions")
    cache = HeadTuningCache(
        features=torch.cat(feature_parts),
        targets=torch.cat(target_parts),
        legal_mask=torch.cat(mask_parts),
        fold_ids=torch.cat(fold_parts),
        games_per_fold=tuple(quotas),
        build_seconds=time.perf_counter() - started,
    )
    del feature_parts, target_parts, mask_parts, fold_parts
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def _l1_penalty(head: nn.Module) -> torch.Tensor:
    """L1 over weight matrices only; biases stay unpenalized."""
    total: torch.Tensor | None = None
    for name, parameter in head.named_parameters():
        if not name.endswith("weight"):
            continue
        term = parameter.abs().sum()
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("Readout exposes no weight matrices to penalize")
    return total


def _fit_cached_readout(
    cache: HeadTuningCache,
    index: torch.Tensor,
    hyperparameters: Mapping[str, Any],
    *,
    head_type: str,
    config: UnifiedEvalConfig,
    device: torch.device,
    seed: int,
) -> nn.Module:
    """Train one readout on cached features, from a fresh initialization."""
    seed_everything(seed)
    head = build_readout(
        int(cache.features.shape[1]),
        int(cache.legal_mask.shape[1]),
        head_type,
        hidden_dim=int(hyperparameters.get("hidden_dim", config.mlp_hidden_dim)),
        dropout=float(hyperparameters.get("dropout", config.mlp_dropout)),
    ).to(device)
    head.float().train()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters.get("weight_decay", 0.0)),
    )
    l1_strength = float(hyperparameters.get("l1_strength", 0.0))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = config.head_tuning_token_batch
    for epoch in range(config.head_tuning_epochs):
        order = index[
            torch.randperm(index.numel(), generator=generator).to(index.device)
        ]
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = head(cache.features[batch].float())
            loss = F.cross_entropy(logits, cache.targets[batch])
            if l1_strength > 0.0:
                loss = loss + l1_strength * _l1_penalty(head)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Readout search produced a non-finite loss for "
                    f"{hyperparameters['label']}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), config.head_gradient_clip
            )
            optimizer.step()
    head.eval()
    return head


def _score_cached_readout(
    head: nn.Module,
    cache: HeadTuningCache,
    index: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    """Legal mass and top-1 legality of one readout on held-out cached positions."""
    mass_sum = 0.0
    top1_sum = 0
    count = 0
    was_training = head.training
    head.eval()
    with torch.inference_mode():
        for start in range(0, index.numel(), batch_size):
            batch = index[start : start + batch_size]
            logits = head(cache.features[batch].float())
            probabilities = torch.softmax(logits, dim=-1)
            legal = cache.legal_mask[batch]
            mass_sum += float((probabilities * legal).sum(dim=-1).sum().cpu())
            top1 = logits.argmax(dim=-1, keepdim=True)
            top1_sum += int(legal.gather(-1, top1).sum().cpu())
            count += int(batch.numel())
    head.train(was_training)
    if count <= 0:
        raise RuntimeError("Readout search fold has no scorable positions")
    return {
        "n_positions": count,
        "legal_probability_mass": mass_sum / count,
        "top1_legal": top1_sum / count,
    }


def tune_frozen_head_hyperparameters(
    cache: HeadTuningCache,
    *,
    head_type: str,
    config: UnifiedEvalConfig,
    device: torch.device,
    grid: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Grouped-CV grid search for one readout family on cached features.

    Folds are whole shards, so no game contributes to both the fit and the score
    of the same fold.  The winner is the configuration with the best mean
    held-out legal probability mass, the same quantity the streaming refit uses
    for early stopping.
    """
    started = time.perf_counter()
    candidates = list(grid if grid is not None else default_head_grid(head_type))
    if not candidates:
        raise ValueError("The readout search grid is empty")
    base_seed = _head_seed(config.seed, head_type)
    rows: list[dict[str, Any]] = []
    for position, hyperparameters in enumerate(candidates):
        fold_scores: list[dict[str, float]] = []
        for fold in range(cache.n_folds):
            train_index = cache.fold_index(fold, held_out=False)
            validation_index = cache.fold_index(fold, held_out=True)
            head = _fit_cached_readout(
                cache,
                train_index,
                hyperparameters,
                head_type=head_type,
                config=config,
                device=device,
                seed=base_seed + 101 * position + fold,
            )
            fold_scores.append(
                _score_cached_readout(
                    head,
                    cache,
                    validation_index,
                    batch_size=config.head_tuning_token_batch,
                )
            )
            del head
            if device.type == "cuda":
                torch.cuda.empty_cache()
        masses = [score["legal_probability_mass"] for score in fold_scores]
        top1s = [score["top1_legal"] for score in fold_scores]
        mean_mass = sum(masses) / len(masses)
        spread = max(masses) - min(masses)
        rows.append(
            {
                **dict(hyperparameters),
                "cv_legal_probability_mass_mean": mean_mass,
                "cv_legal_probability_mass_min": min(masses),
                "cv_legal_probability_mass_max": max(masses),
                "cv_legal_probability_mass_spread": spread,
                "cv_top1_legal_mean": sum(top1s) / len(top1s),
                "folds": fold_scores,
            }
        )
        print(
            f"{head_type} search [{position + 1}/{len(candidates)}] "
            f"{hyperparameters['label']} "
            f"cv_mass={mean_mass:.4f} (spread {spread:.4f}) "
            f"cv_top1={rows[-1]['cv_top1_legal_mean']:.4f}",
            flush=True,
        )

    ranked = sorted(
        rows,
        key=lambda row: row["cv_legal_probability_mass_mean"],
        reverse=True,
    )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    selected = {
        key: best[key]
        for key in (
            "label",
            "learning_rate",
            "regularization",
            "weight_decay",
            "l1_strength",
        )
        if key in best
    }
    if head_type == "mlp":
        selected["dropout"] = float(best.get("dropout", config.mlp_dropout))
        selected["hidden_dim"] = int(best.get("hidden_dim", config.mlp_hidden_dim))
    print(
        f"{head_type} search selected {selected['label']} "
        f"(cv_mass={best['cv_legal_probability_mass_mean']:.4f})",
        flush=True,
    )
    return {
        "head_type": head_type,
        "selected": selected,
        "selection_metric": "cv_legal_probability_mass_mean",
        "grid_size": len(candidates),
        "folds": cache.n_folds,
        "grouping": "held-out shard (leave-one-shard-out)",
        "tuning_games": int(sum(cache.games_per_fold)),
        "games_per_fold": list(cache.games_per_fold),
        "tuning_positions": cache.n_positions,
        "epochs_per_fit": config.head_tuning_epochs,
        "token_batch": config.head_tuning_token_batch,
        "cache_seconds": cache.build_seconds,
        "search_seconds": time.perf_counter() - started,
        "margin_over_runner_up": (
            best["cv_legal_probability_mass_mean"]
            - runner_up["cv_legal_probability_mass_mean"]
            if runner_up
            else None
        ),
        "runner_up_label": runner_up["label"] if runner_up else None,
        "results": rows,
        "data_source": (
            "head-training pool only; the selection and test shards are never "
            "read by the search"
        ),
    }


def train_frozen_next_move_head(
    encoder: nn.Module,
    chunks: Sequence[str | Path],
    *,
    selection_chunks: Sequence[str | Path],
    head_type: str,
    config: UnifiedEvalConfig,
    device: torch.device | str,
    hyperparameters: Mapping[str, Any] | None = None,
) -> tuple[FrozenEncoderNextMoveHead, dict[str, Any]]:
    """Train a frozen readout until its legal-mass selection curve saturates."""
    device = torch.device(device)
    if not chunks:
        raise ValueError("No head-training chunks were provided")
    if not selection_chunks:
        raise ValueError("No head-selection chunks were provided")
    seed = _head_seed(config.seed, head_type)
    seed_everything(seed)
    # Unspecified fields fall back to the protocol defaults, so a v3-style call
    # with no tuning result behaves exactly as before.
    resolved = dict(hyperparameters or {})
    learning_rate = float(resolved.get("learning_rate", config.head_learning_rate))
    weight_decay = float(resolved.get("weight_decay", config.head_weight_decay))
    l1_strength = float(resolved.get("l1_strength", 0.0))
    hidden_dim = int(resolved.get("hidden_dim", config.mlp_hidden_dim))
    dropout = float(resolved.get("dropout", config.mlp_dropout))
    model = FrozenEncoderNextMoveHead(
        encoder,
        head_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        encoder_precision=config.encoder_precision,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    selection_cache, cache_seconds = _build_head_selection_cache(
        model,
        selection_chunks,
        config=config,
        device=device,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -math.inf
    best_shard = 0
    stale_evaluations = 0
    total_games = 0
    total_tokens = 0
    total_start = time.perf_counter()
    for chunk_index, chunk_path in enumerate(chunks):
        started = time.perf_counter()
        games = load_chunk(str(chunk_path))
        dataset = OthelloChunkDataset(
            games=games,
            block_size=int(model.config.block_size),
            board_size=config.board_size,
        )
        generator = torch.Generator().manual_seed(seed + chunk_index)
        loader = DataLoader(
            dataset,
            batch_size=config.head_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model.train()
        chunk_loss_sum = 0.0
        chunk_tokens = 0
        for x_cpu, y_cpu in loader:
            x = x_cpu.to(device, non_blocking=device.type == "cuda")
            y = y_cpu.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Readout training expected a loss")
            cross_entropy = float(loss.detach())
            if l1_strength > 0.0:
                loss = loss + l1_strength * _l1_penalty(model.head)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head.parameters(), config.head_gradient_clip
            )
            optimizer.step()
            valid = int((y != TARGET_PAD).sum().item())
            # The logged curve stays comparable across regularization settings
            # by tracking cross-entropy rather than the penalized objective.
            chunk_loss_sum += cross_entropy * valid
            chunk_tokens += valid
            del logits, loss

        total_games += len(games)
        total_tokens += chunk_tokens
        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            "shard": chunk_index + 1,
            "chunk_name": Path(chunk_path).name,
            "train_loss": chunk_loss_sum / max(1, chunk_tokens),
            "train_tokens": chunk_tokens,
            "cumulative_tokens": total_tokens,
            "train_games": len(games),
            "cumulative_games": total_games,
        }
        should_evaluate = (
            (chunk_index + 1) % config.head_eval_every_shards == 0
            or chunk_index + 1 == len(chunks)
        )
        if should_evaluate:
            selection = _evaluate_cached_head_selection(model, selection_cache)
            metric = float(selection["legal_probability_mass"])
            improved = metric > best_metric + config.head_min_delta
            if improved:
                best_metric = metric
                best_shard = chunk_index + 1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.head.state_dict().items()
                }
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            row.update(
                {
                    "selection": selection,
                    "improved": improved,
                    "stale_evaluations": stale_evaluations,
                }
            )
        history.append(row)
        selection_text = (
            f" val_mass={row['selection']['legal_probability_mass']:.4f} "
            f"val_top1={row['selection']['top1_legal']:.4f} "
            f"best={best_shard} stale={stale_evaluations}"
            if "selection" in row
            else ""
        )
        print(
            f"{head_type} head shard={chunk_index + 1}/{len(chunks)} "
            f"games={total_games:,} tokens={total_tokens:,} "
            f"loss={row['train_loss']:.4f}{selection_text} "
            f"dt={elapsed:.1f}s",
            flush=True,
        )
        del loader, dataset, games
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (
            should_evaluate
            and stale_evaluations >= config.head_patience
        ):
            print(
                f"{head_type} head early stop at shard={chunk_index + 1}; "
                f"restoring shard={best_shard}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Frozen-head training produced no selection checkpoint")
    model.head.load_state_dict(best_state)
    model.eval()
    return model, {
        "head_type": head_type,
        "seed": seed,
        "batch_size": config.head_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "l1_strength": l1_strength,
        "regularization": resolved.get(
            "regularization", "none" if weight_decay == 0.0 else "l2"
        ),
        "hidden_dim": hidden_dim if head_type == "mlp" else None,
        "dropout": dropout if head_type == "mlp" else None,
        "hyperparameters_source": (
            "grouped-CV search" if hyperparameters else "protocol defaults"
        ),
        "gradient_clip": config.head_gradient_clip,
        "max_shards": len(chunks),
        "shards_completed": len(history),
        "best_shard": best_shard,
        "best_selection_legal_probability_mass": best_metric,
        "selection_games": config.head_selection_games,
        "eval_every_shards": config.head_eval_every_shards,
        "patience": config.head_patience,
        "min_delta": config.head_min_delta,
        "selection_cache_seconds": cache_seconds,
        "encoder_precision": config.encoder_precision,
        "readout_precision": config.readout_precision,
        "history": history,
        "wall_seconds": time.perf_counter() - total_start,
    }


def compact_legal_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep finite aggregate metrics and count-backed position curves."""
    aggregate_keys = (
        "n_games",
        "n_tokens",
        "top1_legal_per_token",
        "top1_legal_per_game",
        "topk_legal",
        "legal_prob_mass_per_token",
        "legal_prob_mass_per_game",
        "mean_legal_moves_per_token",
        "random_vocab_top1_legal_baseline",
        "top1_legal_normalized_lift",
        "bootstrap_ci_95_per_game",
    )
    compact = {key: result[key] for key in aggregate_keys if key in result}
    counts = list(result.get("n_tokens_per_position", []))
    curves = {
        "top1_legal": result.get("top1_legal_per_position", []),
        "legal_probability_mass": result.get("legal_prob_mass_per_position", []),
        "mean_legal_moves": result.get("n_legal_per_position_mean", []),
    }
    compact["position_curve"] = [
        {
            "position": position,
            "normalized_phase": position / max(1, len(counts) - 1),
            "n_tokens": int(count),
            **{
                name: float(values[position])
                for name, values in curves.items()
                if position < len(values)
                and values[position] is not None
                and math.isfinite(float(values[position]))
            },
        }
        for position, count in enumerate(counts)
        if count > 0
    ]
    phase_bins: list[dict[str, Any]] = []
    for bin_index in range(4):
        members = [
            row
            for row in compact["position_curve"]
            if min(3, int(float(row["normalized_phase"]) * 4)) == bin_index
        ]
        total = sum(int(row["n_tokens"]) for row in members)
        if not total:
            continue
        metric_names = set().union(*(set(row) for row in members)) - {
            "position",
            "normalized_phase",
            "n_tokens",
        }
        phase_bins.append(
            {
                "phase_bin": bin_index,
                "phase_start": bin_index / 4,
                "phase_end": (bin_index + 1) / 4,
                "n_tokens": total,
                **{
                    name: sum(
                        float(row[name]) * int(row["n_tokens"])
                        for row in members
                        if name in row
                    )
                    / sum(
                        int(row["n_tokens"]) for row in members if name in row
                    )
                    for name in metric_names
                },
            }
        )
    compact["normalized_phase_bins"] = phase_bins
    _assert_finite_tree(compact)
    return compact


def _assert_finite_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric at {path}: {value}")


def file_sha256(path: str | Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"Refusing to serialize non-finite value {value}")
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return _jsonable(value.item())
    return str(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _eligible_game_records(
    chunks: Sequence[str | Path],
    *,
    min_game_length: int,
    games_per_shard: int | None = None,
    total_games: int | None = None,
) -> list[tuple[Path, int]]:
    """Select deterministic corpus records while preserving game indices."""
    if (games_per_shard is None) == (total_games is None):
        raise ValueError("Specify exactly one of games_per_shard or total_games")
    records: list[tuple[Path, int]] = []
    for chunk_path in chunks:
        chunk_path = Path(chunk_path)
        games = load_chunk(str(chunk_path))
        selected = [
            (chunk_path, game_index)
            for game_index, game in enumerate(games)
            if len(game) >= min_game_length
        ]
        if games_per_shard is not None:
            if len(selected) < games_per_shard:
                raise ValueError(
                    f"{chunk_path} has only {len(selected)} eligible games; "
                    f"need {games_per_shard}"
                )
            records.extend(selected[:games_per_shard])
        else:
            assert total_games is not None
            remaining = total_games - len(records)
            records.extend(selected[:remaining])
            if len(records) >= total_games:
                break
        del games
    expected = (
        int(games_per_shard) * len(chunks)
        if games_per_shard is not None
        else int(total_games or 0)
    )
    if len(records) != expected:
        raise ValueError(f"Selected {len(records)} eligible games, expected {expected}")
    return records


def _manifest_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_RESUME_IDENTITY_KEYS = (
    "split_manifest_hash",
    "checkpoint_sha256",
    "data_manifest_sha256",
    "project_source_sha256",
    "split_content_sha256",
)


def _resume_mismatches(
    existing: Mapping[str, Any],
    fresh: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Every identity field that stops ``existing`` results from being resumed."""
    mismatches: dict[str, tuple[Any, Any]] = {}
    if existing.get("schema_version") != fresh.get("schema_version"):
        mismatches["schema_version"] = (
            existing.get("schema_version"),
            fresh.get("schema_version"),
        )
    existing_meta = _jsonable(existing.get("metadata") or {})
    fresh_meta = _jsonable(fresh.get("metadata") or {})
    for key in _RESUME_IDENTITY_KEYS:
        if existing_meta.get(key) != fresh_meta.get(key):
            mismatches[key] = (existing_meta.get(key), fresh_meta.get(key))
    existing_protocol = existing_meta.get("protocol") or {}
    fresh_protocol = fresh_meta.get("protocol") or {}
    for field in sorted(set(existing_protocol) | set(fresh_protocol)):
        if existing_protocol.get(field) != fresh_protocol.get(field):
            mismatches[f"protocol.{field}"] = (
                existing_protocol.get(field),
                fresh_protocol.get(field),
            )
    return mismatches


def _abbreviate(value: Any) -> str:
    text = "None" if value is None else str(value)
    return f"{text[:12]}…" if len(text) > 16 else text


def _format_mismatches(mismatches: Mapping[str, tuple[Any, Any]]) -> str:
    return "\n".join(
        f"  - {key}: saved={_abbreviate(saved)} current={_abbreviate(current)}"
        for key, (saved, current) in mismatches.items()
    )


def _completed_stages(results: Mapping[str, Any]) -> list[str]:
    stages = ["native_ar"] if results.get("native_ar") else []
    for group in ("frozen_next_move", "board_state"):
        stages.extend(f"{group}.{name}" for name in sorted(results.get(group) or ()))
    if results.get("causal_intervention"):
        stages.append("causal_intervention")
    return stages


class UnifiedEvaluator:
    """Stage-oriented evaluator used directly by the Colab notebooks."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        split: EvaluationSplit,
        config: UnifiedEvalConfig,
        device: torch.device | str,
        output_dir: str | Path,
        run_metadata: Mapping[str, Any],
        native_ar_model: nn.Module | None = None,
        overwrite_incompatible: bool = False,
        allow_source_drift: bool = False,
    ) -> None:
        self.encoder = _base_model(encoder)
        self.native_ar_model = native_ar_model
        self.split = split
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._position_splits: PositionProbeSplits | None = None
        self._position_features: tuple[
            PositionFeatureSet, PositionFeatureSet, PositionFeatureSet
        ] | None = None
        self._head_tuning_cache: HeadTuningCache | None = None

        validate_runtime(self.device, config)
        validate_encoder_surface(self.encoder, config.board_size)
        assert_finite_module(self.encoder, "encoder")
        self.encoder.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if native_ar_model is not None:
            native_ar_model.to(self.device).eval()
            if _base_model(native_ar_model) is not self.encoder:
                assert_finite_module(native_ar_model, "native_ar_model")

        fresh_results: dict[str, Any] = {
            "schema_version": config.protocol_id,
            "metadata": {
                **dict(run_metadata),
                "protocol": asdict(config),
                "split": split.to_dict(),
                "split_manifest_hash": split.manifest_hash,
                "device": str(self.device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(self.device)
                    if self.device.type == "cuda"
                    else None
                ),
                "python": platform.python_version(),
                "implicit_pass_policy": "score effective side-to-move",
                "layer_semantics": "L0 embeddings; L1..LN post-block residual",
            },
            "native_ar": None,
            "frozen_next_move": {},
            "board_state": {},
            "causal_intervention": None,
            "timing_seconds": {},
        }
        existing_path = self.results_path
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            mismatches = _resume_mismatches(existing, fresh_results)
            source_drift_only = set(mismatches) == {"project_source_sha256"}
            if not mismatches:
                self.results = existing
                print(f"Resuming compatible evaluation state: {existing_path}")
            elif source_drift_only and allow_source_drift:
                saved_hash, current_hash = mismatches["project_source_sha256"]
                stages = _completed_stages(existing)
                self.results = existing
                metadata = self.results.setdefault("metadata", {})
                metadata.setdefault("source_drift_events", []).append(
                    {
                        "previous_project_source_sha256": saved_hash,
                        "current_project_source_sha256": current_hash,
                        "stages_completed_before_drift": stages,
                    }
                )
                metadata["project_source_sha256"] = current_hash
                print(
                    "WARNING: resuming across a project source change "
                    f"({_abbreviate(saved_hash)} -> {_abbreviate(current_hash)}). "
                    f"Stages computed with the older source: {stages or 'none'}. "
                    "The drift is recorded in metadata.source_drift_events; only "
                    "keep these results if the change cannot affect them."
                )
                self.save_results()
            elif overwrite_incompatible:
                self.results = fresh_results
                self.save_results()
            else:
                hint = (
                    "Only the project source digest changed. If the edited code "
                    "cannot affect the stages already saved, pass "
                    "allow_source_drift=True to resume them (the drift is then "
                    "recorded in the results); otherwise recompute everything."
                    if source_drift_only
                    else "Recompute everything: the saved stages describe a "
                    "different checkpoint, corpus, split, or protocol."
                )
                raise RuntimeError(
                    f"Existing {existing_path} is not resumable; identity "
                    f"mismatch on:\n{_format_mismatches(mismatches)}\n"
                    f"{hint} Use a different output directory or set "
                    "overwrite_incompatible=True explicitly to discard "
                    f"{_completed_stages(existing) or 'the saved state'}."
                )
        else:
            self.results = fresh_results
            self.save_results()

    @property
    def results_path(self) -> Path:
        return self.output_dir / f"results__{self.config.protocol_id}.json"

    def save_results(self) -> Path:
        _atomic_json(self.results_path, self.results)
        return self.results_path

    def run_native_ar(self, *, force: bool = False) -> dict[str, Any]:
        if self.native_ar_model is None:
            raise RuntimeError("This evaluator has no native AR model")
        if self.results.get("native_ar") and not force:
            print("native AR stage already complete; reusing saved result")
            return self.results["native_ar"]
        started = time.perf_counter()
        raw = evaluate_legal_moves(
            model=self.native_ar_model,
            val_chunks=self.split.test,
            board_size=self.config.board_size,
            n_games=self.config.legal_eval_games,
            device=self.device,
            k_values=self.config.legal_k_values,
            batch_size=self.config.legal_eval_batch_size,
        )
        result = compact_legal_result(raw)
        self.results["native_ar"] = result
        self.results["timing_seconds"]["native_ar"] = time.perf_counter() - started
        self.save_results()
        return result

    def _head_tuning_features(self) -> HeadTuningCache:
        """Encode the search corpus once and share it across both head families."""
        if self._head_tuning_cache is None:
            print(
                f"Building readout-search cache: {self.config.head_tuning_games:,} "
                f"games over {self.config.head_tuning_folds} grouped folds",
                flush=True,
            )
            # Any FrozenEncoderNextMoveHead exposes the same frozen encoder
            # surface; the untrained readout it carries is never used here.
            encoder_view = FrozenEncoderNextMoveHead(
                self.encoder,
                "linear",
                encoder_precision=self.config.encoder_precision,
            ).to(self.device)
            self._head_tuning_cache = build_head_tuning_cache(
                encoder_view,
                self.split.head_training_pool(self.config.head_max_shards),
                config=self.config,
                device=self.device,
            )
            del encoder_view
            cache = self._head_tuning_cache
            print(
                f"Readout-search cache: {cache.n_positions:,} positions in "
                f"{cache.build_seconds:.1f}s",
                flush=True,
            )
        return self._head_tuning_cache

    def release_head_tuning_cache(self) -> None:
        """Free the search features before the memory-hungry streaming refit."""
        if self._head_tuning_cache is None:
            return
        self._head_tuning_cache = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def run_frozen_head(
        self,
        head_type: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        saved = self.results.get("frozen_next_move", {}).get(head_type)
        if saved and not force and Path(saved.get("artifact_path", "")).is_file():
            print(f"{head_type} frozen-head stage already complete; reusing artifact")
            return saved
        started = time.perf_counter()
        tuning: dict[str, Any] | None = None
        selected: Mapping[str, Any] | None = None
        if self.config.head_tuning_enabled:
            tuning = tune_frozen_head_hyperparameters(
                self._head_tuning_features(),
                head_type=head_type,
                config=self.config,
                device=self.device,
            )
            selected = tuning["selected"]
            self.results["timing_seconds"][f"head_tuning_{head_type}"] = (
                time.perf_counter() - started
            )
        model, train_result = train_frozen_next_move_head(
            self.encoder,
            self.split.head_training_pool(self.config.head_max_shards),
            selection_chunks=self.split.selection,
            head_type=head_type,
            config=self.config,
            device=self.device,
            hyperparameters=selected,
        )
        head_path = (
            self.output_dir
            / f"{head_type}_next_move_head__{self.config.protocol_id}.pt"
        )
        torch.save(
            {
                "schema_version": self.config.protocol_id,
                "head_type": head_type,
                "head": model.head.state_dict(),
                "split_manifest_hash": self.split.manifest_hash,
                "run_metadata": dict(self.results["metadata"]),
                "tuning": tuning,
                "train": train_result,
            },
            head_path,
        )
        legal_raw = evaluate_legal_moves(
            model=model,
            val_chunks=self.split.test,
            board_size=self.config.board_size,
            n_games=self.config.legal_eval_games,
            device=self.device,
            k_values=self.config.legal_k_values,
            batch_size=self.config.legal_eval_batch_size,
        )
        result = {
            "head_type": head_type,
            "artifact_path": str(head_path),
            "tuning": tuning,
            "train": train_result,
            "test_legal": compact_legal_result(legal_raw),
        }
        del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.results["frozen_next_move"][head_type] = result
        self.results["timing_seconds"][f"frozen_head_{head_type}"] = (
            time.perf_counter() - started
        )
        self.save_results()
        return result

    def _prepare_position_features(
        self,
    ) -> tuple[PositionFeatureSet, PositionFeatureSet, PositionFeatureSet]:
        if self._position_features is not None:
            return self._position_features

        # The readout search cache is no longer needed once the board stage
        # starts, and it is the largest resident tensor on the device.
        self.release_head_tuning_cache()
        started = time.perf_counter()
        min_length = self.config.board_positions_per_game + 1
        train_records = _eligible_game_records(
            self.split.downstream_train,
            min_game_length=min_length,
            total_games=self.config.board_train_games,
        )
        selection_records = _eligible_game_records(
            self.split.selection,
            min_game_length=min_length,
            total_games=self.config.board_selection_games,
        )
        test_records = _eligible_game_records(
            self.split.test,
            min_game_length=min_length,
            total_games=self.config.board_test_games,
        )
        self._position_splits = build_position_probe_splits(
            train_records=train_records,
            validation_records=selection_records,
            test_records=test_records,
            board_size=self.config.board_size,
            seed=self.config.seed,
            n_phase_bins=self.config.board_positions_per_game,
        )
        manifest = self._position_splits.manifest()
        manifest_hash = _manifest_sha256(manifest)
        saved_manifest = self.results.get("metadata", {}).get("position_manifest")
        if (
            saved_manifest
            and saved_manifest.get("sha256") != manifest_hash
        ):
            raise RuntimeError(
                "Regenerated position manifest does not match the saved stage. "
                "The corpus or sampling code changed; start a new evaluation."
            )
        manifest_path = (
            self.output_dir / f"position_manifest__{self.config.protocol_id}.json"
        )
        _atomic_json(manifest_path, manifest)

        layers = tuple(range(len(self.encoder.blocks) + 1))
        print(
            f"Extracting fixed board-position features: "
            f"train={len(self._position_splits.train):,}, "
            f"selection={len(self._position_splits.validation):,}, "
            f"test={len(self._position_splits.test):,}",
            flush=True,
        )
        train_features = extract_position_features(
            self.encoder,
            self._position_splits.train,
            board_size=self.config.board_size,
            layers=layers,
            batch_size=self.config.board_batch_size,
            device=self.device,
        )
        selection_features = extract_position_features(
            self.encoder,
            self._position_splits.validation,
            board_size=self.config.board_size,
            layers=layers,
            batch_size=self.config.board_batch_size,
            device=self.device,
        )
        test_features = extract_position_features(
            self.encoder,
            self._position_splits.test,
            board_size=self.config.board_size,
            layers=layers,
            batch_size=self.config.board_batch_size,
            device=self.device,
        )
        self._position_features = (
            train_features,
            selection_features,
            test_features,
        )
        self.results["metadata"]["position_manifest"] = {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "train_records": len(train_records),
            "selection_records": len(selection_records),
            "test_records": len(test_records),
            "train_positions": train_features.n_samples,
            "selection_positions": selection_features.n_samples,
            "test_positions": test_features.n_samples,
            "phase_bins": self.config.board_positions_per_game,
        }
        self.results["timing_seconds"]["board_feature_extraction"] = (
            time.perf_counter() - started
        )
        self.save_results()
        return self._position_features

    def run_board_probe(
        self,
        probe_type: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run one common phase-stratified board probe and save immediately."""
        if probe_type not in {"linear", "mlp"}:
            raise ValueError("probe_type must be 'linear' or 'mlp'")
        saved = self.results.get("board_state", {}).get(probe_type)
        if saved and not force and Path(saved.get("artifact_path", "")).is_file():
            print(f"{probe_type} board-probe stage already complete; reusing artifact")
            return saved
        train_features, selection_features, test_features = (
            self._prepare_position_features()
        )
        started = time.perf_counter()
        learning_rate = (
            self.config.board_linear_lr
            if probe_type == "linear"
            else self.config.board_mlp_lr
        )
        bank, raw_result = train_position_probe_bank(
            train_features,
            selection_features,
            test_features,
            probe_type=probe_type,
            device=self.device,
            hidden_dim=self.config.mlp_hidden_dim,
            dropout=self.config.mlp_dropout,
            epochs=self.config.board_epochs,
            batch_size=self.config.board_probe_batch_size,
            lr=learning_rate,
            weight_decay=self.config.board_weight_decay,
            gradient_clip=self.config.board_gradient_clip,
            seed=self.config.seed + (0 if probe_type == "linear" else 20_000),
            selection_metric="macro_balanced_accuracy",
            input_layernorm=self.config.board_input_layernorm,
            patience=self.config.board_patience,
            min_delta=self.config.board_min_delta,
        )
        manifest_metadata = self.results["metadata"]["position_manifest"]
        # Exact sample ids live once in the versioned manifest rather than
        # being duplicated into both the linear and MLP result trees.
        raw_result.pop("sample_ids", None)
        n_layers = len(self.encoder.blocks)
        raw_result["normalized_depth"] = {
            str(layer): layer / max(1, n_layers)
            for layer in raw_result["layers"]
        }
        final_layer = int(raw_result["layers"][-1])
        raw_result["final_layer_test"] = {
            mode: raw_result["test_metrics"][str(final_layer)][mode]
            for mode in ("absolute", "relative")
        }
        raw_result["manifest_path"] = manifest_metadata["path"]
        raw_result["manifest_sha256"] = manifest_metadata["sha256"]

        artifact_path = (
            self.output_dir
            / f"{probe_type}_board_probe__{self.config.protocol_id}.pt"
        )
        torch.save(
            {
                "schema_version": self.config.protocol_id,
                "probe_type": probe_type,
                "probe_bank": bank.state_dict(),
                "position_manifest_sha256": manifest_metadata["sha256"],
                "result": raw_result,
            },
            artifact_path,
        )
        result = {**raw_result, "artifact_path": str(artifact_path)}
        del bank
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        _assert_finite_tree(result)
        self.results["board_state"][probe_type] = result
        self.results["timing_seconds"][f"board_probe_{probe_type}"] = (
            time.perf_counter() - started
        )
        self.save_results()
        return result

    def _intervention_probe_bank(self) -> tuple[PositionProbeBank, Path]:
        """Load or fit the raw linear probe required for residual directions."""
        artifact_path = (
            self.output_dir
            / f"raw_linear_intervention_probe__{self.config.protocol_id}.pt"
        )
        layers = tuple(range(len(self.encoder.blocks) + 1))
        d_model = int(self.encoder.config.d_model)
        n_squares = self.config.board_size * self.config.board_size
        bank = PositionProbeBank(
            layers,
            d_model,
            n_squares,
            probe_type="linear",
            hidden_dim=self.config.mlp_hidden_dim,
            dropout=0.0,
            input_layernorm=False,
        ).to(self.device)
        if artifact_path.is_file():
            payload = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=False,
            )
            if payload.get("schema_version") != self.config.protocol_id:
                raise RuntimeError("Incompatible intervention-probe artifact")
            bank.load_state_dict(payload["probe_bank"])
            bank.eval()
            return bank, artifact_path

        train_features, selection_features, test_features = (
            self._prepare_position_features()
        )
        bank, raw_result = train_position_probe_bank(
            train_features,
            selection_features,
            test_features,
            probe_type="linear",
            device=self.device,
            hidden_dim=self.config.mlp_hidden_dim,
            dropout=0.0,
            epochs=self.config.board_epochs,
            batch_size=self.config.board_probe_batch_size,
            lr=self.config.board_linear_lr,
            weight_decay=self.config.board_weight_decay,
            gradient_clip=self.config.board_gradient_clip,
            seed=self.config.seed + 40_000,
            selection_metric="macro_balanced_accuracy",
            input_layernorm=False,
            patience=self.config.board_patience,
            min_delta=self.config.board_min_delta,
            early_stopping_modes=("relative",),
        )
        raw_result.pop("sample_ids", None)
        torch.save(
            {
                "schema_version": self.config.protocol_id,
                "probe_type": "linear_relative_raw_intervention",
                "probe_bank": bank.state_dict(),
                "result": raw_result,
            },
            artifact_path,
        )
        bank.eval()
        return bank, artifact_path

    def run_causal_intervention(
        self,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run the optional 8x8 linear residual-editing causal diagnostic."""
        if self.config.board_size != 8:
            raise ValueError(
                "The literature-anchored causal intervention is restricted to 8x8"
            )
        if self.results.get("causal_intervention") and not force:
            print("causal-intervention stage already complete; reusing result")
            return self.results["causal_intervention"]

        from othello_thesis.evaluation.causal_intervention import (
            build_intervention_cases,
            evaluate_causal_intervention,
        )

        started = time.perf_counter()
        probe_bank, probe_path = self._intervention_probe_bank()
        readout_wrapper: FrozenEncoderNextMoveHead | None = None
        if self.native_ar_model is not None:
            readout = _base_model(self.native_ar_model).lm_head
            readout_label = "native_ar"
        else:
            saved = self.results.get("frozen_next_move", {}).get("mlp")
            if not saved:
                saved = self.run_frozen_head("mlp")
            payload = torch.load(
                saved["artifact_path"],
                map_location="cpu",
                weights_only=False,
            )
            readout_wrapper = FrozenEncoderNextMoveHead(
                self.encoder,
                "mlp",
                hidden_dim=self.config.mlp_hidden_dim,
                dropout=self.config.mlp_dropout,
                encoder_precision=self.config.encoder_precision,
            ).to(self.device)
            readout_wrapper.head.load_state_dict(payload["head"])
            readout_wrapper.eval()
            readout = readout_wrapper.head
            readout_label = "frozen_mlp"

        selection_cases = build_intervention_cases(
            self.split.selection,
            board_size=self.config.board_size,
            n_cases=self.config.intervention_selection_cases,
            seed=self.config.seed + 50_000,
        )
        test_cases = build_intervention_cases(
            self.split.test,
            board_size=self.config.board_size,
            n_cases=self.config.intervention_test_cases,
            seed=self.config.seed + 60_000,
        )
        result = evaluate_causal_intervention(
            self.encoder,
            readout,
            probe_bank,
            selection_cases=selection_cases,
            test_cases=test_cases,
            board_size=self.config.board_size,
            device=self.device,
            alpha_values=self.config.intervention_alpha_values,
            batch_size=self.config.intervention_batch_size,
            seed=self.config.seed + 70_000,
        )
        result["readout"] = readout_label
        result["probe_artifact_path"] = str(probe_path)
        result["interpretation_scope"] = (
            "native model behavior"
            if readout_label == "native_ar"
            else "JEPA encoder plus post-hoc frozen MLP readout"
        )
        _assert_finite_tree(result)
        self.results["causal_intervention"] = result
        self.results["timing_seconds"]["causal_intervention"] = (
            time.perf_counter() - started
        )
        self.save_results()
        del probe_bank, readout_wrapper
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def write_summary(self) -> Path:
        """Generate the single Markdown report directly from the JSON schema."""
        metadata = self.results["metadata"]

        def pct(value: float) -> str:
            return f"{100.0 * float(value):.2f}%"

        def topk(metrics: Mapping[str, Any], k: int) -> float:
            values = metrics["topk_legal"]
            return float(values.get(k, values.get(str(k))))

        lines = [
            f"# Unified evaluation: {metadata.get('run_name', 'unnamed run')}",
            "",
            f"- **Protocol:** `{self.config.protocol_id}`",
            f"- **Board:** `{self.config.board_size}x{self.config.board_size}`",
            f"- **Architecture/objective:** `{metadata.get('architecture')}` / "
            f"`{metadata.get('objective')}`",
            f"- **Checkpoint:** `{metadata.get('checkpoint_name')}` "
            f"(step `{metadata.get('step')}`, games `{int(metadata.get('games_seen') or 0):,}`)",
            f"- **Checkpoint SHA-256:** `{metadata.get('checkpoint_sha256')}`",
            f"- **Split manifest SHA-256:** `{self.split.manifest_hash}`",
            f"- **Data-manifest SHA-256:** `{metadata.get('data_manifest_sha256')}`",
            "- **Per-shard corpus identity:** "
            f"`{metadata.get('split_content_sha256') or 'unavailable-counts-only'}`",
            f"- **Project-source SHA-256:** `{metadata.get('project_source_sha256')}`",
            "- **Exact pretraining budget:** "
            f"`{int(metadata.get('games_seen') or 0):,}` games / "
            f"`{int(metadata.get('chunk_count') or 0)}` shards",
            "- **Checkpoint-reported training precision:** "
            f"`{metadata.get('training_precision') or 'unreported'}`",
            "- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32",
            "- **Pass policy:** effective side-to-move; implicit-pass positions are scored",
        ]
        if metadata.get("architecture") == "mamba":
            lines.append(
                "- **Mamba runtime:** "
                f"`mamba-ssm {metadata.get('mamba_ssm_version')}` / "
                f"`causal-conv1d {metadata.get('causal_conv1d_version')}`"
            )
        objective = str(metadata.get("objective", "")).lower()
        if objective == "ar":
            native = self.results.get("native_ar")
            lines += [
                "",
                "## Native AR next-move prediction",
                "",
                "| Readout | Top-1 legal | Top-3 legal fraction | "
                "Top-5 legal fraction | Legal mass |",
                "|---|---:|---:|---:|---:|",
            ]
            if native:
                lines.append(
                    f"| Native AR | {pct(native['top1_legal_per_token'])} | "
                    f"{pct(topk(native, 3))} | {pct(topk(native, 5))} | "
                    f"{pct(native['legal_prob_mass_per_token'])} |"
                )
        else:
            lines += [
                "",
                "## Frozen-encoder next-move readouts",
                "",
                "| Readout | Top-1 legal | Top-3 legal fraction | "
                "Top-5 legal fraction | Legal mass |",
                "|---|---:|---:|---:|---:|",
            ]
            for head_type in ("linear", "mlp"):
                if head_type not in self.results["frozen_next_move"]:
                    continue
                metrics = self.results["frozen_next_move"][head_type]["test_legal"]
                lines.append(
                    f"| {head_type.upper()} | "
                    f"{pct(metrics['top1_legal_per_token'])} | "
                    f"{pct(topk(metrics, 3))} | "
                    f"{pct(topk(metrics, 5))} | "
                    f"{pct(metrics['legal_prob_mass_per_token'])} |"
                )

        lines += [
            "",
            "## Phase-stratified board-state probes",
            "",
            "Layers are selected only on the selection split; every number in the "
            "`Test` columns comes from the untouched final test split.",
            "",
            "| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for probe_type in ("linear", "mlp"):
            probe = self.results["board_state"].get(probe_type)
            if not probe:
                continue
            for mode in ("absolute", "relative"):
                selected = probe["selected"][mode]
                layer = int(selected["layer"])
                val = selected["validation"]
                test = selected["test"]
                lines.append(
                    f"| {probe_type.upper()} | {mode} | L{layer} | "
                    f"{float(probe['normalized_depth'][str(layer)]):.3f} | "
                    f"{pct(val['macro_balanced_accuracy'])} | "
                    f"{pct(test['macro_balanced_accuracy'])} | "
                    f"{pct(test['accuracy'])} | {pct(test['occupied_accuracy'])} | "
                    f"{pct(test['empty_accuracy'])} |"
                )

        if all(
            probe_type in self.results["board_state"]
            for probe_type in ("linear", "mlp")
        ):
            lines += [
                "",
                "## Board-state probe accuracy by layer",
                "",
                "Each layer table reports untouched test-split accuracy. "
                "`Selected` marks validation-selected layers.",
            ]
            for probe_type in ("linear", "mlp"):
                probe = self.results["board_state"][probe_type]
                selected_layers = {
                    mode: int(probe["selected"][mode]["layer"])
                    for mode in ("absolute", "relative")
                }
                lines += [
                    "",
                    f"### {probe_type.upper()} board-state probe",
                    "",
                    "| Layer | Absolute accuracy | Absolute macro | "
                    "Relative accuracy | Relative macro | Selected |",
                    "|---:|---:|---:|---:|---:|:---|",
                ]
                for layer in probe["layers"]:
                    key = str(layer)
                    test = probe["test_metrics"][key]
                    selected_modes = " + ".join(
                        mode
                        for mode in ("absolute", "relative")
                        if selected_layers[mode] == int(layer)
                    )
                    lines.append(
                        f"| L{layer} | {pct(test['absolute']['accuracy'])} | "
                        f"{pct(test['absolute']['macro_balanced_accuracy'])} | "
                        f"{pct(test['relative']['accuracy'])} | "
                        f"{pct(test['relative']['macro_balanced_accuracy'])} | "
                        f"{selected_modes} |"
                    )

        intervention = self.results.get("causal_intervention")
        if intervention:
            lines += [
                "",
                "## Nanda-style causal intervention (8x8)",
                "",
                f"- **Readout:** `{intervention['readout']}`",
                f"- **Selected alpha:** `{float(intervention['selected_alpha']):g}`",
                f"- **Interpretation scope:** "
                f"{intervention['interpretation_scope']}",
                "",
                "| Control | Target top-1 legal | Target legal mass | "
                "Top-N FP+FN |",
                "|---|---:|---:|---:|",
            ]
            for key, label in (
                ("null", "Null"),
                ("magnitude_matched_random", "Random direction"),
                ("probe_direction", "Relative-board direction"),
            ):
                metrics = intervention["test"][key]
                lines.append(
                    f"| {label} | {pct(metrics['target_top1_legal'])} | "
                    f"{pct(metrics['target_legal_probability_mass'])} | "
                    f"{float(metrics['mean_topn_false_positive_plus_false_negative']):.3f} |"
                )

        lines += [
            "",
            "## Artifacts and timing",
            "",
            f"- Results JSON: `{self.results_path}`",
        ]
        for head_type, result in self.results["frozen_next_move"].items():
            lines.append(f"- {head_type} next-move head: `{result['artifact_path']}`")
        for probe_type, result in self.results["board_state"].items():
            lines.append(f"- {probe_type} board probe: `{result['artifact_path']}`")
        if "position_manifest" in metadata:
            lines.append(
                f"- Position manifest: `{metadata['position_manifest']['path']}`"
            )
        lines += ["", "| Stage | Wall time |", "|---|---:|"]
        for stage, seconds in self.results["timing_seconds"].items():
            lines.append(f"| {stage} | {float(seconds):.1f} s |")

        path = self.output_dir / f"summary__{self.config.protocol_id}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.results["summary_path"] = str(path)
        self.save_results()
        return path

    def print_protocol(self) -> None:
        print(f"Protocol             : {self.config.protocol_id}")
        print(f"Board                : {self.config.board_size}x{self.config.board_size}")
        print(f"Encoder/readout      : bf16 / fp32")
        print(
            "Shard split          : "
            f"0:{self.config.pretraining_shards} pretrain, "
            f"{self.config.pretraining_shards}:"
            f"{self.config.pretraining_shards + self.config.downstream_train_shards} "
            f"readout train, then {self.config.selection_shards} selection + "
            f"{self.config.test_shards} test"
        )
        print(f"Split manifest SHA256: {self.split.manifest_hash}")
        print(
            "Frozen-head stopping : "
            f"no minimum, max={self.config.head_max_shards} shards, "
            f"patience={self.config.head_patience}, "
            f"min_delta={self.config.head_min_delta:g}"
        )
        print(f"Final legal test      : {self.config.legal_eval_games:,} games")
        print(
            "Board-probe budget    : "
            f"{self.config.board_train_games:,} games x "
            f"{self.config.board_positions_per_game} positions = "
            f"{self.config.board_train_games * self.config.board_positions_per_game:,}"
        )
