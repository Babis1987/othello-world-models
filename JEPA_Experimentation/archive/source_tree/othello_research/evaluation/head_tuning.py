"""Leakage-safe head-only tuning for the 8x8 JEPA thesis study.

The common evaluator deliberately fixes one readout configuration so every
architecture is measured under the same protocol.  This module answers a
different question: whether the conclusion changes after tuning *only* the
linear and MLP next-move readouts.

The expensive encoder pass is cached once.  Hyper-parameters are then selected
with shard-grouped cross-validation on ``downstream_train`` only.  The selected
configuration is refit with the selection shards used solely for early
stopping, and the test shards are scored once at the end.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import random
import shutil
import statistics
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.datasets.dataset import load_chunk
from othello_research.datasets.move_mapping import build_mappings
from othello_research.evaluation.legal_moves import replay_game_legality
from othello_research.evaluation.thesis import (
    DEFAULT_REGISTRY,
    _model_from_checkpoint,
    resolve_case,
    resolve_data_dir,
    resolve_run_dir,
    validate_checkpoint_identity,
)
from othello_research.evaluation.unified import (
    UnifiedEvalConfig,
    assert_finite_module,
    build_evaluation_split,
    file_sha256,
    validate_encoder_surface,
)


STUDY_PROTOCOL_ID = "jepa_8x8_head_grouped_cv_v1"
HEAD_TYPES = ("linear", "mlp")


@dataclass(frozen=True)
class HeadTuningConfig:
    """Resource budget and statistical protocol for the tuning study."""

    board_size: int = 8
    seed: int = 42
    cv_folds: int = 5
    train_games: int = 12_000
    selection_games: int = 3_000
    test_games: int = 5_000
    positions_per_game: int = 6
    encoder_batch_size: int = 128
    head_batch_size: int = 4_096
    max_epochs: int = 15
    patience: int = 3
    min_delta: float = 1e-4
    gradient_clip: float = 1.0
    primary_metric: str = "legal_probability_mass"
    phase_bins: int = 4
    bootstrap_resamples: int = 1_000

    def __post_init__(self) -> None:
        if self.board_size != 8:
            raise ValueError("This study is intentionally restricted to 8x8")
        positive = {
            "cv_folds": self.cv_folds,
            "train_games": self.train_games,
            "selection_games": self.selection_games,
            "test_games": self.test_games,
            "positions_per_game": self.positions_per_game,
            "encoder_batch_size": self.encoder_batch_size,
            "head_batch_size": self.head_batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "phase_bins": self.phase_bins,
            "bootstrap_resamples": self.bootstrap_resamples,
        }
        invalid = {key: value for key, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Counts must be positive: {invalid}")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.primary_metric not in {
            "legal_probability_mass",
            "top1_legal",
        }:
            raise ValueError(
                "primary_metric must be legal_probability_mass or top1_legal"
            )
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    @classmethod
    def pilot(cls, *, seed: int = 42) -> "HeadTuningConfig":
        """Small end-to-end validation profile before the full thesis run."""
        return cls(
            seed=seed,
            cv_folds=3,
            train_games=2_000,
            selection_games=500,
            test_games=1_000,
            positions_per_game=4,
            max_epochs=8,
            patience=2,
            bootstrap_resamples=300,
        )


def default_parameter_grid(profile: str = "full") -> list[dict[str, Any]]:
    """Return a bounded, explicit grid for the two requested head families."""
    normalized = str(profile).strip().lower()
    if normalized not in {"pilot", "full"}:
        raise ValueError("profile must be 'pilot' or 'full'")

    if normalized == "pilot":
        linear_lrs = (3e-4, 1e-3)
        linear_regularizers = (
            ("none", 0.0, 0.0),
            ("l2", 1e-3, 0.0),
            ("l1", 0.0, 1e-5),
        )
        mlp_lrs = (3e-4, 1e-3)
        mlp_hidden = (256,)
        mlp_dropout = (0.0, 0.1)
    else:
        linear_lrs = (3e-4, 1e-3, 3e-3)
        linear_regularizers = (
            ("none", 0.0, 0.0),
            ("l2", 1e-4, 0.0),
            ("l2", 1e-3, 0.0),
            ("l1", 0.0, 1e-5),
        )
        mlp_lrs = (1e-4, 3e-4, 1e-3)
        mlp_hidden = (256, 512)
        mlp_dropout = (0.0, 0.1)

    grid: list[dict[str, Any]] = []
    for lr, regularizer in itertools.product(
        linear_lrs,
        linear_regularizers,
    ):
        regularization, weight_decay, l1_strength = regularizer
        strength = weight_decay if regularization == "l2" else l1_strength
        grid.append(
            {
                "id": (
                    f"linear__lr={lr:g}__reg={regularization}__"
                    f"strength={strength:g}"
                ),
                "head_type": "linear",
                "learning_rate": lr,
                "regularization": regularization,
                "weight_decay": weight_decay,
                "l1_strength": l1_strength,
            }
        )
    for lr, hidden_dim, dropout in itertools.product(
        mlp_lrs,
        mlp_hidden,
        mlp_dropout,
    ):
        grid.append(
            {
                "id": (
                    f"mlp__h={hidden_dim}__drop={dropout:g}__"
                    f"lr={lr:g}__wd=0.0001"
                ),
                "head_type": "mlp",
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "learning_rate": lr,
                "weight_decay": 1e-4,
            }
        )
    return grid


@dataclass
class CachedHeadDataset:
    """A compact CPU cache of frozen hidden states and Othello labels."""

    features: torch.Tensor
    targets: torch.Tensor
    legal_mask: torch.Tensor
    shard_ids: torch.Tensor
    game_ids: torch.Tensor
    phases: torch.Tensor
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape (samples, hidden_dim)")
        n_samples = int(self.features.shape[0])
        one_dimensional = {
            "targets": self.targets,
            "shard_ids": self.shard_ids,
            "game_ids": self.game_ids,
            "phases": self.phases,
        }
        for name, tensor in one_dimensional.items():
            if tensor.ndim != 1 or len(tensor) != n_samples:
                raise ValueError(f"{name} is not aligned with features")
        if self.legal_mask.ndim != 2 or len(self.legal_mask) != n_samples:
            raise ValueError("legal_mask is not aligned with features")
        if not n_samples:
            raise ValueError("feature cache is empty")
        vocab_size = int(self.legal_mask.shape[1])
        if int(self.targets.min()) < 0 or int(self.targets.max()) >= vocab_size:
            raise ValueError("targets fall outside the legal-mask vocabulary")
        if not self.legal_mask.gather(1, self.targets[:, None]).all():
            raise ValueError("at least one recorded target is not a legal move")
        if not torch.isfinite(self.features.float()).all():
            raise ValueError("feature cache contains non-finite values")
        if not torch.isfinite(self.phases).all():
            raise ValueError("phase values contain non-finite values")

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.legal_mask.shape[1])

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "cached_head_dataset_v1",
            "features": self.features,
            "targets": self.targets,
            "legal_mask": self.legal_mask,
            "shard_ids": self.shard_ids,
            "game_ids": self.game_ids,
            "phases": self.phases,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CachedHeadDataset":
        if payload.get("schema_version") != "cached_head_dataset_v1":
            raise ValueError("Unsupported cached-head dataset schema")
        result = cls(
            features=payload["features"],
            targets=payload["targets"],
            legal_mask=payload["legal_mask"],
            shard_ids=payload["shard_ids"],
            game_ids=payload["game_ids"],
            phases=payload["phases"],
            metadata=dict(payload["metadata"]),
        )
        result.validate()
        return result


@dataclass
class LoadedFrozenEncoder:
    architecture: str
    label: str
    encoder: nn.Module
    run_dir: Path
    data_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    encoder_parameters: int
    model_config: dict[str, Any]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_jepa_encoder(
    architecture: str,
    *,
    artifacts_root: str | Path,
    local_checkpoint_root: str | Path,
    device: str | torch.device,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> LoadedFrozenEncoder:
    """Load one canonical 8x8 JEPA context encoder without evaluator writes."""
    case = resolve_case(
        architecture,
        "jepa",
        8,
        registry_path=registry_path,
    )
    artifacts_root = Path(artifacts_root)
    run_dir = resolve_run_dir(case, artifacts_root)
    data_dir = resolve_data_dir(case, artifacts_root)
    source_checkpoint = run_dir / "final.pt"
    local_dir = Path(local_checkpoint_root) / case.key
    local_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoint = local_dir / "final.pt"
    if (
        not local_checkpoint.is_file()
        or local_checkpoint.stat().st_size != source_checkpoint.stat().st_size
    ):
        shutil.copy2(source_checkpoint, local_checkpoint)

    checkpoint_sha256 = file_sha256(local_checkpoint)
    checkpoint = torch.load(
        local_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint_identity(case, checkpoint)
    model, encoder, model_config = _model_from_checkpoint(
        case,
        checkpoint,
        load_weights=True,
    )
    del checkpoint
    encoder = encoder._orig_mod if hasattr(encoder, "_orig_mod") else encoder
    validate_encoder_surface(encoder, 8)
    assert_finite_module(encoder, f"{case.key}.encoder")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    config_payload = (
        asdict(model_config)
        if hasattr(model_config, "__dataclass_fields__")
        else dict(vars(model_config))
    )
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    # ``encoder`` remains alive through its explicit reference; the target
    # encoder and predictor owned by the full JEPA wrapper can now be released.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return LoadedFrozenEncoder(
        architecture=case.architecture,
        label=f"{case.architecture.title()}-JEPA 8x8",
        encoder=encoder,
        run_dir=run_dir,
        data_dir=data_dir,
        checkpoint_path=local_checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        encoder_parameters=encoder_parameters,
        model_config=config_payload,
    )


def phase_stratified_positions(
    input_length: int,
    positions_per_game: int,
) -> tuple[int, ...]:
    """Choose deterministic bin centres over all next-move input positions."""
    if input_length <= 0 or positions_per_game <= 0:
        return ()
    count = min(input_length, positions_per_game)
    positions = tuple(
        min(input_length - 1, ((2 * index + 1) * input_length) // (2 * count))
        for index in range(count)
    )
    if len(set(positions)) != count:
        raise AssertionError("phase sampler produced duplicate positions")
    return positions


def _encoder_hidden(
    encoder: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run the shared final-hidden-state surface used by the unified evaluator."""
    sequence_length = int(x.shape[1])
    positions = torch.arange(sequence_length, device=x.device)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if x.device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), context:
        hidden = encoder.drop(encoder.wte(x) + encoder.wpe(positions))
        for block in encoder.blocks:
            hidden = block(hidden)
        hidden = encoder.ln_f(hidden)
    if not torch.isfinite(hidden).all():
        raise RuntimeError("Frozen encoder produced non-finite hidden states")
    return hidden


def _game_quotas(total_games: int, n_shards: int) -> list[int]:
    if total_games < n_shards:
        raise ValueError(
            f"Need at least one game per shard: {total_games} < {n_shards}"
        )
    base, remainder = divmod(total_games, n_shards)
    return [base + (index < remainder) for index in range(n_shards)]


def _tokenize_raw_game(
    game: Sequence[int],
    raw_to_token: Sequence[int],
    board_size: int,
) -> list[int]:
    tokens: list[int] = []
    for raw_move in game:
        if not 0 <= int(raw_move) < board_size * board_size:
            raise ValueError(f"Raw move {raw_move} is outside the board")
        token = int(raw_to_token[int(raw_move)])
        if token < 0:
            raise ValueError(f"Raw move {raw_move} is a starting square")
        tokens.append(token)
    return tokens


def _cache_identity(
    *,
    checkpoint_sha256: str,
    split_name: str,
    chunks: Sequence[Path],
    n_games: int,
    positions_per_game: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "study_protocol": STUDY_PROTOCOL_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "split_name": split_name,
        "chunk_names": [path.name for path in chunks],
        "n_games_requested": n_games,
        "positions_per_game": positions_per_game,
        "seed": seed,
        "sampling": "uniform-games-within-shard__phase-bin-centres",
    }


def build_or_load_feature_cache(
    encoder: nn.Module,
    chunks: Sequence[str | Path],
    *,
    checkpoint_sha256: str,
    split_name: str,
    n_games: int,
    positions_per_game: int,
    board_size: int,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    cache_path: str | Path,
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> CachedHeadDataset:
    """Encode a deterministic, phase-balanced sample and persist it on CPU."""
    device = torch.device(device)
    chunk_paths = tuple(Path(path) for path in chunks)
    cache_path = Path(cache_path)
    identity = _cache_identity(
        checkpoint_sha256=checkpoint_sha256,
        split_name=split_name,
        chunks=chunk_paths,
        n_games=n_games,
        positions_per_game=positions_per_game,
        seed=seed,
    )
    if cache_path.is_file() and not force:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached = CachedHeadDataset.from_payload(payload)
        cached_identity = cached.metadata.get("identity")
        if cached_identity != identity:
            raise ValueError(
                f"Feature-cache identity mismatch at {cache_path}; use force=True"
            )
        progress(
            f"Reusing {split_name} cache: {cached.n_samples:,} positions "
            f"from {cached.metadata['n_games']:,} games"
        )
        return cached

    if board_size != 8:
        raise ValueError("Head-tuning feature extraction is restricted to 8x8")
    encoder = encoder._orig_mod if hasattr(encoder, "_orig_mod") else encoder
    validate_encoder_surface(encoder, board_size)
    encoder.eval()
    block_size = int(encoder.config.block_size)
    vocab_size = int(encoder.config.vocab_size)
    pad_token = board_size * board_size - 4
    raw_to_token, _ = build_mappings(board_size)
    quotas = _game_quotas(n_games, len(chunk_paths))

    feature_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    legal_parts: list[torch.Tensor] = []
    shard_parts: list[torch.Tensor] = []
    game_parts: list[torch.Tensor] = []
    phase_parts: list[torch.Tensor] = []
    global_game_id = 0
    selected_games_total = 0

    for shard_id, (chunk_path, quota) in enumerate(zip(chunk_paths, quotas)):
        games = load_chunk(str(chunk_path))
        eligible = [
            index
            for index, game in enumerate(games)
            if len(game[: block_size + 1]) >= 2
        ]
        if len(eligible) < quota:
            raise ValueError(
                f"{chunk_path.name} has {len(eligible)} eligible games, "
                f"but {quota} were requested"
            )
        rng = random.Random(seed + 1_000_003 * (shard_id + 1))
        chosen = sorted(rng.sample(eligible, quota))

        for batch_start in range(0, len(chosen), batch_size):
            batch_indices = chosen[batch_start : batch_start + batch_size]
            records: list[
                tuple[list[int], list[int], tuple[int, ...], int]
            ] = []
            max_input_length = 0
            for game_index in batch_indices:
                raw = list(games[game_index][: block_size + 1])
                tokens = _tokenize_raw_game(raw, raw_to_token, board_size)
                input_length = len(tokens) - 1
                sampled = phase_stratified_positions(
                    input_length,
                    positions_per_game,
                )
                records.append((raw, tokens, sampled, global_game_id))
                global_game_id += 1
                max_input_length = max(max_input_length, input_length)

            x = torch.full(
                (len(records), max_input_length),
                pad_token,
                dtype=torch.long,
            )
            for row, (_raw, tokens, _sampled, _game_id) in enumerate(records):
                x[row, : len(tokens) - 1] = torch.tensor(
                    tokens[:-1],
                    dtype=torch.long,
                )
            hidden = _encoder_hidden(
                encoder,
                x.to(device, non_blocking=device.type == "cuda"),
            )

            batch_features: list[torch.Tensor] = []
            batch_targets: list[int] = []
            batch_legal: list[torch.Tensor] = []
            batch_shards: list[int] = []
            batch_games: list[int] = []
            batch_phases: list[float] = []
            for row, (raw, tokens, sampled, game_id) in enumerate(records):
                legal_sets = replay_game_legality(raw, board_size)
                input_length = len(tokens) - 1
                for position in sampled:
                    target = tokens[position + 1]
                    legal = legal_sets[position + 1]
                    if target not in legal:
                        raise ValueError(
                            f"Corpus target is illegal in {chunk_path.name}, "
                            f"game={batch_indices[row]}, position={position + 1}"
                        )
                    mask = torch.zeros(vocab_size, dtype=torch.bool)
                    mask[list(sorted(legal))] = True
                    batch_features.append(hidden[row, position].detach().cpu())
                    batch_targets.append(target)
                    batch_legal.append(mask)
                    batch_shards.append(shard_id)
                    batch_games.append(game_id)
                    batch_phases.append((position + 1) / input_length)

            feature_parts.append(torch.stack(batch_features).to(torch.float16))
            target_parts.append(torch.tensor(batch_targets, dtype=torch.long))
            legal_parts.append(torch.stack(batch_legal))
            shard_parts.append(torch.tensor(batch_shards, dtype=torch.long))
            game_parts.append(torch.tensor(batch_games, dtype=torch.long))
            phase_parts.append(torch.tensor(batch_phases, dtype=torch.float32))
            selected_games_total += len(records)
            del x, hidden

        progress(
            f"{split_name}: shard {shard_id + 1}/{len(chunk_paths)} "
            f"games={selected_games_total:,}"
        )
        del games
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dataset = CachedHeadDataset(
        features=torch.cat(feature_parts),
        targets=torch.cat(target_parts),
        legal_mask=torch.cat(legal_parts),
        shard_ids=torch.cat(shard_parts),
        game_ids=torch.cat(game_parts),
        phases=torch.cat(phase_parts),
        metadata={
            "identity": identity,
            "n_games": selected_games_total,
            "n_samples": sum(len(part) for part in target_parts),
            "hidden_dim": int(feature_parts[0].shape[1]),
            "vocab_size": vocab_size,
            "feature_dtype": "float16",
        },
    )
    dataset.validate()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(dataset.to_payload(), temporary)
    temporary.replace(cache_path)
    progress(f"Saved {split_name} feature cache: {cache_path}")
    return dataset


def make_group_kfold_splits(
    groups: torch.Tensor,
    n_splits: int,
    *,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Balanced deterministic GroupKFold with no group crossing a fold."""
    groups = groups.detach().cpu().long().flatten()
    unique_groups, counts = torch.unique(groups, return_counts=True)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups, found {len(unique_groups)}"
        )
    rng = random.Random(seed)
    group_count_pairs = [
        (int(group), int(count))
        for group, count in zip(unique_groups.tolist(), counts.tolist())
    ]
    rng.shuffle(group_count_pairs)
    group_count_pairs.sort(key=lambda item: item[1], reverse=True)
    fold_groups: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for group, count in group_count_pairs:
        fold = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold].append(group)
        fold_sizes[fold] += count

    all_indices = torch.arange(len(groups))
    splits: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen_validation = torch.zeros(len(groups), dtype=torch.bool)
    for validation_groups in fold_groups:
        validation_mask = torch.zeros(len(groups), dtype=torch.bool)
        for group in validation_groups:
            validation_mask |= groups == group
        validation_indices = all_indices[validation_mask]
        training_indices = all_indices[~validation_mask]
        if set(groups[training_indices].tolist()) & set(
            groups[validation_indices].tolist()
        ):
            raise AssertionError("A group crossed the train/validation boundary")
        seen_validation[validation_indices] = True
        splits.append((training_indices, validation_indices))
    if not seen_validation.all():
        raise AssertionError("Some samples never appeared in a validation fold")
    return splits


class _TuningMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, vocab_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_head(
    hidden_dim: int,
    vocab_size: int,
    params: Mapping[str, Any],
) -> nn.Module:
    head_type = str(params["head_type"])
    if head_type == "linear":
        return nn.Linear(hidden_dim, vocab_size)
    if head_type == "mlp":
        return _TuningMLP(
            hidden_dim,
            vocab_size,
            int(params["hidden_dim"]),
            float(params["dropout"]),
        )
    raise ValueError(f"Unsupported head_type: {head_type}")


def _l1_penalty(
    head: nn.Module,
    params: Mapping[str, Any],
) -> torch.Tensor:
    """Return an explicit Lasso penalty over weights, never over biases."""
    strength = float(params.get("l1_strength", 0.0))
    regularization = str(params.get("regularization", "none")).lower()
    if regularization != "l1" or strength <= 0:
        parameter = next(head.parameters())
        return parameter.new_zeros(())
    penalty = sum(
        tensor.abs().sum()
        for name, tensor in head.named_parameters()
        if name.endswith("weight")
    )
    return penalty * strength


def _bootstrap_mean_ci(
    values: torch.Tensor,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, float]:
    values = values.detach().cpu().float().flatten()
    generator = torch.Generator().manual_seed(seed)
    means: list[torch.Tensor] = []
    remaining = n_resamples
    while remaining:
        count = min(100, remaining)
        indices = torch.randint(
            len(values),
            (count, len(values)),
            generator=generator,
        )
        means.append(values[indices].mean(dim=1))
        remaining -= count
    samples = torch.cat(means)
    quantiles = torch.quantile(samples, torch.tensor([0.025, 0.975]))
    return {
        "low": float(quantiles[0]),
        "high": float(quantiles[1]),
    }


def evaluate_cached_head(
    head: nn.Module,
    dataset: CachedHeadDataset,
    indices: torch.Tensor | None,
    *,
    batch_size: int,
    device: str | torch.device,
    detailed: bool = False,
    phase_bins: int = 4,
    bootstrap_resamples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate target loss plus rule-aware legal-move metrics."""
    device = torch.device(device)
    head.eval()
    if indices is None:
        indices = torch.arange(dataset.n_samples)
    indices = indices.detach().cpu().long()
    loss_sum = 0.0
    legal_mass_sum = 0.0
    legal_top1_sum = 0
    target_top1_sum = 0
    sample_count = 0
    detailed_mass: list[torch.Tensor] = []
    detailed_legal: list[torch.Tensor] = []
    detailed_target: list[torch.Tensor] = []
    detailed_games: list[torch.Tensor] = []
    detailed_phases: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            features = dataset.features[batch_indices].to(
                device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            targets = dataset.targets[batch_indices].to(
                device,
                non_blocking=device.type == "cuda",
            )
            legal_mask = dataset.legal_mask[batch_indices].to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = head(features)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Head produced non-finite logits")
            losses = F.cross_entropy(logits, targets, reduction="none")
            probabilities = torch.softmax(logits, dim=-1)
            legal_mass = (probabilities * legal_mask).sum(dim=-1)
            prediction = logits.argmax(dim=-1)
            top1_legal = legal_mask.gather(1, prediction[:, None]).squeeze(1)
            target_top1 = prediction == targets
            batch_count = len(batch_indices)
            loss_sum += float(losses.sum().cpu())
            legal_mass_sum += float(legal_mass.sum().cpu())
            legal_top1_sum += int(top1_legal.sum().cpu())
            target_top1_sum += int(target_top1.sum().cpu())
            sample_count += batch_count
            if detailed:
                detailed_mass.append(legal_mass.cpu())
                detailed_legal.append(top1_legal.float().cpu())
                detailed_target.append(target_top1.float().cpu())
                detailed_games.append(dataset.game_ids[batch_indices].cpu())
                detailed_phases.append(dataset.phases[batch_indices].cpu())

    if sample_count <= 0:
        raise ValueError("Cannot evaluate an empty index set")
    result: dict[str, Any] = {
        "n_positions": sample_count,
        "cross_entropy": loss_sum / sample_count,
        "legal_probability_mass": legal_mass_sum / sample_count,
        "top1_legal": legal_top1_sum / sample_count,
        "target_top1_accuracy": target_top1_sum / sample_count,
    }
    if not detailed:
        return result

    mass = torch.cat(detailed_mass)
    legal = torch.cat(detailed_legal)
    target = torch.cat(detailed_target)
    game_ids = torch.cat(detailed_games)
    phases = torch.cat(detailed_phases)
    unique_games, inverse = torch.unique(game_ids, sorted=True, return_inverse=True)
    game_counts = torch.zeros(len(unique_games), dtype=torch.float32)
    game_mass = torch.zeros(len(unique_games), dtype=torch.float32)
    game_legal = torch.zeros(len(unique_games), dtype=torch.float32)
    game_target = torch.zeros(len(unique_games), dtype=torch.float32)
    game_counts.scatter_add_(0, inverse, torch.ones_like(mass))
    game_mass.scatter_add_(0, inverse, mass)
    game_legal.scatter_add_(0, inverse, legal)
    game_target.scatter_add_(0, inverse, target)
    game_mass /= game_counts
    game_legal /= game_counts
    game_target /= game_counts

    result["n_games"] = len(unique_games)
    result["bootstrap_ci_95_per_game"] = {
        "legal_probability_mass": _bootstrap_mean_ci(
            game_mass,
            seed=seed,
            n_resamples=bootstrap_resamples,
        ),
        "top1_legal": _bootstrap_mean_ci(
            game_legal,
            seed=seed + 1,
            n_resamples=bootstrap_resamples,
        ),
        "target_top1_accuracy": _bootstrap_mean_ci(
            game_target,
            seed=seed + 2,
            n_resamples=bootstrap_resamples,
        ),
    }
    phase_rows: list[dict[str, Any]] = []
    phase_index = torch.clamp((phases * phase_bins).long(), max=phase_bins - 1)
    for phase_bin in range(phase_bins):
        mask = phase_index == phase_bin
        phase_rows.append(
            {
                "phase_bin": phase_bin,
                "phase_start": phase_bin / phase_bins,
                "phase_end": (phase_bin + 1) / phase_bins,
                "n_positions": int(mask.sum()),
                "legal_probability_mass": float(mass[mask].mean()),
                "top1_legal": float(legal[mask].mean()),
                "target_top1_accuracy": float(target[mask].mean()),
            }
        )
    result["phase_metrics"] = phase_rows
    # Internal vectors enable paired linear-vs-MLP uncertainty.  The caller
    # removes them before writing the public JSON result.
    result["_per_game_legal_probability_mass"] = game_mass
    result["_per_game_top1_legal"] = game_legal
    return result


def fit_cached_head(
    dataset: CachedHeadDataset,
    train_indices: torch.Tensor,
    validation_dataset: CachedHeadDataset,
    validation_indices: torch.Tensor | None,
    params: Mapping[str, Any],
    *,
    config: HeadTuningConfig,
    device: str | torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    """Fit one readout with early stopping on the requested validation set."""
    device = torch.device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = build_head(dataset.hidden_dim, dataset.vocab_size, params).to(device)
    head.float()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    train_indices = train_indices.detach().cpu().long()
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, config.max_epochs + 1):
        head.train()
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator)
        ]
        objective_sum = 0.0
        cross_entropy_sum = 0.0
        regularization_sum = 0.0
        sample_count = 0
        for start in range(0, len(permutation), config.head_batch_size):
            batch_indices = permutation[start : start + config.head_batch_size]
            features = dataset.features[batch_indices].to(
                device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            targets = dataset.targets[batch_indices].to(
                device,
                non_blocking=device.type == "cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            cross_entropy = F.cross_entropy(logits, targets)
            regularization_penalty = _l1_penalty(head, params)
            loss = cross_entropy + regularization_penalty
            if not torch.isfinite(loss):
                raise RuntimeError("Head training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                config.gradient_clip,
            )
            optimizer.step()
            batch_count = len(batch_indices)
            objective_sum += float(loss.detach().cpu()) * batch_count
            cross_entropy_sum += (
                float(cross_entropy.detach().cpu()) * batch_count
            )
            regularization_sum += (
                float(regularization_penalty.detach().cpu()) * batch_count
            )
            sample_count += batch_count

        validation = evaluate_cached_head(
            head,
            validation_dataset,
            validation_indices,
            batch_size=config.head_batch_size,
            device=device,
        )
        metric = float(validation[config.primary_metric])
        improved = metric > best_metric + config.min_delta
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_objective": objective_sum / max(1, sample_count),
                "train_cross_entropy": (
                    cross_entropy_sum / max(1, sample_count)
                ),
                "train_regularization_penalty": (
                    regularization_sum / max(1, sample_count)
                ),
                "validation": validation,
                "improved": improved,
            }
        )
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Head fitting did not produce a valid checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    best_validation = evaluate_cached_head(
        head,
        validation_dataset,
        validation_indices,
        batch_size=config.head_batch_size,
        device=device,
    )
    return head, {
        "params": dict(params),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation": best_validation,
        "history": history,
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
    }


def _search_identity(
    dataset: CachedHeadDataset,
    grid: Sequence[Mapping[str, Any]],
    config: HeadTuningConfig,
) -> dict[str, Any]:
    return {
        "protocol": STUDY_PROTOCOL_ID,
        "dataset_identity": dataset.metadata["identity"],
        "dataset_shape": list(dataset.features.shape),
        "grid": [dict(row) for row in grid],
        "cv_folds": config.cv_folds,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "head_batch_size": config.head_batch_size,
        "primary_metric": config.primary_metric,
    }


def grouped_grid_search(
    dataset: CachedHeadDataset,
    grid: Sequence[Mapping[str, Any]],
    *,
    config: HeadTuningConfig,
    device: str | torch.device,
    resume_path: str | Path | None = None,
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run a resumable shard-grouped grid search over cached features."""
    dataset.validate()
    grid = [dict(row) for row in grid]
    for head_type in HEAD_TYPES:
        if not any(row.get("head_type") == head_type for row in grid):
            raise ValueError(f"Grid has no {head_type} configurations")
    identity = _search_identity(dataset, grid, config)
    resume_file = Path(resume_path) if resume_path is not None else None
    completed: dict[str, dict[str, Any]] = {}
    if resume_file is not None and resume_file.is_file() and not force:
        payload = json.loads(resume_file.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError(
                f"Grid-search identity mismatch at {resume_file}; use force=True"
            )
        completed = {
            row["params"]["id"]: row
            for rows in payload.get("results_by_head", {}).values()
            for row in rows
        }
        progress(f"Resuming grid search with {len(completed)} completed configs")

    splits = make_group_kfold_splits(
        dataset.shard_ids,
        config.cv_folds,
        seed=config.seed,
    )
    for config_index, params in enumerate(grid):
        config_id = str(params["id"])
        if config_id in completed:
            continue
        fold_rows: list[dict[str, Any]] = []
        for fold_index, (train_indices, validation_indices) in enumerate(splits):
            run_seed = config.seed + 10_000 * config_index + fold_index
            head, fit = fit_cached_head(
                dataset,
                train_indices,
                dataset,
                validation_indices,
                params,
                config=config,
                device=device,
                seed=run_seed,
            )
            fold_rows.append(
                {
                    "fold": fold_index,
                    "train_positions": len(train_indices),
                    "validation_positions": len(validation_indices),
                    "validation_groups": sorted(
                        set(dataset.shard_ids[validation_indices].tolist())
                    ),
                    "best_epoch": fit["best_epoch"],
                    "epochs_ran": fit["epochs_ran"],
                    "metrics": fit["best_validation"],
                }
            )
            progress(
                f"{config_id} fold={fold_index + 1}/{len(splits)} "
                f"{config.primary_metric}="
                f"{fit['best_validation'][config.primary_metric]:.5f}"
            )
            del head
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metric_values = [
            float(row["metrics"][config.primary_metric]) for row in fold_rows
        ]
        loss_values = [
            float(row["metrics"]["cross_entropy"]) for row in fold_rows
        ]
        completed[config_id] = {
            "params": params,
            "folds": fold_rows,
            "cv_mean": statistics.mean(metric_values),
            "cv_std": (
                statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            ),
            "cv_cross_entropy_mean": statistics.mean(loss_values),
            "median_best_epoch": int(
                round(statistics.median(row["best_epoch"] for row in fold_rows))
            ),
        }
        if resume_file is not None:
            interim_by_head = {
                head_type: [
                    row
                    for row in completed.values()
                    if row["params"]["head_type"] == head_type
                ]
                for head_type in HEAD_TYPES
            }
            _atomic_json(
                resume_file,
                {
                    "identity": identity,
                    "results_by_head": interim_by_head,
                    "complete": len(completed) == len(grid),
                },
            )

    results_by_head: dict[str, list[dict[str, Any]]] = {}
    best_by_head: dict[str, dict[str, Any]] = {}
    for head_type in HEAD_TYPES:
        rows = [
            row
            for row in completed.values()
            if row["params"]["head_type"] == head_type
        ]
        rows.sort(
            key=lambda row: (
                -float(row["cv_mean"]),
                float(row["cv_std"]),
                float(row["cv_cross_entropy_mean"]),
                str(row["params"]["id"]),
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        results_by_head[head_type] = rows
        best_by_head[head_type] = rows[0]

    result = {
        "identity": identity,
        "results_by_head": results_by_head,
        "best_by_head": best_by_head,
        "complete": True,
    }
    if resume_file is not None:
        _atomic_json(resume_file, result)
    return result


def _public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not str(key).startswith("_")
    }


def _paired_delta(
    mlp_values: torch.Tensor,
    linear_values: torch.Tensor,
    *,
    config: HeadTuningConfig,
    seed: int,
) -> dict[str, Any]:
    if len(mlp_values) != len(linear_values):
        raise ValueError("Paired head metrics have different game counts")
    delta = mlp_values.float() - linear_values.float()
    ci = _bootstrap_mean_ci(
        delta,
        seed=seed,
        n_resamples=config.bootstrap_resamples,
    )
    mean = float(delta.mean())
    if ci["low"] > 0:
        interpretation = "MLP advantage"
    elif ci["high"] < 0:
        interpretation = "Linear advantage"
    else:
        interpretation = "No resolved advantage"
    return {
        "mlp_minus_linear": mean,
        "bootstrap_ci_95_per_game": ci,
        "interpretation": interpretation,
    }


def fit_selected_heads(
    train_dataset: CachedHeadDataset,
    selection_dataset: CachedHeadDataset,
    test_dataset: CachedHeadDataset,
    search: Mapping[str, Any],
    *,
    config: HeadTuningConfig,
    device: str | torch.device,
    output_dir: str | Path,
    case_metadata: Mapping[str, Any],
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Refit the CV winners and evaluate each head once on the test cache."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final: dict[str, Any] = {}
    internal_test: dict[str, dict[str, Any]] = {}
    all_train = torch.arange(train_dataset.n_samples)
    for head_index, head_type in enumerate(HEAD_TYPES):
        best = search["best_by_head"][head_type]
        params = best["params"]
        progress(f"Final refit: {case_metadata['label']} / {head_type}")
        head, fit = fit_cached_head(
            train_dataset,
            all_train,
            selection_dataset,
            None,
            params,
            config=config,
            device=device,
            seed=config.seed + 90_000 + head_index,
        )
        selection_metrics = evaluate_cached_head(
            head,
            selection_dataset,
            None,
            batch_size=config.head_batch_size,
            device=device,
            detailed=True,
            phase_bins=config.phase_bins,
            bootstrap_resamples=config.bootstrap_resamples,
            seed=config.seed + 100 + head_index,
        )
        # This is deliberately the only test evaluation in the tuning path.
        test_metrics = evaluate_cached_head(
            head,
            test_dataset,
            None,
            batch_size=config.head_batch_size,
            device=device,
            detailed=True,
            phase_bins=config.phase_bins,
            bootstrap_resamples=config.bootstrap_resamples,
            seed=config.seed + 200 + head_index,
        )
        internal_test[head_type] = test_metrics
        checkpoint_path = output_dir / f"best_{head_type}_head.pt"
        torch.save(
            {
                "schema_version": "tuned_head_checkpoint_v1",
                "study_protocol": STUDY_PROTOCOL_ID,
                "case": dict(case_metadata),
                "params": dict(params),
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in head.state_dict().items()
                },
                "hidden_dim": train_dataset.hidden_dim,
                "vocab_size": train_dataset.vocab_size,
                "selection_best_epoch": fit["best_epoch"],
            },
            checkpoint_path,
        )
        final[head_type] = {
            "params": dict(params),
            "cv": {
                "mean": best["cv_mean"],
                "std": best["cv_std"],
                "cross_entropy_mean": best["cv_cross_entropy_mean"],
                "rank": best["rank"],
            },
            "best_epoch": fit["best_epoch"],
            "epochs_ran": fit["epochs_ran"],
            "head_parameters": fit["head_parameters"],
            "selection": _public_metrics(selection_metrics),
            "test": _public_metrics(test_metrics),
            "checkpoint_path": str(checkpoint_path),
            "history": fit["history"],
        }
        del head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = {
        "legal_probability_mass": _paired_delta(
            internal_test["mlp"]["_per_game_legal_probability_mass"],
            internal_test["linear"]["_per_game_legal_probability_mass"],
            config=config,
            seed=config.seed + 300,
        ),
        "top1_legal": _paired_delta(
            internal_test["mlp"]["_per_game_top1_legal"],
            internal_test["linear"]["_per_game_top1_legal"],
            config=config,
            seed=config.seed + 301,
        ),
    }
    return {
        "case": dict(case_metadata),
        "final": final,
        "paired_comparison": comparison,
    }


def _cache_filename(
    architecture: str,
    split_name: str,
    n_games: int,
    positions_per_game: int,
    checkpoint_sha256: str,
) -> str:
    return (
        f"{architecture}_jepa_b8__{split_name}__g{n_games}__"
        f"p{positions_per_game}__{checkpoint_sha256[:12]}.pt"
    )


def run_head_tuning_case(
    architecture: str,
    *,
    artifacts_root: str | Path,
    local_checkpoint_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    config: HeadTuningConfig,
    grid: Sequence[Mapping[str, Any]],
    device: str | torch.device = "cuda:0",
    registry_path: str | Path = DEFAULT_REGISTRY,
    rebuild_feature_cache: bool = False,
    force_search: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Execute one architecture arm of the 8x8 JEPA head study."""
    loaded = load_frozen_jepa_encoder(
        architecture,
        artifacts_root=artifacts_root,
        local_checkpoint_root=local_checkpoint_root,
        device=device,
        registry_path=registry_path,
    )
    protocol = UnifiedEvalConfig(board_size=8)
    chunks = sorted(loaded.data_dir.glob("*.pickle"), key=lambda path: path.name)
    split = build_evaluation_split(chunks, protocol)
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_specs = {
        "downstream_train": (split.downstream_train, config.train_games),
        "selection": (split.selection, config.selection_games),
        "test": (split.test, config.test_games),
    }
    caches: dict[str, CachedHeadDataset] = {}
    for split_index, (split_name, (split_chunks, n_games)) in enumerate(
        split_specs.items()
    ):
        cache_path = cache_root / _cache_filename(
            loaded.architecture,
            split_name,
            n_games,
            config.positions_per_game,
            loaded.checkpoint_sha256,
        )
        caches[split_name] = build_or_load_feature_cache(
            loaded.encoder,
            split_chunks,
            checkpoint_sha256=loaded.checkpoint_sha256,
            split_name=split_name,
            n_games=n_games,
            positions_per_game=config.positions_per_game,
            board_size=8,
            batch_size=config.encoder_batch_size,
            seed=config.seed + split_index,
            device=device,
            cache_path=cache_path,
            force=rebuild_feature_cache,
            progress=progress,
        )

    case_metadata = {
        "architecture": loaded.architecture,
        "objective": "jepa",
        "board_size": 8,
        "label": loaded.label,
        "run_dir": str(loaded.run_dir),
        "checkpoint_path": str(loaded.checkpoint_path),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "encoder_parameters": loaded.encoder_parameters,
        "model_config": loaded.model_config,
        "split_manifest_hash": split.manifest_hash,
    }
    del loaded.encoder, loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    search = grouped_grid_search(
        caches["downstream_train"],
        grid,
        config=config,
        device=device,
        resume_path=output_dir / "cv_search.json",
        force=force_search,
        progress=progress,
    )
    result = fit_selected_heads(
        caches["downstream_train"],
        caches["selection"],
        caches["test"],
        search,
        config=config,
        device=device,
        output_dir=output_dir,
        case_metadata=case_metadata,
        progress=progress,
    )
    result["search"] = search
    result["cache_metadata"] = {
        name: cache.metadata for name, cache in caches.items()
    }
    result["protocol"] = {
        "id": STUDY_PROTOCOL_ID,
        "config": asdict(config),
        "grid_digest": _json_digest([dict(row) for row in grid]),
        "selection_rule": (
            f"max mean shard-grouped CV {config.primary_metric}; "
            "tie-break lower CV std then lower cross-entropy"
        ),
        "test_access": "once per selected final head",
    }
    _atomic_json(output_dir / "case_results.json", result)
    return result


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _make_study_figures(
    cases: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"transformer": "#3568B8", "mamba": "#E57A3A"}
    markers = {"linear": "o", "mlp": "s"}
    paths: dict[str, Path] = {}

    # CV mean versus instability makes the mean/std trade-off visible.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for case in cases.values():
        architecture = case["case"]["architecture"]
        for head_type in HEAD_TYPES:
            rows = case["search"]["results_by_head"][head_type]
            ax.scatter(
                [row["cv_mean"] for row in rows],
                [row["cv_std"] for row in rows],
                s=75,
                alpha=0.75,
                color=colors[architecture],
                marker=markers[head_type],
                label=f"{architecture.title()} / {head_type.upper()}",
            )
            best = rows[0]
            ax.annotate(
                "winner",
                (best["cv_mean"], best["cv_std"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    ax.set_xlabel("Mean grouped-CV legal probability mass")
    ax.set_ylabel("CV standard deviation (lower is steadier)")
    ax.set_title("Head search: performance–stability map")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "cv_performance_stability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["stability"] = path

    # Paired dumbbell: direct linear-versus-MLP final-test comparison.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = []
    for row_index, case in enumerate(cases.values()):
        architecture = case["case"]["architecture"]
        linear = case["final"]["linear"]["test"]["legal_probability_mass"]
        mlp = case["final"]["mlp"]["test"]["legal_probability_mass"]
        ax.plot([linear, mlp], [row_index, row_index], color="#9AA0A6", lw=3)
        ax.scatter(linear, row_index, s=110, marker="o", color=colors[architecture])
        ax.scatter(mlp, row_index, s=110, marker="s", color=colors[architecture])
        ax.text(linear, row_index + 0.14, "Linear", ha="center", fontsize=8)
        ax.text(mlp, row_index - 0.20, "MLP", ha="center", fontsize=8)
        labels.append(case["case"]["label"])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Final-test legal probability mass")
    ax.set_title("Tuned head capacity: paired final-test result")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "final_head_dumbbell.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["dumbbell"] = path

    # Phase matrix shows whether a head advantage is localized in game time.
    phase_rows = []
    phase_labels = []
    for case in cases.values():
        for head_type in HEAD_TYPES:
            phase_rows.append(
                [
                    row["legal_probability_mass"]
                    for row in case["final"][head_type]["test"]["phase_metrics"]
                ]
            )
            phase_labels.append(f"{case['case']['architecture'].title()} / {head_type}")
    matrix = np.asarray(phase_rows, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < matrix.mean() else "black",
                fontsize=8,
            )
    ax.set_yticks(range(len(phase_labels)), phase_labels)
    ax.set_xticks(
        range(matrix.shape[1]),
        [f"P{index + 1}" for index in range(matrix.shape[1])],
    )
    ax.set_xlabel("Normalized game phase")
    ax.set_title("Where does each tuned head help?")
    fig.colorbar(image, ax=ax, label="Legal probability mass")
    fig.tight_layout()
    path = figure_dir / "phase_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["phase"] = path

    # Final refit learning curves expose unstable or prematurely stopped fits.
    fig, axes = plt.subplots(
        len(cases),
        1,
        figsize=(8.5, 3.4 * len(cases)),
        squeeze=False,
    )
    for axis, case in zip(axes[:, 0], cases.values()):
        for head_type, linestyle in (("linear", "-"), ("mlp", "--")):
            history = case["final"][head_type]["history"]
            axis.plot(
                [row["epoch"] for row in history],
                [
                    row["validation"]["legal_probability_mass"]
                    for row in history
                ],
                linestyle,
                marker=markers[head_type],
                label=head_type.upper(),
            )
        axis.set_title(case["case"]["label"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Selection legal mass")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Final refit / early-stopping trajectories", y=1.01)
    fig.tight_layout()
    path = figure_dir / "final_learning_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["learning"] = path
    return paths


def write_head_tuning_report(
    cases: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: str | Path,
    config: HeadTuningConfig,
    grid: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Path]]:
    """Write the combined Markdown dossier and its non-lineplot visual suite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = _make_study_figures(cases, output_dir)
    final_rows = []
    verdict_rows = []
    leaderboard_rows = []
    for case in cases.values():
        label = case["case"]["label"]
        for head_type in HEAD_TYPES:
            final = case["final"][head_type]
            test = final["test"]
            ci = test["bootstrap_ci_95_per_game"]["legal_probability_mass"]
            final_rows.append(
                [
                    label,
                    head_type.upper(),
                    final["params"]["id"],
                    f"{_fmt(final['cv']['mean'])} ± {_fmt(final['cv']['std'])}",
                    _fmt(final["selection"]["legal_probability_mass"]),
                    f"{_fmt(test['legal_probability_mass'])} "
                    f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]",
                    _fmt(test["top1_legal"]),
                    _fmt(test["target_top1_accuracy"]),
                    f"{final['head_parameters']:,}",
                    final["best_epoch"],
                ]
            )
            for row in case["search"]["results_by_head"][head_type][:3]:
                leaderboard_rows.append(
                    [
                        label,
                        head_type.upper(),
                        row["rank"],
                        row["params"]["id"],
                        _fmt(row["cv_mean"]),
                        _fmt(row["cv_std"]),
                        _fmt(row["cv_cross_entropy_mean"]),
                    ]
                )
        delta = case["paired_comparison"]["legal_probability_mass"]
        ci = delta["bootstrap_ci_95_per_game"]
        verdict_rows.append(
            [
                label,
                _fmt(delta["mlp_minus_linear"], 5),
                f"[{_fmt(ci['low'], 5)}, {_fmt(ci['high'], 5)}]",
                delta["interpretation"],
            ]
        )

    linear_count = sum(row["head_type"] == "linear" for row in grid)
    mlp_count = sum(row["head_type"] == "mlp" for row in grid)
    source = [
        "# 8×8 JEPA — Linear vs MLP Head Tuning",
        "",
        f"**Protocol:** `{STUDY_PROTOCOL_ID}`  ",
        f"**Cases:** {', '.join(case['case']['label'] for case in cases.values())}  ",
        f"**Grid:** {linear_count} Linear + {mlp_count} MLP configurations per encoder  ",
        f"**Cross-validation:** {config.cv_folds}-fold, grouped by downstream shard  ",
        "**Encoder policy:** frozen throughout; only head parameters are updated",
        "",
        "**Linear regularization:** unregularized control, grid-selected "
        "L2/Ridge through AdamW weight decay, and L1/Lasso added explicitly "
        "over weight matrices only.",
        "",
        "## Executive decision",
        "",
        _md_table(
            [
                "Encoder",
                "MLP − Linear legal mass",
                "Paired 95% CI",
                "Reading",
            ],
            verdict_rows,
        ),
        "",
        "A small raw delta is not treated as evidence of non-linearity unless its "
        "paired, per-game interval excludes zero. Parameter count and stability "
        "remain part of the decision: an unresolved MLP gain favors the simpler "
        "linear head.",
        "",
        "![Paired final head comparison](figures/final_head_dumbbell.png)",
        "",
        "## Final held-out results",
        "",
        _md_table(
            [
                "Encoder",
                "Head",
                "Selected config",
                "CV legal mass",
                "Selection legal mass",
                "Test legal mass [95% CI]",
                "Test top-1 legal",
                "Exact next move",
                "Head params",
                "Best epoch",
            ],
            final_rows,
        ),
        "",
        "## Search robustness",
        "",
        "The ranking metric is mean legal probability mass across shard-held-out "
        "folds. Ties prefer lower fold-to-fold variation, then lower "
        "cross-entropy. Linear candidates jointly tune learning rate and "
        "regularization; therefore a simpler sparse or shrinkage-controlled "
        "readout can win without giving the MLP an unfair capacity advantage.",
        "",
        _md_table(
            [
                "Encoder",
                "Head",
                "Rank",
                "Configuration",
                "CV mean",
                "CV std",
                "CV cross-entropy",
            ],
            leaderboard_rows,
        ),
        "",
        "![CV performance stability map](figures/cv_performance_stability.png)",
        "",
        "## Game-phase diagnostic",
        "",
        "Phase-balanced sampling prevents long games and late-game positions from "
        "dominating the fit. The matrix checks whether an apparent head advantage "
        "is global or confined to one part of the game.",
        "",
        "![Phase matrix](figures/phase_matrix.png)",
        "",
        "## Optimization diagnostic",
        "",
        "![Final learning curves](figures/final_learning_curves.png)",
        "",
        "## Leakage and interpretation guardrails",
        "",
        "- CV uses only shards 200–219 and keeps every shard wholly inside one fold.",
        "- Shards 220–228 are used only to early-stop the final refit.",
        "- Shards 229–237 are evaluated once after hyper-parameter selection.",
        "- Games are sampled independently within each shard; positions from the "
        "same game never cross a CV boundary because grouping is stricter at shard level.",
        "- Encoders are frozen and embeddings are cached before any head search.",
        "- Every fold starts from a fresh head initialization; saved Common "
        "Evaluation heads are not warm-started, which would invalidate CV.",
        "- This study tunes next-move readouts; it does not fine-tune the JEPA "
        "encoder or replace the Common Evaluation headline protocol.",
        "",
        "## Reproducibility",
        "",
        "The machine-readable files `study_results.json`, per-case "
        "`cv_search.json`, cached features, and the selected `.pt` head "
        "checkpoints contain the exact configurations and identities.",
    ]
    report_path = output_dir / "head_tuning_report.md"
    report_path.write_text("\n".join(source) + "\n", encoding="utf-8")
    study_payload = {
        "protocol": STUDY_PROTOCOL_ID,
        "config": asdict(config),
        "grid": [dict(row) for row in grid],
        "cases": dict(cases),
        "report_path": str(report_path),
        "figures": {name: str(path) for name, path in figures.items()},
    }
    _atomic_json(output_dir / "study_results.json", study_payload)
    return report_path, figures


__all__ = [
    "STUDY_PROTOCOL_ID",
    "HEAD_TYPES",
    "CachedHeadDataset",
    "HeadTuningConfig",
    "LoadedFrozenEncoder",
    "build_head",
    "build_or_load_feature_cache",
    "default_parameter_grid",
    "evaluate_cached_head",
    "fit_cached_head",
    "fit_selected_heads",
    "grouped_grid_search",
    "load_frozen_jepa_encoder",
    "make_group_kfold_splits",
    "phase_stratified_positions",
    "run_head_tuning_case",
    "write_head_tuning_report",
]
