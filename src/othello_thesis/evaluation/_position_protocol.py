"""Deterministic sample and manifest contracts for position probes."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch

from othello_thesis.data.chunk_dataset import load_chunk

GameRecord: TypeAlias = tuple[str | Path, int]
ChunkLoader: TypeAlias = Callable[[str], list[list[int]]]
ProbeType: TypeAlias = Literal["linear", "mlp"]

ABSOLUTE_CLASS_NAMES = ("black", "white", "empty")
RELATIVE_CLASS_NAMES = ("mine", "yours", "empty")
PROTOCOL_TAG = "position_probe_v1"


def _canonical_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _record_key(record: GameRecord) -> tuple[str, int]:
    path, game_index = record
    return _canonical_path(path), int(game_index)


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        count = int((~torch.isfinite(tensor)).sum().item())
        raise RuntimeError(
            f"{name} contains {count} non-finite values "
            f"(shape={tuple(tensor.shape)}, dtype={tensor.dtype})."
        )


@dataclass(frozen=True)
class PositionSample:
    """One fixed prefix position with complete corpus provenance."""

    chunk_path: str
    game_index: int
    position: int
    phase_bin: int
    normalized_phase: float
    game_length: int
    max_eligible_position: int
    game_sha256: str = ""

    @property
    def record_key(self) -> tuple[str, int]:
        return self.chunk_path, self.game_index

    @property
    def sample_id(self) -> str:
        return f"{self.chunk_path}::{self.game_index}::t={self.position}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PositionProbeSplits:
    """Fixed and record-disjoint samples for the three evaluation splits."""

    train: tuple[PositionSample, ...]
    validation: tuple[PositionSample, ...]
    test: tuple[PositionSample, ...]
    board_size: int
    seed: int
    n_phase_bins: int

    def manifest(self) -> dict[str, Any]:
        return {
            "protocol_tag": PROTOCOL_TAG,
            "board_size": self.board_size,
            "seed": self.seed,
            "n_phase_bins": self.n_phase_bins,
            "train": [sample.to_dict() for sample in self.train],
            "validation": [sample.to_dict() for sample in self.validation],
            "test": [sample.to_dict() for sample in self.test],
        }


@dataclass(frozen=True)
class PositionFeatureSet:
    """CPU fp32 feature cache and labels for a fixed sample sequence."""

    samples: tuple[PositionSample, ...]
    features: dict[int, torch.Tensor]
    absolute_labels: torch.Tensor
    relative_labels: torch.Tensor
    phase_bins: torch.Tensor
    board_size: int

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(int(layer) for layer in self.features))

    @property
    def d_model(self) -> int:
        first = self.features[self.layers[0]]
        return int(first.shape[1])

    @property
    def n_squares(self) -> int:
        return int(self.absolute_labels.shape[1])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)


def _phase_interval(
    phase_bin: int,
    *,
    max_position: int,
    n_phase_bins: int,
) -> tuple[int, int]:
    """Return an inclusive, non-empty integer interval for a phase bin."""
    start = (phase_bin * max_position) // n_phase_bins + 1
    end = ((phase_bin + 1) * max_position) // n_phase_bins
    if start > end:
        raise ValueError(
            f"Phase bin {phase_bin} is empty for max_position={max_position}; "
            f"need at least n_phase_bins={n_phase_bins} eligible positions."
        )
    return start, end


def _game_sha256(game: Sequence[int]) -> str:
    payload = ",".join(str(int(move)) for move in game).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_position_samples(
    records: Sequence[GameRecord],
    *,
    board_size: int,
    seed: int = 42,
    n_phase_bins: int = 4,
    load_chunk_fn: ChunkLoader = load_chunk,
) -> tuple[PositionSample, ...]:
    """Create one deterministic position per normalized phase bin and game.

    Eligible positions are prefixes that retain a real continuation:
    ``t in [1, min(len(game)-1, board_size**2-5)]``.  Each game's eligible
    range is divided into equal-count normalized bins and one position is
    sampled from every bin using a stable hash of seed, path, game index and
    bin.  Python's process-randomized ``hash`` is never used.

    The function is strict: duplicate records, missing games, or games too
    short to populate every bin raise instead of silently changing the sample
    budget.
    """
    if board_size < 4 or board_size % 2:
        raise ValueError(f"board_size must be an even integer >= 4, got {board_size}")
    if n_phase_bins <= 0:
        raise ValueError(f"n_phase_bins must be positive, got {n_phase_bins}")
    if not records:
        raise ValueError("records must not be empty")

    normalized_records = [_record_key(record) for record in records]
    if len(set(normalized_records)) != len(normalized_records):
        raise ValueError("records contain duplicate (chunk_path, game_index) entries")

    indices_by_chunk: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record_order, (chunk_path, game_index) in enumerate(normalized_records):
        if game_index < 0:
            raise ValueError(f"game_index must be non-negative, got {game_index}")
        indices_by_chunk[chunk_path].append((record_order, game_index))

    per_record: list[tuple[PositionSample, ...] | None] = [None] * len(records)
    model_block_size = board_size * board_size - 5
    for chunk_path, chunk_records in indices_by_chunk.items():
        games = load_chunk_fn(chunk_path)
        for record_order, game_index in chunk_records:
            if game_index >= len(games):
                raise IndexError(
                    f"game_index={game_index} is outside chunk {chunk_path!r} "
                    f"with {len(games)} games"
                )
            game_length = len(games[game_index])
            game_sha256 = _game_sha256(games[game_index])
            max_position = min(game_length - 1, model_block_size)
            if max_position < n_phase_bins:
                raise ValueError(
                    f"Game {chunk_path!r}[{game_index}] has only "
                    f"{max_position} eligible positions; need at least "
                    f"{n_phase_bins}."
                )

            samples: list[PositionSample] = []
            for phase_bin in range(n_phase_bins):
                start, end = _phase_interval(
                    phase_bin,
                    max_position=max_position,
                    n_phase_bins=n_phase_bins,
                )
                rng = random.Random(
                    _stable_seed(seed, chunk_path, game_index, phase_bin)
                )
                position = rng.randint(start, end)
                samples.append(
                    PositionSample(
                        chunk_path=chunk_path,
                        game_index=game_index,
                        position=position,
                        phase_bin=phase_bin,
                        normalized_phase=float(position / max_position),
                        game_length=game_length,
                        max_eligible_position=max_position,
                        game_sha256=game_sha256,
                    )
                )
            per_record[record_order] = tuple(samples)
        del games

    assert all(samples is not None for samples in per_record)
    return tuple(
        sample
        for record_samples in per_record
        for sample in (record_samples or ())
    )


def _assert_disjoint_records(
    train_records: Sequence[GameRecord],
    validation_records: Sequence[GameRecord],
    test_records: Sequence[GameRecord],
) -> None:
    split_keys = {
        "train": {_record_key(record) for record in train_records},
        "validation": {_record_key(record) for record in validation_records},
        "test": {_record_key(record) for record in test_records},
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_keys[left] & split_keys[right]
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(
                f"{left} and {right} records overlap; first overlap is {example}"
            )


def build_position_probe_splits(
    *,
    train_records: Sequence[GameRecord],
    validation_records: Sequence[GameRecord],
    test_records: Sequence[GameRecord],
    board_size: int,
    seed: int = 42,
    n_phase_bins: int = 4,
    load_chunk_fn: ChunkLoader = load_chunk,
) -> PositionProbeSplits:
    """Build fixed record-disjoint train/validation/test samples."""
    if not train_records or not validation_records or not test_records:
        raise ValueError("train_records, validation_records and test_records must be non-empty")
    _assert_disjoint_records(train_records, validation_records, test_records)
    return PositionProbeSplits(
        train=build_position_samples(
            train_records,
            board_size=board_size,
            seed=seed,
            n_phase_bins=n_phase_bins,
            load_chunk_fn=load_chunk_fn,
        ),
        validation=build_position_samples(
            validation_records,
            board_size=board_size,
            seed=seed,
            n_phase_bins=n_phase_bins,
            load_chunk_fn=load_chunk_fn,
        ),
        test=build_position_samples(
            test_records,
            board_size=board_size,
            seed=seed,
            n_phase_bins=n_phase_bins,
            load_chunk_fn=load_chunk_fn,
        ),
        board_size=board_size,
        seed=seed,
        n_phase_bins=n_phase_bins,
    )
