"""Shared chunk discovery and DataLoader construction for canonical AR runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from othello_thesis.data.chunk_dataset import OthelloChunkDataset, load_chunk


def list_chunks(data_dir: str) -> list[Path]:
    """List all pickle chunks in deterministic filename order."""
    chunks = sorted(Path(data_dir).glob("*.pickle"))
    if not chunks:
        raise FileNotFoundError(f"No .pickle chunks found in {data_dir}")
    return chunks


def resolve_chunks(cfg: Any) -> tuple[list[Path], list[Path]]:
    """Resolve the historical Li split or explicit disjoint directories."""
    if cfg.data_dir:
        all_chunks = list_chunks(cfg.data_dir)
        n_train = cfg.li_split_train_chunks
        if len(all_chunks) <= n_train:
            raise ValueError(
                f"data_dir has {len(all_chunks)} chunks, but li_split_train_chunks="
                f"{n_train}; no validation chunks would remain. Either lower "
                f"li_split_train_chunks or use --train_dir/--val_dir explicitly."
            )
        train_chunks = all_chunks[:n_train]
        val_chunks = all_chunks[n_train:]
    else:
        if not cfg.train_dir:
            raise ValueError("Provide either --data_dir or --train_dir.")
        train_chunks = list_chunks(cfg.train_dir)
        val_chunks = list_chunks(cfg.val_dir) if cfg.val_dir else []

    train_set = {path.resolve() for path in train_chunks}
    val_set = {path.resolve() for path in val_chunks}
    overlap = train_set & val_set
    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            f"Train/validation chunks overlap. Example: {example}. "
            f"Train and val must be disjoint."
        )
    return train_chunks, val_chunks


def parse_milestones(spec: str) -> list[int]:
    """Parse the historical comma-separated game milestones."""
    if not spec.strip():
        return []
    return sorted({int(value.strip()) for value in spec.split(",") if value.strip()})


def read_manifest_counts(chunk_dir: Path) -> dict[str, int]:
    """Read shard counts, returning an empty mapping on missing/bad manifests."""
    manifest = chunk_dir / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        with open(manifest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    counts = data.get("shard_counts", data)
    if not isinstance(counts, dict):
        return {}
    return {str(key): int(value) for key, value in counts.items()}


def estimate_total_steps(chunks: list[Path], cfg: Any) -> tuple[int, int]:
    """Reproduce the manifest/fallback total-step estimate and stdout."""
    counts: dict[str, int] = {}
    for parent in sorted({path.parent for path in chunks}):
        counts.update(read_manifest_counts(parent))

    if counts and all(path.name in counts for path in chunks):
        total_games = sum(counts[path.name] for path in chunks)
        source = "manifest"
    else:
        total_games = len(chunks) * cfg.games_per_chunk
        source = f"games_per_chunk={cfg.games_per_chunk}"

    if cfg.max_chunks:
        per_chunk = (
            counts.get(chunks[0].name, cfg.games_per_chunk)
            if counts
            else cfg.games_per_chunk
        )
        total_games = min(total_games, cfg.max_chunks * per_chunk)

    if cfg.total_steps > 0:
        total_steps = cfg.total_steps
        print(f"using explicit total_steps={total_steps}", flush=True)
    else:
        total_steps = math.ceil(total_games / cfg.batch_size) * cfg.passes_over_data
        print(
            f"estimated total_steps={total_steps} ({source}, "
            f"~{total_games:,} games over {cfg.passes_over_data} pass(es))",
            flush=True,
        )
    return total_steps, total_games


def make_chunk_loader(
    chunk_path: Path,
    *,
    block_size: int,
    board_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[DataLoader, list[list[int]], OthelloChunkDataset]:
    """Load one shard and construct the historical fresh DataLoader."""
    games = load_chunk(str(chunk_path))
    dataset = OthelloChunkDataset(
        games=games,
        block_size=block_size,
        board_size=board_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, games, dataset
