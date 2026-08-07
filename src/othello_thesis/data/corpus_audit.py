"""Audit a generated Othello corpus without scanning every move in RAM.

The audit verifies the complete manifest/file inventory and then performs
SHA-256, pickle-schema, dataset-compatibility, and independent legal-replay
checks on shards spread across the full global-index range and split boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from othello_thesis.data.chunk_dataset import OthelloChunkDataset
from othello_thesis.game_engine.board import OthelloBoardState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_indices(count: int, requested: int) -> list[int]:
    if count <= 0 or requested <= 0:
        return []
    return sorted(
        set(
            int(value)
            for value in np.linspace(
                0, count - 1, num=min(count, requested), dtype=np.int64
            )
        )
    )


def _selected_shard_indices(
    shard_count: int,
    requested: int,
    entries: list[dict[str, Any]],
    train_games: int | None,
) -> list[int]:
    selected = set(_sample_indices(shard_count, requested))
    if train_games is not None:
        for index, entry in enumerate(entries):
            start = int(entry["global_start_index"])
            end = int(entry["global_end_index_exclusive"])
            if start == train_games or end == train_games:
                selected.add(index)
    return sorted(selected)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("shards", [])

    assert manifest["complete"] is True
    assert int(manifest["board_size"]) == args.board_size
    assert int(manifest["max_moves"]) == args.board_size**2 - 4
    assert int(manifest["model_block_size"]) == args.board_size**2 - 5
    assert int(manifest["generated_games"]) == args.expected_total
    assert int(manifest["num_games_requested"]) == args.expected_total
    if args.expected_train is not None:
        assert int(manifest["training_games"]) == args.expected_train
        assert int(manifest["heldout_games"]) == (
            args.expected_total - args.expected_train
        )

    disk_files = sorted(path.name for path in data_dir.glob("games_*.pickle"))
    manifest_files = sorted(str(entry["filename"]) for entry in entries)
    assert disk_files == manifest_files
    assert len(entries) == len(manifest["shard_counts"])

    expected_start = int(manifest["global_start_index"])
    counted_games = 0
    counted_bytes = 0
    for entry in entries:
        path = data_dir / entry["filename"]
        start = int(entry["global_start_index"])
        end = int(entry["global_end_index_exclusive"])
        count = int(entry["count"])
        assert start == expected_start
        assert end - start == count
        assert int(manifest["shard_counts"][entry["filename"]]) == count
        assert path.stat().st_size == int(entry["bytes"])
        assert len(str(entry["sha256"])) == 64
        expected_start = end
        counted_games += count
        counted_bytes += int(entry["bytes"])
    assert counted_games == args.expected_total

    chosen = _selected_shard_indices(
        len(entries), args.sample_shards, entries, args.expected_train
    )
    center = {
        (args.board_size // 2 - 1) * args.board_size
        + (args.board_size // 2 - 1),
        (args.board_size // 2 - 1) * args.board_size
        + args.board_size // 2,
        (args.board_size // 2) * args.board_size
        + (args.board_size // 2 - 1),
        (args.board_size // 2) * args.board_size
        + args.board_size // 2,
    }
    sampled_games = 0
    replayed_games = 0
    observed_lengths: list[int] = []
    sampled_shards: list[dict[str, Any]] = []

    for shard_index in tqdm(chosen, desc="Auditing sampled shards"):
        entry = entries[shard_index]
        path = data_dir / entry["filename"]
        assert _sha256(path) == entry["sha256"]
        with path.open("rb") as handle:
            games = pickle.load(handle)
        assert type(games) is list
        assert len(games) == int(entry["count"])

        game_indices = _sample_indices(
            len(games), args.games_per_sampled_shard
        )
        replay_indices = set(
            _sample_indices(len(games), args.replays_per_sampled_shard)
        )
        compatible_games: list[list[int]] = []
        for game_index in game_indices:
            game = games[game_index]
            assert type(game) is list
            assert 0 < len(game) <= args.board_size**2 - 4
            assert all(type(move) is int for move in game)
            assert all(0 <= move < args.board_size**2 for move in game)
            assert all(move not in center for move in game)
            assert len(game) == len(set(game))
            observed_lengths.append(len(game))
            compatible_games.append(game)
            sampled_games += 1

        for game_index in replay_indices:
            board = OthelloBoardState(args.board_size)
            board.update(games[game_index])
            assert board.is_game_over()
            replayed_games += 1

        dataset = OthelloChunkDataset(
            compatible_games,
            block_size=args.board_size**2 - 5,
            board_size=args.board_size,
        )
        x, y = dataset[0]
        assert tuple(x.shape) == (args.board_size**2 - 5,)
        assert tuple(y.shape) == (args.board_size**2 - 5,)
        assert int(x.min()) >= 0
        assert int(x.max()) <= args.board_size**2 - 4
        assert int(y.max()) <= args.board_size**2 - 5

        sampled_shards.append(
            {
                "shard_index": shard_index,
                "filename": entry["filename"],
                "global_start_index": entry["global_start_index"],
                "count": entry["count"],
                "sha256_verified": True,
            }
        )
        del games

    result = {
        "status": "PASS",
        "data_dir": str(data_dir),
        "board_size": args.board_size,
        "max_moves": args.board_size**2 - 4,
        "model_block_size": args.board_size**2 - 5,
        "vocab_size": args.board_size**2 - 3,
        "total_games": counted_games,
        "training_games": manifest.get("training_games"),
        "heldout_games": manifest.get("heldout_games"),
        "shards": len(entries),
        "pickle_bytes": counted_bytes,
        "sampled_shards": sampled_shards,
        "sampled_games_schema_checked": sampled_games,
        "independent_games_replayed": replayed_games,
        "sample_min_moves": min(observed_lengths),
        "sample_max_moves": max(observed_lengths),
        "sample_mean_moves": float(np.mean(observed_lengths)),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--board-size", required=True, type=int)
    parser.add_argument("--expected-total", required=True, type=int)
    parser.add_argument("--expected-train", type=int, default=None)
    parser.add_argument("--sample-shards", type=int, default=12)
    parser.add_argument("--games-per-sampled-shard", type=int, default=128)
    parser.add_argument("--replays-per-sampled-shard", type=int, default=4)
    parser.add_argument("--report", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args)
    rendered = json.dumps(result, indent=2)
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report: {report_path}")
    print(rendered)


if __name__ == "__main__":
    main()
