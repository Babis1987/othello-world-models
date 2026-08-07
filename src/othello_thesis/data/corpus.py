"""Generate deterministic, resumable Othello game corpora.

Each pickle shard contains ``list[list[int]]`` raw square ids, matching the
format used by the training code.  The default Numba engine supports any even
board size and derives an independent RNG stream from each global game index.
The resulting corpus is therefore unchanged by resumes, shard sizes, or thread
counts.

Example (20 million 12x12 training games):

    python scripts/generate_corpus.py \
        --board-size 12 \
        --num-games 23800000 \
        --train-games 20000000 \
        --output-dir "G:/My Drive/Master_Thesis_Artifacts/data/othello_12x12/corpus_v1" \
        --games-per-shard 100000 \
        --seed 42 \
        --num-threads 16 \
        --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from othello_thesis.game_engine.board import OthelloBoardState
from othello_thesis.game_engine.generator import (
    dense_to_game_lists,
    generate_games_array,
    validate_board_size,
)


SCHEMA_VERSION = 2
GENERATOR_VERSION = "deterministic_numba_v1"


def generate_game(n: int, rng: random.Random | None = None) -> list[int]:
    """Generate one game with the slow reference engine.

    This remains useful for smoke tests.  Full corpora should use the default
    Numba engine.
    """
    validate_board_size(n)
    rng = rng or random.Random()
    board = OthelloBoardState(n)
    moves: list[int] = []
    while not board.is_game_over():
        valid = board.get_valid_moves()
        if not valid:
            board.next_hand_color *= -1
            valid = board.get_valid_moves()
        move = rng.choice(valid)
        board.umpire(move)
        moves.append(move)
    return moves


def _reference_seed(seed: int, global_game_index: int) -> int:
    payload = f"{seed}:{global_game_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _generate_games(
    *,
    board_size: int,
    global_start_index: int,
    count: int,
    seed: int,
    num_threads: int,
    engine: str,
) -> list[list[int]]:
    if engine == "numba":
        moves, lengths = generate_games_array(
            board_size,
            global_start_index=global_start_index,
            count=count,
            seed=seed,
            num_threads=num_threads,
        )
        return dense_to_game_lists(moves, lengths)

    return [
        generate_game(
            board_size,
            random.Random(_reference_seed(seed, global_start_index + offset)),
        )
        for offset in range(count)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_pickle(path: Path, games: list[list[int]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(games, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_games(
    games: list[list[int]],
    board_size: int,
    sample_count: int,
) -> None:
    if not games:
        return
    count = min(sample_count, len(games))
    indices = np.linspace(0, len(games) - 1, num=count, dtype=np.int64)
    for index in np.unique(indices):
        game = games[int(index)]
        if not 0 < len(game) <= board_size * board_size - 4:
            raise RuntimeError(
                f"Generated game {index} has invalid length {len(game)}"
            )
        board = OthelloBoardState(board_size)
        try:
            board.update(game)
        except (AssertionError, IndexError) as exc:
            raise RuntimeError(
                f"Generated game {index} failed reference replay"
            ) from exc
        if not board.is_game_over():
            raise RuntimeError(f"Generated game {index} is not terminal")


def _new_manifest(args: argparse.Namespace) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "format": "pickle_list_of_raw_move_lists",
        "generator_version": GENERATOR_VERSION,
        "engine": args.engine,
        "board_size": args.board_size,
        "max_moves": args.board_size * args.board_size - 4,
        "model_block_size": args.board_size * args.board_size - 5,
        "seed": args.seed,
        "global_start_index": args.global_start_index,
        "num_games_requested": args.num_games,
        "training_games": args.train_games,
        "heldout_games": (
            args.num_games - args.train_games
            if args.train_games is not None
            else None
        ),
        "games_per_shard": args.games_per_shard,
        "generated_games": 0,
        "complete": False,
        "created_utc": now,
        "updated_utc": now,
        "shard_counts": {},
        "shards": [],
    }


def _assert_resume_compatible(
    manifest: dict[str, Any], args: argparse.Namespace
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "engine": args.engine,
        "board_size": args.board_size,
        "seed": args.seed,
        "global_start_index": args.global_start_index,
        "num_games_requested": args.num_games,
        "training_games": args.train_games,
        "games_per_shard": args.games_per_shard,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: existing={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise RuntimeError(f"Cannot resume incompatible dataset: {details}")


def _load_or_create_manifest(
    output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    pickle_files = list(output_dir.glob("*.pickle"))
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} already exists; pass --resume to continue safely"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _assert_resume_compatible(manifest, args)
        return manifest
    if pickle_files:
        raise RuntimeError(
            f"{output_dir} contains pickle shards but no manifest.json; refusing "
            "to guess their provenance"
        )
    return _new_manifest(args)


def _expected_shards(args: argparse.Namespace):
    for offset in range(0, args.num_games, args.games_per_shard):
        count = min(args.games_per_shard, args.num_games - offset)
        start = args.global_start_index + offset
        end = start + count
        filename = f"games_{start:09d}_{end - 1:09d}.pickle"
        yield filename, start, end, count


def _verify_completed_shards(
    output_dir: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> set[str]:
    entries = {entry["filename"]: entry for entry in manifest.get("shards", [])}
    completed: set[str] = set()
    for filename, start, end, count in _expected_shards(args):
        entry = entries.get(filename)
        if entry is None:
            continue
        path = output_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Manifest references missing shard: {path}")
        expected_fields = {
            "global_start_index": start,
            "global_end_index_exclusive": end,
            "count": count,
        }
        for key, expected in expected_fields.items():
            if entry.get(key) != expected:
                raise RuntimeError(
                    f"Invalid {key} for existing shard {filename}: "
                    f"{entry.get(key)!r} != {expected!r}"
                )
        if not args.skip_existing_hash_check:
            actual_hash = _sha256(path)
            if actual_hash != entry.get("sha256"):
                raise RuntimeError(f"SHA256 mismatch for existing shard: {path}")
        completed.add(filename)
    return completed


def _record_shard(
    manifest: dict[str, Any],
    *,
    filename: str,
    start: int,
    end: int,
    games: list[list[int]],
    sha256: str,
) -> None:
    lengths = np.fromiter((len(game) for game in games), dtype=np.int16)
    entry = {
        "filename": filename,
        "global_start_index": start,
        "global_end_index_exclusive": end,
        "count": len(games),
        "sha256": sha256,
        "bytes": None,
        "min_moves": int(lengths.min()),
        "max_moves": int(lengths.max()),
        "mean_moves": round(float(lengths.mean()), 6),
    }
    entries = {
        existing["filename"]: existing
        for existing in manifest.get("shards", [])
    }
    entries[filename] = entry
    manifest["shards"] = sorted(
        entries.values(), key=lambda item: item["global_start_index"]
    )
    manifest["shard_counts"] = {
        item["filename"]: item["count"] for item in manifest["shards"]
    }
    manifest["generated_games"] = sum(manifest["shard_counts"].values())
    manifest["complete"] = (
        manifest["generated_games"] == manifest["num_games_requested"]
    )
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    validate_board_size(args.board_size)
    if args.num_games <= 0:
        raise ValueError("--num-games must be positive")
    if args.games_per_shard <= 0:
        raise ValueError("--games-per-shard must be positive")
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")
    if args.verify_sample < 0:
        raise ValueError("--verify-sample must be non-negative")
    if args.global_start_index < 0:
        raise ValueError("--global-start-index must be non-negative")
    if not 0 <= args.seed <= 2**64 - 1:
        raise ValueError("--seed must be in [0, 2**64 - 1]")
    if args.train_games is not None:
        if not 0 < args.train_games < args.num_games:
            raise ValueError(
                "--train-games must be positive and smaller than --num-games"
            )
        if args.train_games % args.games_per_shard != 0:
            raise ValueError(
                "--train-games must align with --games-per-shard so a shard "
                "cannot cross the train/held-out boundary"
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = _load_or_create_manifest(output_dir, args)
    _atomic_json(manifest_path, manifest)
    completed = _verify_completed_shards(output_dir, manifest, args)

    already_done = sum(
        count
        for filename, _start, _end, count in _expected_shards(args)
        if filename in completed
    )
    progress = tqdm(
        total=args.num_games,
        initial=already_done,
        unit="game",
        desc=f"Generating {args.board_size}x{args.board_size}",
    )
    try:
        for filename, start, end, count in _expected_shards(args):
            if filename in completed:
                continue
            games = _generate_games(
                board_size=args.board_size,
                global_start_index=start,
                count=count,
                seed=args.seed,
                num_threads=args.num_threads,
                engine=args.engine,
            )
            if len(games) != count:
                raise RuntimeError(
                    f"Generator returned {len(games)} games, expected {count}"
                )
            _validate_games(games, args.board_size, args.verify_sample)
            path = output_dir / filename
            _atomic_pickle(path, games)
            digest = _sha256(path)
            _record_shard(
                manifest,
                filename=filename,
                start=start,
                end=end,
                games=games,
                sha256=digest,
            )
            for entry in manifest["shards"]:
                if entry["filename"] == filename:
                    entry["bytes"] = path.stat().st_size
                    break
            _atomic_json(manifest_path, manifest)
            progress.update(count)
            del games
    finally:
        progress.close()

    if not manifest["complete"]:
        raise RuntimeError(
            f"Generation stopped at {manifest['generated_games']:,}/"
            f"{args.num_games:,} games"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic random Othello game sequences"
    )
    parser.add_argument("--board-size", type=int, required=True)
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument(
        "--train-games",
        type=int,
        default=None,
        help=(
            "Optional train/held-out boundary recorded in the manifest. "
            "For 8x8-compatible splitting use 20000000 of 23800000."
        ),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--games-per-shard", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-start-index", type=int, default=0)
    parser.add_argument(
        "--num-threads",
        "--num-workers",
        dest="num_threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Numba threads (legacy alias: --num-workers)",
    )
    parser.add_argument(
        "--engine", choices=("numba", "reference"), default="numba"
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=128,
        help="Games per new shard replayed with the reference engine",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and skip shards already recorded in manifest.json",
    )
    parser.add_argument(
        "--skip-existing-hash-check",
        action="store_true",
        help="Faster resume, but do not verify existing shard SHA256 values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    print(
        f"Done: {manifest['generated_games']:,} "
        f"{manifest['board_size']}x{manifest['board_size']} games in "
        f"{len(manifest['shards'])} shards -> {Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
