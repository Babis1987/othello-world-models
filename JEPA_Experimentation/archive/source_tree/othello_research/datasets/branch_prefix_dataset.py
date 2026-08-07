"""Prefix-grouped batch builder for hard-disjoint action JEPA (v6).

The v6 objective needs every batch to be structured as ``G`` prefix groups
of ``S`` games each, where all games inside one group share the same prefix
of length ``t``. Because the per-game OthelloChunkDataset cannot express
this batch structure on its own, this module loads one pickle chunk into
memory, builds a flat index of (game_idx, t, prefix-tuple), and exposes an
iterable that yields fully-formed batches.

The same module is reused at evaluation time to compute the v6 pretext
metrics on val chunks. Per-chunk RAM footprint stays modest because the
index stores integer game ids, not duplicated token sequences.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
import random
from pathlib import Path

import torch

from othello_research.datasets.dataset import load_chunk
from othello_research.datasets.move_mapping import build_mappings


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BranchPrefixBatchConfig:
    """Hyperparameters for branch-prefix batch construction."""

    board_size: int = 8
    t_min: int = 4
    t_max: int = 10
    groups_per_batch: int = 64
    samples_per_group: int = 4
    seed: int = 42


# ============================================================================
# Index of (t, prefix) -> game indices
# ============================================================================

def tokenize_chunk_games(
    games: list[list[int]],
    board_size: int,
) -> list[list[int]]:
    """Convert raw board positions to compact token ids for one chunk."""
    raw_to_token, _ = build_mappings(board_size)
    tokenized: list[list[int]] = []
    for game in games:
        tokens: list[int] = []
        invalid = False
        for raw in game:
            if not (0 <= raw < board_size * board_size):
                invalid = True
                break
            tok = raw_to_token[raw]
            if tok == -1:
                invalid = True
                break
            tokens.append(tok)
        if not invalid:
            tokenized.append(tokens)
    return tokenized


def build_prefix_index(
    tokenized_games: list[list[int]],
    *,
    t_min: int,
    t_max: int,
    samples_per_group: int,
) -> dict[int, dict[tuple[int, ...], list[int]]]:
    """For each ``t`` in ``[t_min, t_max]``, group game indices by prefix tuple.

    Only prefixes with at least ``samples_per_group`` games are retained, so
    every batch built from this index can satisfy the group size constraint
    without extra fallback paths in the sampler.
    """
    index: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for t in range(t_min, t_max + 1):
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for gi, game in enumerate(tokenized_games):
            if len(game) <= t:
                continue
            buckets[tuple(game[:t])].append(gi)
        viable = {prefix: gids for prefix, gids in buckets.items() if len(gids) >= samples_per_group}
        index[t] = viable
    return index


def index_diagnostics(
    index: dict[int, dict[tuple[int, ...], list[int]]],
) -> dict[int, dict[str, int]]:
    """Return per-``t`` counts of viable groups and total covered games."""
    stats: dict[int, dict[str, int]] = {}
    for t, buckets in index.items():
        total_games = sum(len(gids) for gids in buckets.values())
        stats[t] = {"viable_groups": len(buckets), "covered_games": total_games}
    return stats


# ============================================================================
# Batched iterable
# ============================================================================

class BranchPrefixChunkBatches:
    """Iterable yielding prefix-grouped batches from one in-RAM chunk.

    The iterable is finite: it emits all batches it can construct from the
    chunk's viable prefix groups, sampled without replacement (within one
    pass). Each batch picks a context length ``t`` from the viable set,
    then samples ``groups_per_batch`` distinct prefix groups and
    ``samples_per_group`` games from each.

    Output of ``next``:
        dict with keys:
            ``t``            : int, chosen context length
            ``x_context``    : ``(B, t)`` long
            ``next_actions`` : ``(B,)`` long
            ``prefix_ids``   : ``(B,)`` long, group ids in ``[0, G)``
            ``n_groups``     : int
            ``games_consumed`` : int
    """

    def __init__(
        self,
        games_path: str | Path,
        cfg: BranchPrefixBatchConfig,
    ) -> None:
        self.cfg = cfg
        raw_games = load_chunk(str(games_path))
        self.games = tokenize_chunk_games(raw_games, cfg.board_size)
        self.index = build_prefix_index(
            self.games,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            samples_per_group=cfg.samples_per_group,
        )
        self.diagnostics = index_diagnostics(self.index)
        self.rng = random.Random(cfg.seed)
        # Books we keep updated during one pass so groups are not reused.
        self._t_pool: dict[int, list[tuple[int, ...]]] = {
            t: list(buckets.keys()) for t, buckets in self.index.items()
        }
        for t in self._t_pool:
            self.rng.shuffle(self._t_pool[t])

    def __iter__(self) -> Iterator[dict[str, object]]:
        return self

    def __next__(self) -> dict[str, object]:
        viable_ts = [t for t, pool in self._t_pool.items() if len(pool) >= self.cfg.groups_per_batch]
        if not viable_ts:
            raise StopIteration
        t = self.rng.choice(viable_ts)
        chosen_prefixes = [self._t_pool[t].pop() for _ in range(self.cfg.groups_per_batch)]

        x_rows: list[list[int]] = []
        next_actions: list[int] = []
        prefix_ids: list[int] = []
        for gid_idx, prefix in enumerate(chosen_prefixes):
            game_ids = self.index[t][prefix]
            sampled = self.rng.sample(game_ids, self.cfg.samples_per_group)
            for gi in sampled:
                game = self.games[gi]
                x_rows.append(game[:t])
                next_actions.append(game[t])
                prefix_ids.append(gid_idx)

        return {
            "t": t,
            "x_context": torch.tensor(x_rows, dtype=torch.long),
            "next_actions": torch.tensor(next_actions, dtype=torch.long),
            "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
            "n_groups": len(chosen_prefixes),
            "games_consumed": len(x_rows),
        }

    def remaining_batches(self) -> int:
        """Approximate number of batches still constructible from this pass."""
        return sum(len(pool) // self.cfg.groups_per_batch for pool in self._t_pool.values())


# ============================================================================
# Smoke test
# ============================================================================

def _smoke() -> None:
    import pickle
    import tempfile

    # Build a synthetic chunk of 200 games, each 20 tokens, with controlled
    # prefix collisions so the smoke can exercise grouping logic.
    rng = random.Random(0)
    board_size = 8
    raw_to_token, token_to_raw = build_mappings(board_size)
    valid_raw = [r for r, tok in enumerate(raw_to_token) if tok != -1]
    games: list[list[int]] = []
    # Pick a small pool of seed prefixes (length 6 in raw token space).
    n_seeds = 25
    seeds = [tuple(rng.sample(valid_raw, k=6)) for _ in range(n_seeds)]
    for _ in range(200):
        seed = list(rng.choice(seeds))
        suffix = rng.sample(valid_raw, k=14)
        game = seed + suffix
        games.append(game)

    with tempfile.TemporaryDirectory() as tmp:
        chunk_path = Path(tmp) / "smoke.pickle"
        with open(chunk_path, "wb") as f:
            pickle.dump(games, f)

        cfg = BranchPrefixBatchConfig(
            board_size=board_size,
            t_min=4,
            t_max=6,
            groups_per_batch=4,
            samples_per_group=3,
            seed=42,
        )
        batches = BranchPrefixChunkBatches(chunk_path, cfg)
        print("index diagnostics:", batches.diagnostics)
        first = next(batches)
        assert first["x_context"].shape == (cfg.groups_per_batch * cfg.samples_per_group, first["t"]), \
            first["x_context"].shape
        assert first["next_actions"].shape == (cfg.groups_per_batch * cfg.samples_per_group,)
        assert first["prefix_ids"].shape == (cfg.groups_per_batch * cfg.samples_per_group,)
        # Group invariant: rows with the same prefix_id share the same context.
        for g in range(cfg.groups_per_batch):
            mask = (first["prefix_ids"] == g)
            members = first["x_context"][mask]
            assert (members == members[0:1]).all(), f"prefix-group {g} has differing contexts"
        print(
            "v6 sampler smoke OK",
            "B=", first["x_context"].shape[0],
            "t=", first["t"],
            "remaining=", batches.remaining_batches(),
        )


if __name__ == "__main__":
    _smoke()
