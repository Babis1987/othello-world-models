"""(t, next-action)-grouped batches for v7 grouped InfoNCE."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
import math
from pathlib import Path
import random

import torch

from othello_research.datasets.dataset import load_chunk
from othello_research.datasets.move_mapping import build_mappings


@dataclass
class ActionGroupedBatchConfig:
    board_size: int = 8
    t_min: int = 4
    t_max: int = 10
    groups_per_batch: int = 8
    samples_per_group: int = 16
    chunk_sample_multiplier: float = 1.0
    seed: int = 42


def tokenize_chunk_games(
    games: list[list[int]],
    board_size: int,
) -> list[list[int]]:
    """Convert raw board positions to the repository's compact move tokens."""
    raw_to_token, _ = build_mappings(board_size)
    tokenized: list[list[int]] = []
    for game in games:
        tokens: list[int] = []
        for raw in game:
            if not (0 <= raw < board_size * board_size):
                tokens = []
                break
            token = raw_to_token[raw]
            if token < 0:
                tokens = []
                break
            tokens.append(token)
        if tokens:
            tokenized.append(tokens)
    return tokenized


def build_action_index(
    tokenized_games: list[list[int]],
    *,
    t_min: int,
    t_max: int,
    samples_per_group: int,
) -> dict[int, dict[int, list[int]]]:
    """Build usable ``t -> next_action -> distinct game ids`` buckets."""
    index: dict[int, dict[int, list[int]]] = {}
    for t in range(t_min, t_max + 1):
        buckets: dict[int, list[int]] = defaultdict(list)
        for game_idx, game in enumerate(tokenized_games):
            if len(game) > t:
                buckets[game[t]].append(game_idx)
        index[t] = {
            action: game_ids
            for action, game_ids in buckets.items()
            if len(game_ids) >= samples_per_group
        }
    return index


def action_index_diagnostics(
    tokenized_games: list[list[int]],
    index: dict[int, dict[int, list[int]]],
    *,
    t_min: int,
    t_max: int,
    samples_per_group: int,
) -> dict[str, object]:
    total_buckets = 0
    usable_buckets = 0
    covered_pairs = 0
    per_t: dict[int, dict[str, int]] = {}
    for t in range(t_min, t_max + 1):
        all_actions: dict[int, int] = defaultdict(int)
        for game in tokenized_games:
            if len(game) > t:
                all_actions[game[t]] += 1
        usable = index.get(t, {})
        total_buckets += len(all_actions)
        usable_buckets += len(usable)
        covered_pairs += sum(len(game_ids) for game_ids in usable.values())
        per_t[t] = {
            "total_buckets": len(all_actions),
            "usable_buckets": len(usable),
            "covered_game_action_pairs": sum(len(ids) for ids in usable.values()),
        }
    return {
        "total_buckets": total_buckets,
        "usable_buckets": usable_buckets,
        "usable_fraction": usable_buckets / max(1, total_buckets),
        "covered_game_action_pairs": covered_pairs,
        "minimum_bucket_size": samples_per_group,
        "per_t": per_t,
    }


class ActionGroupedChunkBatches:
    """Finite iterable yielding G action groups of M distinct games.

    Every emitted batch uses one context length ``t`` because the causal
    Transformer has no padding attention mask. Groups within the batch use
    distinct actions, and each group contains distinct game indices.
    """

    def __init__(
        self,
        games_path: str | Path,
        cfg: ActionGroupedBatchConfig,
    ) -> None:
        if cfg.groups_per_batch < 1:
            raise ValueError("groups_per_batch must be positive")
        if cfg.samples_per_group < 2:
            raise ValueError("samples_per_group must be at least 2")
        if cfg.t_min < 1 or cfg.t_max < cfg.t_min:
            raise ValueError("Expected 1 <= t_min <= t_max")
        if cfg.chunk_sample_multiplier < 0:
            raise ValueError("chunk_sample_multiplier must be non-negative")

        self.cfg = cfg
        self.games = tokenize_chunk_games(load_chunk(str(games_path)), cfg.board_size)
        effective_batch_size = cfg.groups_per_batch * cfg.samples_per_group
        self.max_batches = (
            math.ceil(len(self.games) * cfg.chunk_sample_multiplier / effective_batch_size)
            if cfg.chunk_sample_multiplier > 0
            else None
        )
        self._batches_emitted = 0
        self.index = build_action_index(
            self.games,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            samples_per_group=cfg.samples_per_group,
        )
        self.diagnostics = action_index_diagnostics(
            self.games,
            self.index,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            samples_per_group=cfg.samples_per_group,
        )
        self.rng = random.Random(cfg.seed)
        self._game_pools = {
            t: {action: list(game_ids) for action, game_ids in actions.items()}
            for t, actions in self.index.items()
        }
        for actions in self._game_pools.values():
            for game_ids in actions.values():
                self.rng.shuffle(game_ids)

    def __iter__(self) -> Iterator[dict[str, object]]:
        return self

    def __next__(self) -> dict[str, object]:
        if self.max_batches is not None and self._batches_emitted >= self.max_batches:
            raise StopIteration
        viable_by_t = {
            t: [
                action
                for action, game_ids in actions.items()
                if len(game_ids) >= self.cfg.samples_per_group
            ]
            for t, actions in self._game_pools.items()
        }
        viable_t = [
            t for t, actions in viable_by_t.items()
            if len(actions) >= self.cfg.groups_per_batch
        ]
        if not viable_t:
            raise StopIteration
        t = self.rng.choice(viable_t)
        selected_actions = self.rng.sample(
            viable_by_t[t],
            self.cfg.groups_per_batch,
        )

        contexts: list[list[int]] = []
        targets: list[list[int]] = []
        actions: list[int] = []
        group_ids: list[int] = []
        game_indices: list[int] = []
        for group_id, action in enumerate(selected_actions):
            selected_games = [
                self._game_pools[t][action].pop()
                for _ in range(self.cfg.samples_per_group)
            ]
            for game_idx in selected_games:
                game = self.games[game_idx]
                contexts.append(game[:t])
                targets.append(game[: t + 1])
                actions.append(action)
                group_ids.append(group_id)
                game_indices.append(game_idx)

        self._batches_emitted += 1
        return {
            "t": t,
            "x_context": torch.tensor(contexts, dtype=torch.long),
            "x_target": torch.tensor(targets, dtype=torch.long),
            "next_actions": torch.tensor(actions, dtype=torch.long),
            "group_ids": torch.tensor(group_ids, dtype=torch.long),
            "game_indices": torch.tensor(game_indices, dtype=torch.long),
            "n_groups": len(selected_actions),
            "games_consumed": len(contexts),
        }

    def remaining_batches(self) -> int:
        available = sum(
            sum(len(game_ids) // self.cfg.samples_per_group for game_ids in actions.values())
            // self.cfg.groups_per_batch
            for actions in self._game_pools.values()
        )
        if self.max_batches is None:
            return available
        return min(available, max(0, self.max_batches - self._batches_emitted))
