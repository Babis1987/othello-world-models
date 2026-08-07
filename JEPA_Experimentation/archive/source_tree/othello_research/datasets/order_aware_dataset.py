"""Pair-index-backed grouped batches for v8 order-aware JEPA."""

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
from othello_research.datasets.order_aware_pair_index import OrderAwarePairIndex


@dataclass
class OrderAwareBatchConfig:
    board_size: int = 8
    t_min: int = 4
    t_max: int = 40
    groups_per_batch: int = 8
    samples_per_group: int = 16
    hard_negatives_per_anchor: int = 1
    chunk_sample_multiplier: float = 1.0
    pair_chunk_cache_size: int = 4
    min_pair_anchors_per_group: int = 0
    expected_use_constructive_pairs: bool | None = None
    expected_use_surface_hard_negatives: bool | None = None
    expected_surface_jaccard_threshold: float | None = None
    expected_positions_per_game: int | None = None
    seed: int = 42


class OrderAwareChunkBatches:
    """Finite grouped batches constructed exclusively from moves and pair index."""

    def __init__(
        self,
        games_path: str | Path,
        pair_index_path: str | Path,
        data_dir: str | Path,
        cfg: OrderAwareBatchConfig,
    ) -> None:
        if cfg.groups_per_batch < 1:
            raise ValueError("groups_per_batch must be positive")
        if cfg.samples_per_group < 2:
            raise ValueError("samples_per_group must be at least 2")
        if cfg.hard_negatives_per_anchor < 0:
            raise ValueError("hard_negatives_per_anchor must be non-negative")
        if not 0 <= cfg.min_pair_anchors_per_group <= cfg.samples_per_group:
            raise ValueError(
                "min_pair_anchors_per_group must be in "
                "[0, samples_per_group]"
            )
        self.cfg = cfg
        self.games_path = Path(games_path)
        raw_games = load_chunk(str(self.games_path))
        raw_to_token, _ = build_mappings(cfg.board_size)
        self.games: dict[int, list[int]] = {}
        for original_game_idx, raw_game in enumerate(raw_games):
            tokens: list[int] = []
            for raw_move in raw_game:
                if not (0 <= raw_move < cfg.board_size * cfg.board_size):
                    tokens = []
                    break
                token = raw_to_token[raw_move]
                if token < 0:
                    tokens = []
                    break
                tokens.append(token)
            if tokens:
                self.games[original_game_idx] = tokens
        self.pairs = OrderAwarePairIndex(
            pair_index_path,
            data_dir,
            board_size=cfg.board_size,
            chunk_cache_size=cfg.pair_chunk_cache_size,
        )
        if cfg.t_min < self.pairs.t_min or cfg.t_max > self.pairs.t_max:
            self.pairs.close()
            raise ValueError(
                f"Requested t=[{cfg.t_min},{cfg.t_max}] falls outside pair-index "
                f"coverage [{self.pairs.t_min},{self.pairs.t_max}]"
            )
        if (
            cfg.expected_use_constructive_pairs is not None
            and self.pairs.use_constructive_pairs
            != cfg.expected_use_constructive_pairs
        ):
            self.pairs.close()
            raise ValueError(
                "Pair-index constructive setting does not match the training "
                "config; rebuild the v8 pair index"
            )
        if (
            cfg.expected_use_surface_hard_negatives is not None
            and self.pairs.use_surface_hard_negatives
            != cfg.expected_use_surface_hard_negatives
        ):
            self.pairs.close()
            raise ValueError(
                "Pair-index surface hard-negative setting does not match the "
                "training config; rebuild the v8 pair index"
            )
        if (
            cfg.expected_surface_jaccard_threshold is not None
            and abs(
                self.pairs.surface_jaccard_threshold
                - cfg.expected_surface_jaccard_threshold
            )
            > 1e-9
        ):
            self.pairs.close()
            raise ValueError(
                "Pair-index surface_jaccard_threshold does not match the "
                "training config; rebuild the v8 pair index"
            )
        if (
            cfg.expected_positions_per_game is not None
            and self.pairs.positions_per_game != cfg.expected_positions_per_game
        ):
            self.pairs.close()
            raise ValueError(
                "Pair-index positions_per_game does not match the training "
                "config; rebuild the v8 pair index"
            )
        self.pairs.pairs_for_chunk(self.games_path.name)
        self.rng = random.Random(cfg.seed)
        self._batches_emitted = 0
        effective_batch = cfg.groups_per_batch * cfg.samples_per_group

        raw_pools: dict[int, dict[int, dict[bytes, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        raw_pair_pools: dict[int, dict[int, dict[bytes, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        indexed_anchor_count = 0
        for game_idx, t, state_key in self.pairs.indexed_states(
            self.games_path.name
        ):
            game = self.games.get(game_idx)
            if game is None or not (cfg.t_min <= t <= cfg.t_max) or len(game) <= t:
                continue
            raw_pools[t][game[t]][state_key].append(game_idx)
            indexed_anchor_count += 1
            if (
                self.pairs.positive_refs(self.games_path.name, game_idx, t)
                or self.pairs.hard_negative_refs(self.games_path.name, game_idx, t)
            ):
                raw_pair_pools[t][game[t]][state_key].append(game_idx)
        self.indexed_anchor_count = indexed_anchor_count
        self.max_batches = (
            math.ceil(
                indexed_anchor_count
                * cfg.chunk_sample_multiplier
                / effective_batch
            )
            if cfg.chunk_sample_multiplier > 0
            else None
        )

        regular_needed = cfg.samples_per_group - cfg.min_pair_anchors_per_group
        pools: dict[int, dict[int, dict[bytes, list[int]]]] = {}
        pair_pools: dict[int, dict[int, dict[bytes, list[int]]]] = {}
        for t in range(cfg.t_min, cfg.t_max + 1):
            pools[t] = {}
            pair_pools[t] = {}
            for action, state_buckets in raw_pools.get(t, {}).items():
                action_pair_pools = dict(
                    raw_pair_pools.get(t, {}).get(action, {})
                )
                if len(action_pair_pools) < cfg.min_pair_anchors_per_group:
                    continue
                regular_pools = (
                    {
                        state_key: game_ids
                        for state_key, game_ids in state_buckets.items()
                        if state_key not in action_pair_pools
                    }
                    if cfg.min_pair_anchors_per_group
                    else dict(state_buckets)
                )
                if len(regular_pools) < regular_needed:
                    continue
                pools[t][action] = regular_pools
                pair_pools[t][action] = action_pair_pools
        self._game_pools = pools
        self._pair_game_pools = pair_pools
        for actions in self._game_pools.values():
            for state_buckets in actions.values():
                for game_ids in state_buckets.values():
                    self.rng.shuffle(game_ids)

        # Incremental viability bookkeeping.
        #
        # The naive __next__ rebuilds viable_by_t from scratch every call, an
        # O(sum_t |actions_t| * |state_buckets|) Python scan that dominates the
        # batch generation cost for large chunks (hundreds of thousands of ops
        # per batch). We maintain it incrementally instead:
        #
        #   _nonempty_buckets[t][action] = #state_buckets whose game_ids list is
        #                                  still non-empty
        #   _viable_actions[t]           = ordered set of actions with
        #                                  _nonempty_buckets[t][action] >= samples_per_group
        #   _viable_t                    = ordered set of t where
        #                                  len(_viable_actions[t]) >= groups_per_batch
        #
        # We use insertion-ordered dicts (dict.fromkeys / dict[K, None]) as the
        # ordered-set type so list(_viable_actions[t]) and list(_viable_t) match
        # the iteration order the naive implementation produced, which keeps
        # rng.choice / rng.sample byte-identical across versions. The slow path
        # is retained behind self._use_fast_viability so the rewrite is reversible
        # without code changes if a regression is ever suspected.
        self._use_fast_viability = True
        self._nonempty_buckets: dict[int, dict[int, int]] = {}
        self._viable_actions: dict[int, dict[int, None]] = {}
        self._viable_t: dict[int, None] = {}
        self._active_state_keys: dict[int, dict[int, list[bytes]]] = {}
        self._active_state_positions: dict[int, dict[int, dict[bytes, int]]] = {}
        for t, actions_pool in self._game_pools.items():
            self._nonempty_buckets[t] = {}
            viable_for_t: dict[int, None] = {}
            self._active_state_keys[t] = {}
            self._active_state_positions[t] = {}
            for action, state_buckets in actions_pool.items():
                active = [
                    state_key
                    for state_key, game_ids in state_buckets.items()
                    if game_ids
                ]
                self._active_state_keys[t][action] = active
                self._active_state_positions[t][action] = {
                    state_key: index for index, state_key in enumerate(active)
                }
                count = len(active)
                self._nonempty_buckets[t][action] = count
                if count >= regular_needed:
                    viable_for_t[action] = None
            self._viable_actions[t] = viable_for_t
            if len(viable_for_t) >= cfg.groups_per_batch:
                self._viable_t[t] = None
        if not self._viable_t:
            self.pairs.close()
            raise ValueError(
                f"Chunk {self.games_path.name} has no viable v8 groups "
                f"for G={cfg.groups_per_batch}, M={cfg.samples_per_group}, "
                f"minimum pair anchors/group={cfg.min_pair_anchors_per_group}, "
                f"t=[{cfg.t_min},{cfg.t_max}]"
            )

    def __iter__(self) -> Iterator[dict[str, object]]:
        return self

    def close(self) -> None:
        self.pairs.close()

    def __next__(self) -> dict[str, object]:
        if self.max_batches is not None and self._batches_emitted >= self.max_batches:
            self.close()
            raise StopIteration
        if self._use_fast_viability:
            viable_t_list = list(self._viable_t)
        else:
            viable_by_t = {
                t: [
                    action
                    for action, state_buckets in actions.items()
                    if sum(bool(game_ids) for game_ids in state_buckets.values())
                    >= (
                        self.cfg.samples_per_group
                        - self.cfg.min_pair_anchors_per_group
                    )
                ]
                for t, actions in self._game_pools.items()
            }
            viable_t_list = [
                t
                for t, actions in viable_by_t.items()
                if len(actions) >= self.cfg.groups_per_batch
            ]
        if not viable_t_list:
            self.close()
            raise StopIteration
        t = self.rng.choice(viable_t_list)
        if self._use_fast_viability:
            viable_actions_for_t = list(self._viable_actions[t])
        else:
            viable_actions_for_t = viable_by_t[t]
        selected_actions = self.rng.sample(
            viable_actions_for_t, self.cfg.groups_per_batch
        )

        contexts: list[list[int]] = []
        positive_targets: list[list[int]] = []
        hard_targets: list[list[list[int]]] = []
        hard_masks: list[list[bool]] = []
        group_ids: list[int] = []
        fallback_masks: list[bool] = []
        game_indices: list[int] = []

        for group_id, action in enumerate(selected_actions):
            state_buckets = self._game_pools[t][action]
            selected_pair_state_keys = self.rng.sample(
                list(self._pair_game_pools[t][action]),
                self.cfg.min_pair_anchors_per_group,
            )
            selected_regular_state_keys = self.rng.sample(
                self._active_state_keys[t][action],
                self.cfg.samples_per_group
                - self.cfg.min_pair_anchors_per_group,
            )
            selected_games = [
                self.rng.choice(self._pair_game_pools[t][action][state_key])
                for state_key in selected_pair_state_keys
            ]
            for state_key in selected_regular_state_keys:
                game_ids_list = state_buckets[state_key]
                selected_games.append(game_ids_list.pop())
                if not game_ids_list and self._use_fast_viability:
                    active = self._active_state_keys[t][action]
                    positions = self._active_state_positions[t][action]
                    removed_position = positions.pop(state_key)
                    last_state_key = active.pop()
                    if removed_position < len(active):
                        active[removed_position] = last_state_key
                        positions[last_state_key] = removed_position
                    # Bucket emptied: drop the bucket from the non-empty count
                    # and propagate to action/t viability if thresholds break.
                    self._nonempty_buckets[t][action] -= 1
                    if (
                        self._nonempty_buckets[t][action]
                        < (
                            self.cfg.samples_per_group
                            - self.cfg.min_pair_anchors_per_group
                        )
                        and action in self._viable_actions[t]
                    ):
                        del self._viable_actions[t][action]
                        if (
                            len(self._viable_actions[t])
                            < self.cfg.groups_per_batch
                            and t in self._viable_t
                        ):
                            del self._viable_t[t]
            for game_idx in selected_games:
                context = self.games[game_idx][:t]
                positive_refs = self.pairs.positive_refs(
                    self.games_path.name, game_idx, t
                )
                if positive_refs:
                    positive_prefix = self.pairs.resolve_prefix_tokens(
                        self.rng.choice(positive_refs)
                    )
                    fallback = False
                else:
                    positive_prefix = context
                    fallback = True
                if len(positive_prefix) != t:
                    raise ValueError("Positive partner prefix length does not match t")
                positive_target = [*positive_prefix, action]

                negative_refs = list(
                    self.pairs.hard_negative_refs(
                        self.games_path.name, game_idx, t
                    )
                )
                self.rng.shuffle(negative_refs)
                anchor_hard_targets: list[list[int]] = []
                anchor_hard_mask: list[bool] = []
                for ref in negative_refs[: self.cfg.hard_negatives_per_anchor]:
                    negative_prefix = self.pairs.resolve_prefix_tokens(ref)
                    if len(negative_prefix) != t:
                        raise ValueError("Hard-negative prefix length does not match t")
                    anchor_hard_targets.append([*negative_prefix, action])
                    anchor_hard_mask.append(True)
                while len(anchor_hard_targets) < self.cfg.hard_negatives_per_anchor:
                    anchor_hard_targets.append(positive_target)
                    anchor_hard_mask.append(False)

                contexts.append(context)
                positive_targets.append(positive_target)
                hard_targets.append(anchor_hard_targets)
                hard_masks.append(anchor_hard_mask)
                group_ids.append(group_id)
                fallback_masks.append(fallback)
                game_indices.append(game_idx)

        self._batches_emitted += 1
        batch_size = len(contexts)
        hard_count = sum(sum(mask) for mask in hard_masks)
        if self.cfg.hard_negatives_per_anchor:
            hard_target_tensor = torch.tensor(hard_targets, dtype=torch.long)
        else:
            hard_target_tensor = torch.empty(
                batch_size, 0, t + 1, dtype=torch.long
            )
        return {
            "t": t,
            "x_context": torch.tensor(contexts, dtype=torch.long),
            "x_positive_target": torch.tensor(positive_targets, dtype=torch.long),
            "x_hard_negative_targets": hard_target_tensor,
            "hard_negative_mask": torch.tensor(hard_masks, dtype=torch.bool),
            "group_ids": torch.tensor(group_ids, dtype=torch.long),
            "positive_fallback_mask": torch.tensor(fallback_masks, dtype=torch.bool),
            "game_indices": torch.tensor(game_indices, dtype=torch.long),
            "positive_fallback_rate": sum(fallback_masks) / max(1, batch_size),
            "mean_hard_negatives": hard_count / max(1, batch_size),
            "games_consumed": batch_size,
        }
