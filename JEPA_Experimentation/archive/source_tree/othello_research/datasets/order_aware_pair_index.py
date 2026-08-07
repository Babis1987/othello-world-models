"""Offline full-state pair index and runtime reader for order-aware JEPA."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
import json
import os
import sqlite3
import time

import numpy as np

from othello_research.datasets.dataset import load_chunk
from othello_research.datasets.move_mapping import build_mappings
from othello_research.othello.board import OthelloBoardState


INDEX_VERSION = 4
OCCURRENCE_INSERT_BATCH = 10_000

_FAST_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _fast_init_board(n: int) -> bytearray:
    """Initialize a flat bytearray board matching the OthelloBoardState convention.

    Encoding (matches ``state + 1`` in the reference implementation so packing
    produces byte-identical state keys):
        0 = white, 1 = empty, 2 = black.
    """
    cells = bytearray([1] * (n * n))
    mid = n // 2
    cells[(mid - 1) * n + (mid - 1)] = 0  # white
    cells[(mid - 1) * n + mid] = 2  # black
    cells[mid * n + (mid - 1)] = 2  # black
    cells[mid * n + mid] = 0  # white
    return cells


def _fast_find_flips(cells: bytearray, r: int, c: int, color: int, n: int) -> list[int]:
    opp = 2 - color
    to_flip: list[int] = []
    for dr, dc in _FAST_DIRECTIONS:
        buf: list[int] = []
        cr, cc = r + dr, c + dc
        while 0 <= cr < n and 0 <= cc < n:
            cell = cells[cr * n + cc]
            if cell == opp:
                buf.append(cr * n + cc)
                cr += dr
                cc += dc
            elif cell == color:
                if buf:
                    to_flip.extend(buf)
                break
            else:
                break
    return to_flip


def _fast_has_legal_move(cells: bytearray, color: int, n: int) -> bool:
    return any(
        cells[move] == 1
        and _fast_find_flips(cells, move // n, move % n, color, n)
        for move in range(n * n)
    )


def _fast_apply_known_legal(cells: bytearray, move: int, side: int, n: int) -> int:
    """Apply a known-legal dataset move; returns the new side-to-move."""
    r, c = divmod(move, n)
    if cells[r * n + c] != 1:
        raise ValueError(f"Cell ({r},{c}) is already occupied")
    color = side
    flips = _fast_find_flips(cells, r, c, color, n)
    if not flips:
        if _fast_has_legal_move(cells, color, n):
            raise ValueError(f"Move {move} is illegal for the current player")
        color = 2 - side
        flips = _fast_find_flips(cells, r, c, color, n)
    if not flips:
        raise ValueError(f"Illegal move {move}")
    for f in flips:
        cells[f] = color
    cells[r * n + c] = color
    return 2 - color


def _fast_pack_cells(cells: bytearray, n: int) -> bytes:
    size = n * n
    padded = ((size + 3) // 4) * 4
    if padded != size:
        view = bytes(cells) + bytes([1] * (padded - size))
    else:
        view = cells
    out = bytearray(padded // 4)
    for i in range(len(out)):
        base = 4 * i
        out[i] = view[base] | (view[base + 1] << 2) | (view[base + 2] << 4) | (view[base + 3] << 6)
    return bytes(out)


def _fast_occupancy_key(cells: bytearray, n: int) -> bytes:
    size = n * n
    out = bytearray((size + 7) // 8)
    for i in range(size):
        if cells[i] != 1:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def _fast_legal_actions_key(
    cells: bytearray,
    current_side: int,
    n: int,
) -> bytes:
    """Encode actions legal for the effective side-to-move as a bitset."""
    size = n * n
    out = bytearray((size + 7) // 8)
    side = current_side
    legal = [
        move
        for move in range(size)
        if cells[move] == 1
        and _fast_find_flips(cells, move // n, move % n, side, n)
    ]
    if not legal:
        side = 2 - side
        legal = [
            move
            for move in range(size)
            if cells[move] == 1
            and _fast_find_flips(cells, move // n, move % n, side, n)
        ]
    for move in legal:
        out[move >> 3] |= 1 << (move & 7)
    return bytes(out)


def _action_is_in_key(actions_key: bytes, action: int) -> bool:
    return bool(actions_key[action >> 3] & (1 << (action & 7)))


def _occupancy_positions_from_key(key: bytes, board_size: int) -> tuple[int, ...]:
    """Decode occupied cell positions from the compact occupancy bitset."""
    size = board_size * board_size
    return tuple(
        pos
        for pos in range(size)
        if key[pos >> 3] & (1 << (pos & 7))
    )


def _occupancy_jaccard(first: bytes, second: bytes) -> float:
    """Return Jaccard overlap between two occupancy bitsets."""
    intersection = sum((a & b).bit_count() for a, b in zip(first, second))
    union = sum((a | b).bit_count() for a, b in zip(first, second))
    return intersection / union if union else 1.0


def _surface_minhash_signature(
    positions: Sequence[int],
    *,
    hash_count: int = 16,
) -> tuple[int, ...]:
    """Small deterministic MinHash signature for near-surface candidate mining."""
    if not positions:
        return tuple(0 for _ in range(hash_count))
    prime = 4_294_967_291
    signature: list[int] = []
    for seed in range(hash_count):
        a = 1_103_515_245 + 2_654_435_761 * (seed + 1)
        b = 12_345 + 97_531 * (seed + 1)
        signature.append(min(((a * (pos + 1) + b) % prime) for pos in positions))
    return tuple(signature)


def _surface_band_keys(
    occupancy_key_value: bytes,
    *,
    board_size: int,
    band_size: int = 4,
) -> tuple[tuple[int, ...], ...]:
    positions = _occupancy_positions_from_key(occupancy_key_value, board_size)
    signature = _surface_minhash_signature(positions)
    return tuple(
        signature[start : start + band_size]
        for start in range(0, len(signature), band_size)
    )


def _build_surface_hard_negative_pairs_for_chunk(
    occurrences: Sequence[tuple],
    *,
    board_size: int,
    threshold: float,
    max_pairs_per_anchor: int,
    max_bucket_scan: int = 512,
) -> list[tuple]:
    """Return same-chunk near-surface hard negatives for indexed occurrences.

    Exact same-occupancy hard negatives are still built globally by SQLite. This
    optional pass densifies negatives by adding same-ply, same-side candidates
    whose occupied-cell Jaccard overlap is above ``threshold`` but whose full
    board states differ. Candidate discovery uses deterministic MinHash bands
    and then verifies exact Jaccard before inserting a pair.
    """
    if not occurrences:
        return []
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(occurrences):
        _chunk_id, _game_idx, t, _action, state_key, *_rest = row
        side_key = state_key[-1]
        groups.setdefault((int(t), int(side_key)), []).append(index)

    hard_rows: list[tuple] = []
    for (_t, _side), indices in groups.items():
        bands: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        band_keys_by_index: dict[int, tuple[tuple[int, ...], ...]] = {}
        for index in indices:
            occupancy = occurrences[index][5]
            band_keys = _surface_band_keys(occupancy, board_size=board_size)
            band_keys_by_index[index] = band_keys
            for band_id, band_key in enumerate(band_keys):
                bands.setdefault((band_id, band_key), []).append(index)

        for anchor_index in indices:
            (
                anchor_chunk_id,
                anchor_game,
                anchor_t,
                anchor_action,
                anchor_state,
                anchor_occupancy,
                _anchor_legal_actions,
                _anchor_prefix,
            ) = occurrences[anchor_index]
            seen: set[int] = {anchor_index}
            inserted = 0
            for band_id, band_key in enumerate(band_keys_by_index[anchor_index]):
                for candidate_index in bands.get((band_id, band_key), [])[:max_bucket_scan]:
                    if candidate_index in seen:
                        continue
                    seen.add(candidate_index)
                    (
                        negative_chunk_id,
                        negative_game,
                        negative_t,
                        _negative_action,
                        negative_state,
                        negative_occupancy,
                        negative_legal_actions,
                        negative_prefix,
                    ) = occurrences[candidate_index]
                    if negative_state == anchor_state:
                        continue
                    if not _action_is_in_key(negative_legal_actions, int(anchor_action)):
                        continue
                    if _occupancy_jaccard(anchor_occupancy, negative_occupancy) < threshold:
                        continue
                    hard_rows.append(
                        (
                            int(anchor_chunk_id),
                            int(anchor_game),
                            int(anchor_t),
                            int(negative_chunk_id),
                            int(negative_game),
                            int(negative_t),
                            negative_prefix,
                        )
                    )
                    inserted += 1
                    if inserted >= max_pairs_per_anchor:
                        break
                if inserted >= max_pairs_per_anchor:
                    break
    return hard_rows


def _fast_state_key_for_next_action(
    cells: bytearray, next_action: int, current_side: int, n: int
) -> bytes:
    """Match _full_board_state_key_for_next_action byte-for-byte using the fast board."""
    r, c = divmod(next_action, n)
    side = current_side
    flips = _fast_find_flips(cells, r, c, side, n)
    if not flips:
        if _fast_has_legal_move(cells, side, n):
            raise ValueError(
                f"Next action {next_action} is illegal for the current player"
            )
        side = 2 - side
        flips = _fast_find_flips(cells, r, c, side, n)
        if not flips:
            raise ValueError(f"Next action {next_action} is illegal for both players")
    return _fast_pack_cells(cells, n) + bytes([1 if side == 2 else 0])


def effective_side_to_move(board: OthelloBoardState) -> int:
    """Return the player that will make the next represented move."""
    side = board.next_hand_color
    if board.get_valid_moves():
        return side
    board.next_hand_color *= -1
    opponent_has_move = bool(board.get_valid_moves())
    board.next_hand_color *= -1
    return -side if opponent_has_move else side


def full_board_state_key(board: OthelloBoardState) -> bytes:
    """Encode all cell colors exactly in two bits plus effective side-to-move."""
    cells = _packed_cell_colors(board)
    side = 1 if effective_side_to_move(board) == 1 else 0
    return cells + bytes([side])


def occupancy_key(board: OthelloBoardState) -> bytes:
    """Encode occupied versus empty cells only."""
    return np.packbits(
        board.state.reshape(-1) != 0,
        bitorder="little",
    ).tobytes()


def _packed_cell_colors(board: OthelloBoardState) -> bytes:
    values = (board.state.reshape(-1) + 1).astype(np.uint8, copy=False)
    padded_size = ((values.size + 3) // 4) * 4
    if padded_size != values.size:
        values = np.pad(values, (0, padded_size - values.size), constant_values=1)
    packed = (
        values[0::4]
        | (values[1::4] << 2)
        | (values[2::4] << 4)
        | (values[3::4] << 6)
    )
    return packed.tobytes()


def _full_board_state_key_for_next_action(
    board: OthelloBoardState,
    next_action: int,
) -> bytes:
    """Encode the exact state while inferring forced passes from the next move."""
    row, col = divmod(next_action, board.n)
    side = board.next_hand_color
    if not board._find_flips(row, col, side):
        side *= -1
        if not board._find_flips(row, col, side):
            raise ValueError(f"Next action {next_action} is illegal for both players")
    return _packed_cell_colors(board) + bytes([1 if side == 1 else 0])


def _fast_umpire_known_legal_game(board: OthelloBoardState, move: int) -> None:
    """Apply a known-legal dataset move without scanning all 64 legal moves."""
    if not (0 <= move < board.n * board.n):
        raise ValueError(f"Invalid move: {move}")
    row, col = divmod(move, board.n)
    if board.state[row, col] != 0:
        raise ValueError(f"Cell ({row},{col}) is already occupied")
    color = board.next_hand_color
    to_flip = board._find_flips(row, col, color)
    if not to_flip:
        color *= -1
        to_flip = board._find_flips(row, col, color)
    if not to_flip:
        raise ValueError(f"Illegal move {move} in indexed dataset")
    for flip_row, flip_col in to_flip:
        board.state[flip_row, flip_col] *= -1
    board.state[row, col] = color
    board.next_hand_color = -color
    board.history.append(move)


def indexed_positions(
    game_length: int,
    *,
    t_min: int,
    t_max: int,
    positions_per_game: int,
    game_idx: int,
    sampling_seed: int,
) -> tuple[int, ...]:
    """Return deterministic, balanced prefix positions selected for one game."""
    last_t = min(t_max, game_length - 1)
    if last_t < t_min:
        return ()
    available = last_t - t_min + 1
    if positions_per_game <= 0 or positions_per_game >= available:
        return tuple(range(t_min, last_t + 1))
    start = (game_idx + sampling_seed) % available
    offsets = {
        (start + (sample_idx * available) // positions_per_game) % available
        for sample_idx in range(positions_per_game)
    }
    return tuple(sorted(t_min + offset for offset in offsets))


def replay_prefix(moves: Sequence[int], board_size: int) -> OthelloBoardState:
    """Replay a raw-move prefix with the repository Othello engine."""
    board = OthelloBoardState(n=board_size)
    board.update(list(moves))
    return board


def classify_order_pair(
    first: Sequence[int],
    second: Sequence[int],
    board_size: int,
) -> str:
    """Classify two equal-length legal prefixes by full state and occupancy."""
    if len(first) != len(second):
        raise ValueError("Order-pair prefixes must have the same length")
    first_board = replay_prefix(first, board_size)
    second_board = replay_prefix(second, board_size)
    if full_board_state_key(first_board) == full_board_state_key(second_board):
        return "positive"
    if occupancy_key(first_board) == occupancy_key(second_board):
        return "hard_negative"
    return "unmatched"


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE occurrences (
            id INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL,
            game_idx INTEGER NOT NULL,
            t INTEGER NOT NULL,
            next_action_raw INTEGER NOT NULL,
            state_key BLOB NOT NULL,
            occupancy_key BLOB NOT NULL,
            legal_actions_key BLOB NOT NULL,
            prefix_key BLOB NOT NULL
        );
        CREATE TABLE positive_pairs (
            anchor_chunk_id INTEGER NOT NULL,
            anchor_game INTEGER NOT NULL,
            t INTEGER NOT NULL,
            partner_chunk_id INTEGER,
            partner_game INTEGER,
            partner_t INTEGER NOT NULL,
            partner_prefix BLOB
        );
        CREATE TABLE hard_negative_pairs (
            anchor_chunk_id INTEGER NOT NULL,
            anchor_game INTEGER NOT NULL,
            t INTEGER NOT NULL,
            negative_chunk_id INTEGER,
            negative_game INTEGER,
            negative_t INTEGER NOT NULL,
            negative_prefix BLOB
        );
        """
    )


def _create_occurrence_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX occurrences_state
            ON occurrences(state_key, t);
        CREATE INDEX occurrences_occupancy_action
            ON occurrences(occupancy_key, t, next_action_raw);
        CREATE INDEX occurrences_chunk
            ON occurrences(chunk_id, game_idx, t);
        """
    )


def _create_pair_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX positive_anchor
            ON positive_pairs(anchor_chunk_id, anchor_game, t);
        CREATE INDEX hard_negative_anchor
            ON hard_negative_pairs(anchor_chunk_id, anchor_game, t);
        """
    )


def _raw_prefix_blob(prefix: Sequence[int]) -> bytes:
    if any(move < 0 or move > 255 for move in prefix):
        raise ValueError("Raw prefix moves must fit in one byte")
    return bytes(prefix)


def _action_is_legal(board: OthelloBoardState, action: int) -> bool:
    if board.get_valid_moves():
        return action in board.get_valid_moves()
    board.next_hand_color *= -1
    legal = action in board.get_valid_moves()
    board.next_hand_color *= -1
    return legal


def _constructive_prefixes(prefix: Sequence[int]) -> Iterable[list[int]]:
    """Yield the two local order perturbations requested by the v8 design."""
    if len(prefix) >= 2:
        adjacent = list(prefix)
        adjacent[-2], adjacent[-1] = adjacent[-1], adjacent[-2]
        yield adjacent
    if len(prefix) >= 3:
        same_player = list(prefix)
        same_player[-3], same_player[-1] = same_player[-1], same_player[-3]
        yield same_player


def _enumerate_constructive_pairs(
    *,
    chunk_id: int,
    game_idx: int,
    prefix: Sequence[int],
    next_action: int,
    board_size: int,
) -> tuple[list[tuple], list[tuple]]:
    """Enumerate constructive perturbations as (positive_rows, hard_negative_rows).

    Picklable, top-level helper used by worker processes. Each row matches the
    ``positive_pairs`` / ``hard_negative_pairs`` insertion tuple shape.
    """
    anchor_board = replay_prefix(prefix, board_size)
    anchor_state = full_board_state_key(anchor_board)
    anchor_occupancy = occupancy_key(anchor_board)
    positives: list[tuple] = []
    hard_negatives: list[tuple] = []
    for candidate in _constructive_prefixes(prefix):
        try:
            candidate_board = replay_prefix(candidate, board_size)
        except AssertionError:
            continue
        candidate_blob = _raw_prefix_blob(candidate)
        if full_board_state_key(candidate_board) == anchor_state:
            positives.append(
                (chunk_id, game_idx, len(prefix), None, None, len(prefix), candidate_blob)
            )
        elif (
            occupancy_key(candidate_board) == anchor_occupancy
            and _action_is_legal(candidate_board, next_action)
        ):
            hard_negatives.append(
                (chunk_id, game_idx, len(prefix), None, None, len(prefix), candidate_blob)
            )
    return positives, hard_negatives


def _enumerate_constructive_pairs_fast(
    *,
    chunk_id: int,
    game_idx: int,
    prefix: Sequence[int],
    next_action: int,
    board_size: int,
    anchor_state_key: bytes,
    anchor_occupancy_key: bytes,
    pre_move_snapshots: dict[int, tuple[bytes, int]],
) -> tuple[list[tuple], list[tuple]]:
    """Classify local reorderings by replaying only their changed suffix.

    ``pre_move_snapshots[i]`` is the board and side immediately before move
    ``prefix[i]``. The adjacent and same-player perturbations only alter the
    final two or three moves, so replaying the full prefix is unnecessary.
    """
    positives: list[tuple] = []
    hard_negatives: list[tuple] = []
    suffixes: list[tuple[int, list[int], list[int]]] = []
    t = len(prefix)
    if t >= 2:
        adjacent = list(prefix)
        adjacent[-2], adjacent[-1] = adjacent[-1], adjacent[-2]
        suffixes.append((t - 2, adjacent[-2:], adjacent))
    if t >= 3:
        same_player = list(prefix)
        same_player[-3], same_player[-1] = same_player[-1], same_player[-3]
        suffixes.append((t - 3, same_player[-3:], same_player))

    for start, suffix, candidate in suffixes:
        snapshot = pre_move_snapshots.get(start)
        if snapshot is None:
            raise ValueError(f"Missing constructive snapshot before move {start}")
        candidate_cells = bytearray(snapshot[0])
        candidate_side = snapshot[1]
        try:
            for move in suffix:
                candidate_side = _fast_apply_known_legal(
                    candidate_cells,
                    move,
                    candidate_side,
                    board_size,
                )
            candidate_state_key = _fast_state_key_for_next_action(
                candidate_cells,
                next_action,
                candidate_side,
                board_size,
            )
        except ValueError:
            continue

        candidate_blob = _raw_prefix_blob(candidate)
        row = (
            chunk_id,
            game_idx,
            t,
            None,
            None,
            t,
            candidate_blob,
        )
        if candidate_state_key == anchor_state_key:
            positives.append(row)
        elif _fast_occupancy_key(candidate_cells, board_size) == anchor_occupancy_key:
            hard_negatives.append(row)
    return positives, hard_negatives


def _extract_occurrences_for_chunk(task: tuple) -> dict:
    """Worker entry point: process one chunk and return occurrence rows.

    Runs in a separate process so the Python replay loop scales across cores.
    The return value is a dict serialized back to the parent for SQLite insert.
    """
    (
        chunk_id,
        chunk_path_str,
        chunk_name,
        board_size,
        t_min,
        t_max,
        positions_per_game,
        sampling_seed,
        use_constructive_pairs,
        use_surface_hard_negatives,
        surface_jaccard_threshold,
        surface_max_pairs_per_anchor,
    ) = task

    games = load_chunk(chunk_path_str)
    n_games = len(games)
    n_candidate_occurrences = 0
    n_replayed_moves = 0
    occurrences: list[tuple] = []
    indexed_t_counts: list[int] = []
    constructive_positives: list[tuple] = []
    constructive_hard_negatives: list[tuple] = []
    surface_hard_negatives: list[tuple] = []

    for game_idx, game in enumerate(games):
        last_t = min(t_max, len(game) - 1)
        if last_t >= t_min:
            n_candidate_occurrences += last_t - t_min + 1
        selected_positions = indexed_positions(
            len(game),
            t_min=t_min,
            t_max=t_max,
            positions_per_game=positions_per_game,
            game_idx=game_idx,
            sampling_seed=sampling_seed,
        )
        if not selected_positions:
            continue
        selected = set(selected_positions)
        snapshot_positions = {
            start
            for t in selected_positions
            for start in (t - 2, t - 3)
            if start >= 0
        }
        pre_move_snapshots: dict[int, tuple[bytes, int]] = {}
        final_selected_t = selected_positions[-1]
        cells = _fast_init_board(board_size)
        side = 2  # black moves first (matches OthelloBoardState init: next_hand_color=1)
        for move_index, move in enumerate(game[:final_selected_t]):
            if move_index in snapshot_positions:
                pre_move_snapshots[move_index] = (bytes(cells), side)
            side = _fast_apply_known_legal(cells, move, side, board_size)
            n_replayed_moves += 1
            t = move_index + 1
            if t not in selected:
                continue
            prefix = game[:t]
            state_key = _fast_state_key_for_next_action(
                cells, game[t], side, board_size
            )
            occ_key = _fast_occupancy_key(cells, board_size)
            legal_actions_key = _fast_legal_actions_key(cells, side, board_size)
            if not _action_is_in_key(legal_actions_key, game[t]):
                raise ValueError(
                    f"Dataset next action {game[t]} is not legal at "
                    f"{chunk_name}:{game_idx}@{t}"
                )
            occurrences.append(
                (
                    chunk_id,
                    game_idx,
                    t,
                    game[t],
                    state_key,
                    occ_key,
                    legal_actions_key,
                    _raw_prefix_blob(prefix),
                )
            )
            indexed_t_counts.append(t)
            if use_constructive_pairs:
                pos_rows, neg_rows = _enumerate_constructive_pairs_fast(
                    chunk_id=chunk_id,
                    game_idx=game_idx,
                    prefix=prefix,
                    next_action=game[t],
                    board_size=board_size,
                    anchor_state_key=state_key,
                    anchor_occupancy_key=occ_key,
                    pre_move_snapshots=pre_move_snapshots,
                )
                constructive_positives.extend(pos_rows)
                constructive_hard_negatives.extend(neg_rows)

    if use_surface_hard_negatives:
        surface_hard_negatives = _build_surface_hard_negative_pairs_for_chunk(
            occurrences,
            board_size=board_size,
            threshold=surface_jaccard_threshold,
            max_pairs_per_anchor=surface_max_pairs_per_anchor,
        )

    return {
        "chunk_id": chunk_id,
        "chunk_name": chunk_name,
        "n_games": n_games,
        "n_candidate_occurrences": n_candidate_occurrences,
        "n_replayed_moves": n_replayed_moves,
        "occurrences": occurrences,
        "indexed_t_counts": indexed_t_counts,
        "constructive_positives": constructive_positives,
        "constructive_hard_negatives": constructive_hard_negatives,
        "surface_hard_negatives": surface_hard_negatives,
    }


def _insert_constructive_pairs(
    connection: sqlite3.Connection,
    *,
    chunk_id: int,
    game_idx: int,
    prefix: Sequence[int],
    next_action: int,
    board_size: int,
) -> tuple[int, int]:
    anchor_board = replay_prefix(prefix, board_size)
    anchor_state = full_board_state_key(anchor_board)
    anchor_occupancy = occupancy_key(anchor_board)
    positives = 0
    hard_negatives = 0
    for candidate in _constructive_prefixes(prefix):
        try:
            candidate_board = replay_prefix(candidate, board_size)
        except AssertionError:
            continue
        candidate_blob = _raw_prefix_blob(candidate)
        if full_board_state_key(candidate_board) == anchor_state:
            connection.execute(
                "INSERT INTO positive_pairs VALUES (?, ?, ?, NULL, NULL, ?, ?)",
                (chunk_id, game_idx, len(prefix), len(prefix), candidate_blob),
            )
            positives += 1
        elif (
            occupancy_key(candidate_board) == anchor_occupancy
            and _action_is_legal(candidate_board, next_action)
        ):
            connection.execute(
                "INSERT INTO hard_negative_pairs VALUES (?, ?, ?, NULL, NULL, ?, ?)",
                (chunk_id, game_idx, len(prefix), len(prefix), candidate_blob),
            )
            hard_negatives += 1
    return positives, hard_negatives


def _build_positive_pairs(
    connection: sqlite3.Connection,
    max_pairs_per_anchor: int,
) -> int:
    rows = connection.execute(
        """
        SELECT state_key, t, chunk_id, game_idx, prefix_key
        FROM occurrences
        ORDER BY state_key, t, id
        """
    )
    count = 0
    group_key: tuple[bytes, int] | None = None
    group: list[tuple[int, int, bytes]] = []
    pair_rows: list[tuple[int, int, int, int, int, int, bytes]] = []

    def write_pairs() -> None:
        if not pair_rows:
            return
        connection.executemany(
            "INSERT INTO positive_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
            pair_rows,
        )
        pair_rows.clear()

    def flush() -> int:
        inserted = 0
        by_prefix: OrderedDict[bytes, list[tuple[int, int]]] = OrderedDict()
        for chunk_id, game_idx, prefix_key in group:
            by_prefix.setdefault(prefix_key, []).append((chunk_id, game_idx))
        if len(by_prefix) < 2:
            return 0
        prefixes = list(by_prefix)
        for prefix_index, prefix_key in enumerate(prefixes):
            candidates: list[tuple[int, int, bytes]] = []
            for offset in range(1, len(prefixes)):
                candidate_prefix = prefixes[(prefix_index + offset) % len(prefixes)]
                candidates.extend(
                    (chunk_id, game_idx, candidate_prefix)
                    for chunk_id, game_idx in by_prefix[candidate_prefix]
                )
            for anchor_chunk_id, anchor_game in by_prefix[prefix_key]:
                for (
                    partner_chunk_id,
                    partner_game,
                    partner_prefix,
                ) in candidates[:max_pairs_per_anchor]:
                    pair_rows.append(
                        (
                            anchor_chunk_id,
                            anchor_game,
                            group_key[1],
                            partner_chunk_id,
                            partner_game,
                            group_key[1],
                            partner_prefix,
                        )
                    )
                    if len(pair_rows) >= OCCURRENCE_INSERT_BATCH:
                        write_pairs()
                    inserted += 1
        return inserted

    for state_key, t, chunk_id, game_idx, prefix_key in rows:
        key = (state_key, int(t))
        if group_key is not None and key != group_key:
            count += flush()
            group = []
        group_key = key
        group.append((int(chunk_id), int(game_idx), prefix_key))
    if group:
        count += flush()
    write_pairs()
    return count


def _build_hard_negative_pairs(
    connection: sqlite3.Connection,
    max_pairs_per_anchor: int,
) -> int:
    rows = connection.execute(
        """
        SELECT occupancy_key, t, next_action_raw, state_key, chunk_id, game_idx,
               legal_actions_key, prefix_key
        FROM occurrences
        ORDER BY occupancy_key, t, id
        """
    )
    count = 0
    group_key: tuple[bytes, int] | None = None
    group: list[tuple[int, bytes, int, int, bytes, bytes]] = []
    pair_rows: list[tuple[int, int, int, int, int, int, bytes]] = []

    def write_pairs() -> None:
        if not pair_rows:
            return
        connection.executemany(
            "INSERT INTO hard_negative_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
            pair_rows,
        )
        pair_rows.clear()

    def flush() -> int:
        inserted = 0
        by_state: OrderedDict[
            bytes,
            list[tuple[int, int, int, bytes, bytes]],
        ] = OrderedDict()
        for action, state_key, chunk_id, game_idx, legal_actions_key, prefix_key in group:
            by_state.setdefault(state_key, []).append(
                (action, chunk_id, game_idx, legal_actions_key, prefix_key)
            )
        if len(by_state) < 2:
            return 0
        states = list(by_state)
        for state_index, state_key in enumerate(states):
            candidates: list[tuple[int, int, bytes, bytes]] = []
            for offset in range(1, len(states)):
                candidates.extend(
                    (
                        chunk_id,
                        game_idx,
                        legal_actions_key,
                        prefix_key,
                    )
                    for (
                        _action,
                        chunk_id,
                        game_idx,
                        legal_actions_key,
                        prefix_key,
                    ) in by_state[states[(state_index + offset) % len(states)]]
                )
            for (
                anchor_action,
                anchor_chunk_id,
                anchor_game,
                _anchor_legal_actions,
                _anchor_prefix,
            ) in by_state[state_key]:
                legal_candidates = (
                    candidate
                    for candidate in candidates
                    if _action_is_in_key(candidate[2], anchor_action)
                )
                for (
                    negative_chunk_id,
                    negative_game,
                    _negative_legal_actions,
                    negative_prefix,
                ) in islice(legal_candidates, max_pairs_per_anchor):
                    pair_rows.append(
                        (
                            anchor_chunk_id,
                            anchor_game,
                            group_key[1],
                            negative_chunk_id,
                            negative_game,
                            group_key[1],
                            negative_prefix,
                        )
                    )
                    if len(pair_rows) >= OCCURRENCE_INSERT_BATCH:
                        write_pairs()
                    inserted += 1
        return inserted

    for (
        occupancy,
        t,
        action,
        state_key,
        chunk_id,
        game_idx,
        legal_actions_key,
        prefix_key,
    ) in rows:
        key = (occupancy, int(t))
        if group_key is not None and key != group_key:
            count += flush()
            group = []
        group_key = key
        group.append(
            (
                int(action),
                state_key,
                int(chunk_id),
                int(game_idx),
                legal_actions_key,
                prefix_key,
            )
        )
    if group:
        count += flush()
    write_pairs()
    return count


def _histogram(connection: sqlite3.Connection, table: str) -> dict[int, int]:
    return {
        int(t): int(count)
        for t, count in connection.execute(
            f"SELECT t, COUNT(*) FROM {table} GROUP BY t ORDER BY t"
        )
    }


def _distinct_anchor_count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT anchor_chunk_id, anchor_game, t FROM {table}
        )
        """
    ).fetchone()
    return int(row[0])


def build_pair_index(
    chunk_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    board_size: int = 8,
    t_min: int = 4,
    t_max: int = 40,
    max_pairs_per_anchor: int = 1,
    use_constructive_pairs: bool = False,
    use_surface_hard_negatives: bool = False,
    surface_jaccard_threshold: float = 0.9,
    surface_max_pairs_per_anchor: int = 1,
    positions_per_game: int = 0,
    sampling_seed: int = 42,
    overwrite: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    """Build a compact streaming SQLite pair index and return coverage stats.

    ``positions_per_game=0`` preserves exhaustive indexing. A positive value
    selects that many balanced prefix positions per game, which is the practical
    mode for full-dataset v8 training.

    ``workers`` controls multiprocessing fanout for the per-chunk replay/keying
    phase. ``workers <= 1`` runs in-process (test-friendly). With ``workers > 1``
    chunks are dispatched via ``ProcessPoolExecutor.map``; results are merged in
    submission order so the SQLite row order (and pair selection under
    ``max_pairs_per_anchor``) is deterministic regardless of completion order.
    """
    if t_min < 1 or t_max < t_min:
        raise ValueError("Expected 1 <= t_min <= t_max")
    if max_pairs_per_anchor < 1:
        raise ValueError("max_pairs_per_anchor must be positive")
    if positions_per_game < 0:
        raise ValueError("positions_per_game must be non-negative")
    if not 0.0 < surface_jaccard_threshold <= 1.0:
        raise ValueError("surface_jaccard_threshold must be in (0, 1]")
    if surface_max_pairs_per_anchor < 1:
        raise ValueError("surface_max_pairs_per_anchor must be positive")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    output = Path(output_path)
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    connection = _connect(output)
    _create_schema(connection)
    metadata = {
        "version": INDEX_VERSION,
        "board_size": board_size,
        "t_min": t_min,
        "t_max": t_max,
        "use_constructive_pairs": use_constructive_pairs,
        "use_surface_hard_negatives": use_surface_hard_negatives,
        "surface_jaccard_threshold": surface_jaccard_threshold,
        "surface_max_pairs_per_anchor": surface_max_pairs_per_anchor,
        "positions_per_game": positions_per_game,
        "sampling_seed": sampling_seed,
        "embedded_pair_prefixes": True,
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in metadata.items()],
    )

    tasks: list[tuple] = []
    for chunk_id, chunk_path_value in enumerate(chunk_paths, start=1):
        chunk_path = Path(chunk_path_value)
        connection.execute(
            "INSERT INTO chunks(id, name) VALUES (?, ?)",
            (chunk_id, chunk_path.name),
        )
        tasks.append(
            (
                chunk_id,
                str(chunk_path),
                chunk_path.name,
                board_size,
                t_min,
                t_max,
                positions_per_game,
                sampling_seed,
                use_constructive_pairs,
                use_surface_hard_negatives,
                surface_jaccard_threshold,
                surface_max_pairs_per_anchor,
            )
        )
    connection.commit()

    n_games = 0
    n_occurrences = 0
    n_candidate_occurrences = 0
    n_replayed_moves = 0
    indexed_t_histogram: Counter[int] = Counter()
    constructive_positive_count = 0
    constructive_hard_negative_count = 0
    surface_hard_negative_count = 0

    n_chunks_total = len(tasks)
    t_start = time.monotonic()

    if workers <= 1 or n_chunks_total <= 1:
        results_iter: Iterable[dict] = (
            _extract_occurrences_for_chunk(task) for task in tasks
        )
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results_iter = executor.map(_extract_occurrences_for_chunk, tasks)

    try:
        for done, result in enumerate(results_iter, start=1):
            n_games += result["n_games"]
            n_candidate_occurrences += result["n_candidate_occurrences"]
            n_replayed_moves += result["n_replayed_moves"]
            n_occurrences += len(result["occurrences"])
            indexed_t_histogram.update(result["indexed_t_counts"])

            occurrences = result["occurrences"]
            for batch_start in range(0, len(occurrences), OCCURRENCE_INSERT_BATCH):
                connection.executemany(
                    "INSERT INTO occurrences("
                    "chunk_id, game_idx, t, next_action_raw, state_key, occupancy_key, "
                    "legal_actions_key, prefix_key"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    occurrences[batch_start : batch_start + OCCURRENCE_INSERT_BATCH],
                )

            if result["constructive_positives"]:
                connection.executemany(
                    "INSERT INTO positive_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    result["constructive_positives"],
                )
                constructive_positive_count += len(result["constructive_positives"])
            if result["constructive_hard_negatives"]:
                connection.executemany(
                    "INSERT INTO hard_negative_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    result["constructive_hard_negatives"],
                )
                constructive_hard_negative_count += len(
                    result["constructive_hard_negatives"]
                )
            if result["surface_hard_negatives"]:
                connection.executemany(
                    "INSERT INTO hard_negative_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    result["surface_hard_negatives"],
                )
                surface_hard_negative_count += len(result["surface_hard_negatives"])

            connection.commit()

            elapsed = time.monotonic() - t_start
            eta = (elapsed / done) * (n_chunks_total - done) if done else 0.0
            print(
                f"[v8 build] {done:4d}/{n_chunks_total} {result['chunk_name']:<32s} "
                f"occ={len(occurrences):>7d}  "
                f"elapsed={elapsed/60:6.2f}min  ETA={eta/60:6.2f}min",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    _create_occurrence_indexes(connection)
    natural_positive_count = _build_positive_pairs(connection, max_pairs_per_anchor)
    natural_hard_negative_count = _build_hard_negative_pairs(
        connection, max_pairs_per_anchor
    )
    _create_pair_indexes(connection)
    connection.commit()
    positive_anchor_count = _distinct_anchor_count(connection, "positive_pairs")
    hard_negative_anchor_count = _distinct_anchor_count(
        connection, "hard_negative_pairs"
    )
    positive_count = natural_positive_count + constructive_positive_count
    hard_negative_count = (
        natural_hard_negative_count + constructive_hard_negative_count
        + surface_hard_negative_count
    )
    stats = {
        "index_path": str(output),
        "n_chunks": len(chunk_paths),
        "n_games": n_games,
        "n_occurrences": n_occurrences,
        "n_candidate_occurrences": n_candidate_occurrences,
        "sampling_fraction": n_occurrences / max(1, n_candidate_occurrences),
        "positions_per_game": positions_per_game,
        "sampling_seed": sampling_seed,
        "use_surface_hard_negatives": use_surface_hard_negatives,
        "surface_jaccard_threshold": surface_jaccard_threshold,
        "surface_max_pairs_per_anchor": surface_max_pairs_per_anchor,
        "n_replayed_moves": n_replayed_moves,
        "indexed_t_histogram": dict(sorted(indexed_t_histogram.items())),
        "positive_count": positive_count,
        "hard_negative_count": hard_negative_count,
        "positive_anchor_count": positive_anchor_count,
        "hard_negative_anchor_count": hard_negative_anchor_count,
        "estimated_positive_fallback_rate": (
            1.0 - positive_anchor_count / max(1, n_occurrences)
        ),
        "estimated_mean_hard_negatives_per_anchor": (
            hard_negative_count / max(1, n_occurrences)
        ),
        "natural_positive_count": natural_positive_count,
        "natural_hard_negative_count": natural_hard_negative_count,
        "constructive_positive_count": constructive_positive_count,
        "constructive_hard_negative_count": constructive_hard_negative_count,
        "surface_hard_negative_count": surface_hard_negative_count,
        "positive_t_histogram": _histogram(connection, "positive_pairs"),
        "hard_negative_t_histogram": _histogram(
            connection, "hard_negative_pairs"
        ),
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('coverage_stats', ?)",
        (json.dumps(stats),),
    )
    connection.commit()
    connection.close()
    return stats


class OrderAwarePairIndex:
    """Read-only runtime access to precomputed pairs; never invokes the engine."""

    def __init__(
        self,
        index_path: str | Path,
        data_dir: str | Path,
        *,
        board_size: int,
        chunk_cache_size: int = 4,
    ) -> None:
        self.index_path = Path(index_path)
        self.data_dir = Path(data_dir)
        self.board_size = board_size
        self.chunk_cache_size = max(1, int(chunk_cache_size))
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        self.connection = sqlite3.connect(str(self.index_path))
        metadata = {
            key: json.loads(value)
            for key, value in self.connection.execute("SELECT key, value FROM metadata")
        }
        required_metadata = {"version", "board_size", "t_min", "t_max"}
        missing_metadata = required_metadata - metadata.keys()
        if missing_metadata:
            raise ValueError(
                f"Pair index is missing metadata: {sorted(missing_metadata)}"
            )
        indexed_board_size = int(metadata["board_size"])
        if int(metadata["version"]) != INDEX_VERSION:
            raise ValueError(
                f"Pair index version={metadata['version']}, expected {INDEX_VERSION}; "
                "rebuild it with tools/build_v8_pair_index.py"
            )
        if indexed_board_size != board_size:
            raise ValueError(
                f"Pair index board_size={indexed_board_size}, requested {board_size}"
            )
        self.t_min = int(metadata["t_min"])
        self.t_max = int(metadata["t_max"])
        self.positions_per_game = int(metadata.get("positions_per_game", 0))
        self.use_constructive_pairs = bool(
            metadata.get("use_constructive_pairs", False)
        )
        self.use_surface_hard_negatives = bool(
            metadata.get("use_surface_hard_negatives", False)
        )
        self.surface_jaccard_threshold = float(
            metadata.get("surface_jaccard_threshold", 0.9)
        )
        self.surface_max_pairs_per_anchor = int(
            metadata.get("surface_max_pairs_per_anchor", 1)
        )
        self.coverage_stats = dict(metadata.get("coverage_stats") or {})
        self.embedded_pair_prefixes = bool(
            metadata.get("embedded_pair_prefixes", False)
        )
        self.raw_to_token, _ = build_mappings(board_size)
        self._chunk_ids = {
            str(name): int(chunk_id)
            for chunk_id, name in self.connection.execute("SELECT id, name FROM chunks")
        }
        self._chunk_names = {
            chunk_id: name for name, chunk_id in self._chunk_ids.items()
        }
        self._chunk_cache: OrderedDict[str, list[list[int]]] = OrderedDict()
        self._positive_cache: dict[tuple[int, int], list[tuple]] = {}
        self._hard_cache: dict[tuple[int, int], list[tuple]] = {}
        self._state_cache: dict[tuple[int, int], bytes] = {}
        self._current_chunk_name: str | None = None
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def pairs_for_chunk(self, chunk_name: str) -> None:
        self._positive_cache.clear()
        self._hard_cache.clear()
        self._state_cache.clear()
        self._current_chunk_name = chunk_name
        try:
            chunk_id = self._chunk_ids[chunk_name]
        except KeyError as error:
            raise KeyError(f"Pair index has no chunk named {chunk_name}") from error
        for game_idx, t, state_key in self.connection.execute(
            """
            SELECT game_idx, t, state_key
            FROM occurrences WHERE chunk_id = ?
            """,
            (chunk_id,),
        ):
            self._state_cache[(int(game_idx), int(t))] = state_key
        positive_query = (
            """
            SELECT anchor_game, t, partner_chunk_id, partner_game, partner_t,
                   partner_prefix
            FROM positive_pairs
            WHERE anchor_chunk_id = ?
            """
            if self.embedded_pair_prefixes
            else
            """
            SELECT pairs.anchor_game, pairs.t, pairs.partner_chunk_id,
                   pairs.partner_game, pairs.partner_t,
                   COALESCE(pairs.partner_prefix, partner.prefix_key)
            FROM positive_pairs AS pairs
            LEFT JOIN occurrences AS partner
              ON partner.chunk_id = pairs.partner_chunk_id
             AND partner.game_idx = pairs.partner_game
             AND partner.t = pairs.partner_t
            WHERE pairs.anchor_chunk_id = ?
            """
        )
        for row in self.connection.execute(positive_query, (chunk_id,)):
            key = (int(row[0]), int(row[1]))
            partner_chunk = (
                self._chunk_names[int(row[2])] if row[2] is not None else None
            )
            self._positive_cache.setdefault(key, []).append(
                (partner_chunk, *tuple(row[3:]))
            )
        hard_query = (
            """
            SELECT anchor_game, t, negative_chunk_id, negative_game, negative_t,
                   negative_prefix
            FROM hard_negative_pairs
            WHERE anchor_chunk_id = ?
            """
            if self.embedded_pair_prefixes
            else
            """
            SELECT pairs.anchor_game, pairs.t, pairs.negative_chunk_id,
                   pairs.negative_game, pairs.negative_t,
                   COALESCE(pairs.negative_prefix, negative.prefix_key)
            FROM hard_negative_pairs AS pairs
            LEFT JOIN occurrences AS negative
              ON negative.chunk_id = pairs.negative_chunk_id
             AND negative.game_idx = pairs.negative_game
             AND negative.t = pairs.negative_t
            WHERE pairs.anchor_chunk_id = ?
            """
        )
        for row in self.connection.execute(hard_query, (chunk_id,)):
            key = (int(row[0]), int(row[1]))
            negative_chunk = (
                self._chunk_names[int(row[2])] if row[2] is not None else None
            )
            self._hard_cache.setdefault(key, []).append(
                (negative_chunk, *tuple(row[3:]))
            )

    def positive_refs(self, chunk_name: str, game_idx: int, t: int) -> list[tuple]:
        self._require_current_chunk(chunk_name)
        return self._positive_cache.get((game_idx, t), [])

    def hard_negative_refs(
        self, chunk_name: str, game_idx: int, t: int
    ) -> list[tuple]:
        self._require_current_chunk(chunk_name)
        return self._hard_cache.get((game_idx, t), [])

    def state_key(self, chunk_name: str, game_idx: int, t: int) -> bytes:
        self._require_current_chunk(chunk_name)
        try:
            return self._state_cache[(game_idx, t)]
        except KeyError as error:
            raise KeyError(
                f"Pair index has no occurrence for {chunk_name}:{game_idx}@{t}"
            ) from error

    def state_key_or_none(
        self,
        chunk_name: str,
        game_idx: int,
        t: int,
    ) -> bytes | None:
        self._require_current_chunk(chunk_name)
        return self._state_cache.get((game_idx, t))

    def indexed_states(self, chunk_name: str) -> Iterable[tuple[int, int, bytes]]:
        """Iterate the current chunk's indexed ``(game_idx, t, state_key)`` rows."""
        self._require_current_chunk(chunk_name)
        for (game_idx, t), state_key in self._state_cache.items():
            yield game_idx, t, state_key

    def _require_current_chunk(self, chunk_name: str) -> None:
        if chunk_name != self._current_chunk_name:
            raise ValueError(
                f"Pair cache contains {self._current_chunk_name!r}, "
                f"not {chunk_name!r}; call pairs_for_chunk first"
            )

    @property
    def indexed_anchor_count(self) -> int:
        return len(self._state_cache)

    def _load_chunk(self, chunk_name: str) -> list[list[int]]:
        if chunk_name in self._chunk_cache:
            games = self._chunk_cache.pop(chunk_name)
            self._chunk_cache[chunk_name] = games
            return games
        games = load_chunk(str(self.data_dir / chunk_name))
        self._chunk_cache[chunk_name] = games
        while len(self._chunk_cache) > self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return games

    def resolve_prefix_tokens(self, ref: tuple) -> list[int]:
        chunk_name, game_idx, t, prefix_blob = ref
        if prefix_blob is not None:
            raw_prefix = list(prefix_blob)
        else:
            if chunk_name is None or game_idx is None:
                raise ValueError("Pair reference has neither chunk/game nor prefix")
            raw_prefix = self._load_chunk(str(chunk_name))[int(game_idx)][: int(t)]
        tokens = [self.raw_to_token[move] for move in raw_prefix]
        if any(token < 0 for token in tokens):
            raise ValueError("Pair prefix contains an un-tokenizable starting square")
        return tokens
