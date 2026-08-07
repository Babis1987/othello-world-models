"""Fast deterministic random Othello generation for arbitrary even boards.

The training corpus stores raw square ids, so a 12x12 game contains integers
in ``[0, 143]`` and at most ``12**2 - 4 == 140`` moves.  Generation uses a
per-game RNG stream derived from ``(seed, global_game_index)``; consequently
the corpus is reproducible across shard sizes, resumes, and thread counts.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - full generation requires numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def wrap(fn):
            return fn

        return wrap

    def prange(*args):
        return range(*args)

    def set_num_threads(_count: int) -> None:
        return None


_DR = np.asarray((-1, -1, 0, 1, 1, 1, 0, -1), dtype=np.int8)
_DC = np.asarray((0, 1, 1, 1, 0, -1, -1, -1), dtype=np.int8)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_SM64_A = np.uint64(0x9E3779B97F4A7C15)
_SM64_B = np.uint64(0xBF58476D1CE4E5B9)
_SM64_C = np.uint64(0x94D049BB133111EB)
_XORSHIFT_MUL = np.uint64(0x2545F4914F6CDD1D)


@njit(cache=True)
def _splitmix64(value: np.uint64) -> np.uint64:
    value = (value + _SM64_A) & _MASK64
    value = ((value ^ (value >> np.uint64(30))) * _SM64_B) & _MASK64
    value = ((value ^ (value >> np.uint64(27))) * _SM64_C) & _MASK64
    return value ^ (value >> np.uint64(31))


@njit(cache=True)
def _next_random(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    state ^= state >> np.uint64(12)
    state ^= (state << np.uint64(25)) & _MASK64
    state ^= state >> np.uint64(27)
    return state, (state * _XORSHIFT_MUL) & _MASK64


@njit(cache=True)
def _is_legal(cells: np.ndarray, n: int, square: int, color: int) -> bool:
    if cells[square] != 0:
        return False
    row = square // n
    col = square - row * n
    for direction in range(8):
        dr = int(_DR[direction])
        dc = int(_DC[direction])
        r = row + dr
        c = col + dc
        found_opponent = False
        while 0 <= r < n and 0 <= c < n and cells[r * n + c] == -color:
            found_opponent = True
            r += dr
            c += dc
        if found_opponent and 0 <= r < n and 0 <= c < n:
            if cells[r * n + c] == color:
                return True
    return False


@njit(cache=True)
def _collect_legal(
    cells: np.ndarray,
    n: int,
    color: int,
    legal_buffer: np.ndarray,
) -> int:
    count = 0
    for square in range(n * n):
        if _is_legal(cells, n, square, color):
            legal_buffer[count] = square
            count += 1
    return count


@njit(cache=True)
def _apply_move(cells: np.ndarray, n: int, square: int, color: int) -> None:
    row = square // n
    col = square - row * n
    cells[square] = color
    for direction in range(8):
        dr = int(_DR[direction])
        dc = int(_DC[direction])
        r = row + dr
        c = col + dc
        run_length = 0
        while 0 <= r < n and 0 <= c < n and cells[r * n + c] == -color:
            run_length += 1
            r += dr
            c += dc
        if run_length == 0 or not (0 <= r < n and 0 <= c < n):
            continue
        if cells[r * n + c] != color:
            continue
        r = row + dr
        c = col + dc
        for _ in range(run_length):
            cells[r * n + c] = color
            r += dr
            c += dc


@njit(parallel=True, cache=True)
def _generate_games_kernel(
    board_size: int,
    global_start_index: int,
    count: int,
    seed: np.uint64,
) -> tuple[np.ndarray, np.ndarray]:
    max_moves = board_size * board_size - 4
    moves = np.full((count, max_moves), -1, dtype=np.int16)
    lengths = np.zeros(count, dtype=np.int16)

    for local_index in prange(count):
        cells = np.zeros(board_size * board_size, dtype=np.int8)
        legal_buffer = np.empty(board_size * board_size, dtype=np.int16)
        mid = board_size // 2
        cells[(mid - 1) * board_size + (mid - 1)] = -1
        cells[(mid - 1) * board_size + mid] = 1
        cells[mid * board_size + (mid - 1)] = 1
        cells[mid * board_size + mid] = -1

        game_index = np.uint64(global_start_index + local_index)
        rng_state = _splitmix64(seed ^ _splitmix64(game_index))
        if rng_state == np.uint64(0):
            rng_state = _SM64_A
        color = 1
        length = 0

        for _ply in range(max_moves):
            n_legal = _collect_legal(
                cells, board_size, color, legal_buffer
            )
            if n_legal == 0:
                color = -color
                n_legal = _collect_legal(
                    cells, board_size, color, legal_buffer
                )
                if n_legal == 0:
                    break

            rng_state, random_value = _next_random(rng_state)
            selected = int(random_value % np.uint64(n_legal))
            move = int(legal_buffer[selected])
            _apply_move(cells, board_size, move, color)
            moves[local_index, length] = move
            length += 1
            color = -color

        lengths[local_index] = length

    return moves, lengths


def validate_board_size(board_size: int) -> None:
    if board_size < 4 or board_size % 2 != 0:
        raise ValueError(
            f"board_size must be an even integer >= 4, got {board_size}"
        )
    if board_size * board_size - 1 > np.iinfo(np.int16).max:
        raise ValueError("board_size is too large for the int16 move format")


def generate_games_array(
    board_size: int,
    *,
    global_start_index: int,
    count: int,
    seed: int,
    num_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic dense move array plus per-game lengths."""
    validate_board_size(board_size)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if global_start_index < 0:
        raise ValueError("global_start_index must be non-negative")
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Numba is required for large-corpus generation. Install numba or "
            "use scripts/generate_data.py --engine reference for a smoke run."
        )
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        set_num_threads(num_threads)
    return _generate_games_kernel(
        int(board_size),
        int(global_start_index),
        int(count),
        np.uint64(seed),
    )


def dense_to_game_lists(
    moves: np.ndarray,
    lengths: np.ndarray,
) -> list[list[int]]:
    """Convert the dense Numba output into the repository pickle schema."""
    if moves.ndim != 2 or lengths.ndim != 1 or len(moves) != len(lengths):
        raise ValueError("moves/lengths shapes are inconsistent")
    # NumPy's C-level conversion is materially faster than iterating over every
    # move in Python (a 100k-game 12x12 shard contains about 14M integers).
    return [
        moves[i, : int(lengths[i])].tolist() for i in range(len(lengths))
    ]
