"""Move-vocabulary mapping for Othello.

The raw board representation indexes squares 0..n²-1 in row-major order.
However, the 4 center squares are occupied by the starting pieces and can
never be played as moves. Following Li et al. (2023), we collapse the move
vocabulary by skipping these 4 starting positions, yielding n²-4 valid
move tokens plus 1 padding token.

For an 8×8 board:
    raw positions   : 0..63 (64 values, with 27, 28, 35, 36 unused)
    token ids       : 0..59 (60 values, contiguous)
    padding token   : 60
    vocab_size      : 61

This module provides the forward/inverse mapping needed at the dataset
boundary. The model itself is unaware of the raw representation: it only
ever sees token ids in [0, vocab_size).
"""

from functools import lru_cache


def starting_positions(board_size: int) -> list[int]:
    """Return the 4 raw positions occupied by the starting pieces.

    Othello starts with 4 pieces in the center 2×2 block. For an n×n board
    (n must be even), these are at rows n/2-1, n/2 and columns n/2-1, n/2.

    Args:
        board_size: side length of the (square) board. Must be even.

    Returns:
        Sorted list of 4 raw position indices (0..n²-1).
    """
    assert board_size % 2 == 0, f"board_size must be even, got {board_size}"
    n = board_size
    r1, r2 = n // 2 - 1, n // 2
    return sorted([
        r1 * n + r1,    # top-left of center block
        r1 * n + r2,    # top-right
        r2 * n + r1,    # bottom-left
        r2 * n + r2,    # bottom-right
    ])


@lru_cache(maxsize=None)
def build_mappings(board_size: int) -> tuple[list[int], list[int]]:
    """Build the raw↔token mapping tables for a given board size.

    Cached: each board size is computed only once per process.

    Args:
        board_size: side length of the board.

    Returns:
        (raw_to_token, token_to_raw) where:
          raw_to_token[r] = token id for raw position r, or -1 if r is a
              starting position (and therefore has no token).
          token_to_raw[t] = raw position for token id t. Length n²-4.
    """
    n = board_size
    skip = set(starting_positions(n))

    raw_to_token = [-1] * (n * n)
    token_to_raw = []

    next_token = 0
    for raw in range(n * n):
        if raw in skip:
            continue
        raw_to_token[raw] = next_token
        token_to_raw.append(raw)
        next_token += 1

    assert next_token == n * n - 4
    assert len(token_to_raw) == n * n - 4
    return raw_to_token, token_to_raw


def raw_to_token_list(moves: list[int], board_size: int) -> list[int]:
    """Convert a list of raw moves to token ids.

    Args:
        moves: list of raw position indices (0..n²-1). Must not contain
            any starting positions.
        board_size: board side length.

    Returns:
        List of token ids of the same length.

    Raises:
        ValueError: if any move is a starting position.
    """
    raw_to_token, _ = build_mappings(board_size)
    out = []
    for m in moves:
        token = raw_to_token[m]
        if token == -1:
            raise ValueError(
                f"Move {m} is a starting position on a {board_size}×{board_size} "
                f"board and cannot be a valid move."
            )
        out.append(token)
    return out


def token_to_raw_list(tokens: list[int], board_size: int) -> list[int]:
    """Convert a list of token ids back to raw positions.

    Args:
        tokens: list of token ids (0..n²-5).
        board_size: board side length.

    Returns:
        List of raw position indices.
    """
    _, token_to_raw = build_mappings(board_size)
    return [token_to_raw[t] for t in tokens]