"""Canonical Othello rules and deterministic game generation."""

from .board import OthelloBoardState
from .generator import (
    NUMBA_AVAILABLE,
    dense_to_game_lists,
    generate_games_array,
    validate_board_size,
)

__all__ = [
    "NUMBA_AVAILABLE",
    "OthelloBoardState",
    "dense_to_game_lists",
    "generate_games_array",
    "validate_board_size",
]
