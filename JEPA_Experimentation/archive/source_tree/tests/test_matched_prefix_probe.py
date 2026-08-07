from __future__ import annotations

import numpy as np
import torch

from othello_research.othello.board import OthelloBoardState
from othello_research.probes.board_state import board_state_labels_after_moves
from othello_research.probes.matched_prefix_probe import (
    MatchedPrefixSample,
    _labels_from_masks,
    _replay_passfree,
    extract_last_position_features,
    sample_matched_prefixes,
    train_matched_probe_bank,
)


def _random_games(
    n_games: int, seed: int, board_size: int = 8
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(n_games):
        board = OthelloBoardState(n=board_size)
        moves: list[int] = []
        while len(moves) < board_size * board_size - 4:
            legal = board.get_valid_moves()
            if not legal:
                board.next_hand_color *= -1
                legal = board.get_valid_moves()
                board.next_hand_color *= -1
                if not legal:
                    break
            move = int(legal[rng.integers(len(legal))])
            try:
                board.umpire(move)
            except Exception:
                break
            moves.append(move)
        games.append(moves)
    return games


def test_labels_match_reference_convention_on_passfree_prefixes() -> None:
    """Bitboard-derived labels must equal board_state.py's per exact t."""
    games = _random_games(40, seed=3)
    checked = 0
    for game in games:
        states = _replay_passfree(game)
        if len(states) < 6:
            continue
        abs_ref, rel_ref = board_state_labels_after_moves(
            game[: len(states)], 8
        )
        for t in (4, len(states) // 2, len(states)):
            black, white = states[t - 1]
            y_abs, y_rel = _labels_from_masks(black, white, t, 8)
            assert y_abs.tolist() == abs_ref[t - 1], f"abs mismatch at t={t}"
            assert y_rel.tolist() == rel_ref[t - 1], f"rel mismatch at t={t}"
            checked += 1
    assert checked >= 30


def test_sampler_and_labels_support_12x12() -> None:
    games = _random_games(12, seed=31, board_size=12)
    checked = 0
    for game in games:
        states = _replay_passfree(game, board_size=12)
        if len(states) < 12:
            continue
        abs_ref, rel_ref = board_state_labels_after_moves(
            game[: len(states)], 12
        )
        for t in (4, 8, 12):
            black, white = states[t - 1]
            y_abs, y_rel = _labels_from_masks(black, white, t, 12)
            assert y_abs.shape == (144,)
            assert y_rel.shape == (144,)
            assert y_abs.tolist() == abs_ref[t - 1]
            assert y_rel.tolist() == rel_ref[t - 1]
            checked += 1

    samples = sample_matched_prefixes(
        games,
        board_size=12,
        t_min=4,
        t_max=12,
        prefixes_per_game=2,
        seed=5,
    )
    assert checked >= 12
    assert samples
    assert all(sample.y_abs.shape == (144,) for sample in samples)
    assert all(sample.y_rel.shape == (144,) for sample in samples)


def test_sampler_respects_t_range_and_passfree_rule() -> None:
    games = _random_games(60, seed=9)
    samples = sample_matched_prefixes(
        games, t_min=4, t_max=20, prefixes_per_game=3, seed=0
    )
    assert samples
    for s in samples:
        assert 4 <= s.t <= 20
        assert len(s.tokens) == s.t
        assert s.tokens.min() >= 0 and s.tokens.max() < 60
        assert set(np.unique(s.y_abs)).issubset({0, 1, 2})
        assert set(np.unique(s.y_rel)).issubset({0, 1, 2})
        # occupancy consistency: same cells non-empty in both label systems
        assert ((s.y_abs == 2) == (s.y_rel == 2)).all()


def test_feature_extraction_and_probe_training_end_to_end() -> None:
    from othello_research.models.gpt import GPTConfig, OthelloGPT

    encoder = OthelloGPT(GPTConfig(board_size=8, n_layers=1, n_heads=4,
                                   d_model=32, dropout=0.0))
    encoder.eval()
    games = _random_games(50, seed=21)
    device = torch.device("cpu")

    samples = sample_matched_prefixes(games, t_min=4, t_max=12,
                                      prefixes_per_game=2, seed=1)
    feats, y_abs, y_rel, t_arr = extract_last_position_features(
        encoder, samples, layers=(0, 1), device=device, batch_size=16,
        precision="fp32",
    )
    assert set(feats) == {0, 1}
    assert feats[0].shape == (len(samples), 32)
    assert y_abs.shape == (len(samples), 64)

    # Padding must not affect the gathered position (causal mask): re-extract
    # the first sample alone and compare.
    single = [samples[0]]
    feats_single, *_ = extract_last_position_features(
        encoder, single, layers=(1,), device=device, precision="fp32",
    )
    joint_idx = 0
    assert torch.allclose(feats[1][joint_idx], feats_single[1][0], atol=1e-5)

    result = train_matched_probe_bank(
        encoder, games[:40], games[40:],
        layers=(0, 1), device=device, probe_type="linear",
        epochs=1, batch_size=32, lr=1e-3, weight_decay=0.0,
        precision="fp32", t_min=4, t_max=12, prefixes_per_game=2, seed=2,
        input_layernorm=True, gradient_clip=1.0,
    )
    assert result["layers"] == [0, 1]
    assert result["input_layernorm"] is True
    assert result["gradient_clip"] == 1.0
    assert result["seed"] == 2
    for layer in ("0", "1"):
        for mode in ("absolute", "relative"):
            acc = result["metrics"][layer][mode]["accuracy"]
            assert 0.0 <= acc <= 1.0
