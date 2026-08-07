"""Tests for the v4 frozen-readout hyperparameter search.

The search runs on cached encoder features, so these tests build a small real
encoder and real legal Othello games rather than mocking the cache: the parts
most likely to break are the target/legal-mask alignment and the fold grouping,
and both are only meaningful against genuine replay data.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import pytest
import torch

from othello_research.evaluation.unified import (
    PROTOCOL_ID,
    V2_PROTOCOL_ID,
    V3_PROTOCOL_ID,
    FrozenEncoderNextMoveHead,
    UnifiedEvalConfig,
    _fit_cached_readout,
    build_head_tuning_cache,
    build_readout,
    default_head_grid,
    train_frozen_next_move_head,
    tune_frozen_head_hyperparameters,
)
from othello_research.models.gpt import GPTConfig, OthelloGPT
from othello_research.othello.board import OthelloBoardState

BOARD = 8
CPU = torch.device("cpu")


def _random_game(rng: random.Random) -> list[int]:
    board = OthelloBoardState(n=BOARD)
    moves: list[int] = []
    while True:
        legal = board.get_valid_moves()
        if not legal:
            return moves
        move = rng.choice(sorted(legal))
        board.umpire(move)
        moves.append(move)


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    rng = random.Random(0)
    root = tmp_path_factory.mktemp("head_search_shards")
    paths: list[Path] = []
    for index in range(3):
        games: list[list[int]] = []
        while len(games) < 40:
            game = _random_game(rng)
            if len(game) >= 6:
                games.append(game)
        path = root / f"games_{index:03d}.pickle"
        path.write_bytes(pickle.dumps(games))
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def config() -> UnifiedEvalConfig:
    return UnifiedEvalConfig(
        board_size=BOARD,
        require_cuda_bf16=False,
        head_batch_size=16,
        head_tuning_folds=3,
        head_tuning_games=90,
        head_tuning_epochs=2,
        head_tuning_token_batch=256,
        head_max_shards=3,
        head_selection_games=20,
        head_eval_every_shards=1,
        head_patience=1,
    )


@pytest.fixture(scope="module")
def encoder() -> OthelloGPT:
    model = OthelloGPT(
        GPTConfig(board_size=BOARD, n_layers=2, n_heads=2, d_model=32, dropout=0.0)
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def cache(encoder: OthelloGPT, shards: list[Path], config: UnifiedEvalConfig):
    view = FrozenEncoderNextMoveHead(encoder, "linear", encoder_precision="fp32")
    return build_head_tuning_cache(view, shards, config=config, device=CPU)


def test_protocol_is_v4_and_keeps_older_ids_loadable() -> None:
    assert PROTOCOL_ID == "unified_eval_v4"
    assert UnifiedEvalConfig(board_size=BOARD).protocol_id == PROTOCOL_ID
    assert UnifiedEvalConfig.v3(BOARD).protocol_id == V3_PROTOCOL_ID
    assert UnifiedEvalConfig.v2(BOARD).protocol_id == V2_PROTOCOL_ID
    # The reproductions of older protocols must not silently gain the search.
    assert UnifiedEvalConfig(board_size=BOARD).head_tuning_enabled
    assert not UnifiedEvalConfig.v3(BOARD).head_tuning_enabled
    assert not UnifiedEvalConfig.v2(BOARD).head_tuning_enabled


def test_grid_is_small_and_contains_the_v3_fixed_point() -> None:
    linear = default_head_grid("linear")
    mlp = default_head_grid("mlp")
    assert len(linear) == 6
    assert len(mlp) == 4
    # Reading the tuning gain as best-minus-default requires the v3 setting to
    # be one of the searched points.
    assert any(
        entry["learning_rate"] == 1e-3
        and entry["weight_decay"] == 0.0
        and entry["l1_strength"] == 0.0
        for entry in linear
    )
    assert any(
        entry["learning_rate"] == 1e-3 and entry["dropout"] == 0.1 for entry in mlp
    )
    # Linear must vary regularization, not only learning rate.
    assert {entry["regularization"] for entry in linear} == {
        "none",
        "l2=1e-3",
        "l1=1e-5",
    }
    with pytest.raises(ValueError):
        default_head_grid("attention")


def test_cache_targets_are_legal_and_folds_are_disjoint(cache) -> None:
    assert cache.n_positions > 0
    assert cache.n_folds == 3
    assert sum(cache.games_per_fold) == 90
    # Each recorded target must lie inside its own legal set. This is the check
    # that catches an off-by-one between hidden states and next-move targets.
    assert bool(cache.legal_mask.gather(1, cache.targets[:, None]).all())
    assert int(cache.legal_mask.sum(dim=1).min()) >= 1
    for fold in range(cache.n_folds):
        held_out = cache.fold_index(fold, held_out=True)
        train = cache.fold_index(fold, held_out=False)
        assert held_out.numel() > 0 and train.numel() > 0
        assert set(held_out.tolist()).isdisjoint(train.tolist())
        assert held_out.numel() + train.numel() == cache.n_positions


def test_search_reports_every_grid_point_and_picks_the_cv_winner(
    cache,
    config: UnifiedEvalConfig,
) -> None:
    record = tune_frozen_head_hyperparameters(
        cache, head_type="linear", config=config, device=CPU
    )
    assert record["grid_size"] == len(record["results"]) == 6
    assert record["folds"] == 3
    assert record["tuning_games"] == 90
    assert "shard" in record["grouping"]
    best = max(
        record["results"], key=lambda row: row["cv_legal_probability_mass_mean"]
    )
    assert record["selected"]["label"] == best["label"]
    assert record["selection_metric"] == "cv_legal_probability_mass_mean"
    for row in record["results"]:
        assert len(row["folds"]) == 3
        assert 0.0 <= row["cv_legal_probability_mass_mean"] <= 1.0
        assert row["cv_legal_probability_mass_spread"] >= 0.0


def test_mlp_search_reports_its_dropout_and_width(
    cache,
    config: UnifiedEvalConfig,
) -> None:
    record = tune_frozen_head_hyperparameters(
        cache, head_type="mlp", config=config, device=CPU
    )
    selected = record["selected"]
    assert selected["dropout"] in {0.0, 0.1}
    assert selected["hidden_dim"] == config.mlp_hidden_dim


def test_l1_shrinks_weights_relative_to_no_regularization(
    cache,
    config: UnifiedEvalConfig,
) -> None:
    index = cache.fold_index(0, held_out=False)
    magnitudes = {}
    for label, l1_strength in (("none", 0.0), ("l1", 1e-2)):
        head = _fit_cached_readout(
            cache,
            index,
            {
                "label": label,
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "l1_strength": l1_strength,
            },
            head_type="linear",
            config=config,
            device=CPU,
            seed=7,
        )
        magnitudes[label] = float(head.weight.detach().abs().mean())
    assert magnitudes["l1"] < magnitudes["none"]


def test_l1_penalty_leaves_biases_unpenalized() -> None:
    from othello_research.evaluation.unified import _l1_penalty

    head = build_readout(4, 3, "linear")
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(5.0)
    assert float(_l1_penalty(head)) == 0.0


def test_refit_uses_the_selected_hyperparameters(
    encoder: OthelloGPT,
    shards: list[Path],
    config: UnifiedEvalConfig,
) -> None:
    selected = {
        "label": "linear__lr=0.0003__l1=1e-5",
        "learning_rate": 3e-4,
        "regularization": "l1=1e-5",
        "weight_decay": 0.0,
        "l1_strength": 1e-5,
    }
    _, tuned = train_frozen_next_move_head(
        encoder,
        shards,
        selection_chunks=shards[-1:],
        head_type="linear",
        config=config,
        device=CPU,
        hyperparameters=selected,
    )
    assert tuned["hyperparameters_source"] == "grouped-CV search"
    assert tuned["learning_rate"] == pytest.approx(3e-4)
    assert tuned["l1_strength"] == pytest.approx(1e-5)
    assert tuned["regularization"] == "l1=1e-5"

    _, default = train_frozen_next_move_head(
        encoder,
        shards,
        selection_chunks=shards[-1:],
        head_type="linear",
        config=config,
        device=CPU,
    )
    assert default["hyperparameters_source"] == "protocol defaults"
    assert default["learning_rate"] == pytest.approx(config.head_learning_rate)
    assert default["l1_strength"] == 0.0


def test_search_never_reads_the_selection_or_test_shards(
    encoder: OthelloGPT,
    shards: list[Path],
    config: UnifiedEvalConfig,
) -> None:
    """The cache must be buildable from the head pool alone."""
    view = FrozenEncoderNextMoveHead(encoder, "linear", encoder_precision="fp32")
    built = build_head_tuning_cache(view, shards, config=config, device=CPU)
    assert built.n_positions > 0
    # Fewer shards than folds cannot be grouped and must fail loudly rather
    # than silently reusing one shard for both fitting and scoring.
    with pytest.raises(ValueError, match="shards to group by"):
        build_head_tuning_cache(view, shards[:2], config=config, device=CPU)


def test_tuning_config_validation() -> None:
    with pytest.raises(ValueError, match="head_tuning_folds"):
        UnifiedEvalConfig(board_size=BOARD, head_tuning_folds=1)
    with pytest.raises(ValueError, match="at least one game per fold"):
        UnifiedEvalConfig(board_size=BOARD, head_tuning_games=2, head_tuning_folds=3)
    # Disabling the search must skip its validation entirely.
    UnifiedEvalConfig(
        board_size=BOARD,
        head_tuning_enabled=False,
        head_tuning_games=0,
        head_tuning_folds=1,
    )
