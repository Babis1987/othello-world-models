from __future__ import annotations

import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

import othello_research.datasets.order_aware_pair_index as pair_index_module
from othello_research.datasets.order_aware_dataset import (
    OrderAwareBatchConfig,
    OrderAwareChunkBatches,
)
from othello_research.datasets.order_aware_pair_index import (
    OrderAwarePairIndex,
    _fast_umpire_known_legal_game,
    _full_board_state_key_for_next_action,
    build_pair_index,
    classify_order_pair,
    full_board_state_key,
    occupancy_key,
    replay_prefix,
)
from othello_research.objectives.jepa_order_aware import (
    JEPAOrderAwareConfig,
    OthelloJEPAOrderAware,
    order_aware_infonce_loss,
)
from othello_research.othello.board import OthelloBoardState
from othello_research.training import train_jepa as tjm


POSITIVE_ORDERS = (
    [19, 18, 26],
    [26, 18, 19],
)
HARD_NEGATIVE_ORDERS = (
    [19, 18, 37, 20],
    [19, 20, 37, 18],
)
SURFACE_HARD_NEGATIVE_PREFIXES = (
    [44, 43, 18, 29, 51, 34, 30, 22],
    [44, 43, 18, 29, 51, 50, 30, 22],
)


def _write_chunk(path: Path, games: list[list[int]]) -> None:
    with open(path, "wb") as handle:
        pickle.dump(games, handle)


def _build_known_index(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    positive_chunk = tmp_path / "positive.pickle"
    hard_chunk = tmp_path / "hard.pickle"
    _write_chunk(
        positive_chunk,
        [
            [*POSITIVE_ORDERS[0], 20],
            [*POSITIVE_ORDERS[1], 20],
            [*HARD_NEGATIVE_ORDERS[0]],
        ],
    )
    _write_chunk(
        hard_chunk,
        [[*HARD_NEGATIVE_ORDERS[0], 9], [*HARD_NEGATIVE_ORDERS[1], 9]],
    )
    index_path = tmp_path / "pairs.sqlite"
    stats = build_pair_index(
        [positive_chunk, hard_chunk],
        index_path,
        board_size=8,
        t_min=3,
        t_max=4,
    )
    return positive_chunk, hard_chunk, index_path, stats


def _tiny_cfg(**overrides) -> JEPAOrderAwareConfig:
    values = {
        "board_size": 8,
        "n_layers": 1,
        "n_heads": 4,
        "d_model": 32,
        "dropout": 0.0,
        "predictor_hidden_mult": 2,
        "predictor_n_layers": 2,
        "predictor_dropout": 0.0,
        "action_dim": 16,
    }
    values.update(overrides)
    return JEPAOrderAwareConfig(**values)


def _objective_batch(
    model: OthelloJEPAOrderAware,
    *,
    batch_size: int = 4,
    t: int = 5,
    hard_count: int = 1,
) -> dict[str, torch.Tensor]:
    context = torch.randint(0, model.config.vocab_size, (batch_size, t))
    actions = torch.randint(0, model.config.vocab_size - 1, (batch_size, 1))
    positive = torch.cat([context.roll(1, 0), actions], dim=1)
    hard = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, hard_count, t + 1),
    )
    if hard_count:
        hard[:, :, -1] = actions
    return {
        "x_context": context,
        "x_positive_target": positive,
        "x_hard_negative_targets": hard,
        "hard_negative_mask": torch.ones(batch_size, hard_count, dtype=torch.bool),
        "group_ids": torch.tensor([0, 0, 1, 1]),
        "positive_fallback_mask": torch.zeros(batch_size, dtype=torch.bool),
    }


def test_known_cross_order_pair_has_identical_full_board_state() -> None:
    assert classify_order_pair(*POSITIVE_ORDERS, board_size=8) == "positive"
    first = replay_prefix(POSITIVE_ORDERS[0], 8)
    second = replay_prefix(POSITIVE_ORDERS[1], 8)
    assert full_board_state_key(first) == full_board_state_key(second)


def test_known_same_set_pair_is_hard_negative_not_positive() -> None:
    assert (
        classify_order_pair(*HARD_NEGATIVE_ORDERS, board_size=8)
        == "hard_negative"
    )
    first = replay_prefix(HARD_NEGATIVE_ORDERS[0], 8)
    second = replay_prefix(HARD_NEGATIVE_ORDERS[1], 8)
    assert occupancy_key(first) == occupancy_key(second)
    assert full_board_state_key(first) != full_board_state_key(second)


def test_fast_index_replay_and_next_action_key_match_reference_engine() -> None:
    generator = torch.Generator().manual_seed(7)
    source = OthelloBoardState(n=8)
    moves: list[int] = []
    for _ in range(30):
        legal = source.get_valid_moves()
        if not legal:
            source.next_hand_color *= -1
            legal = source.get_valid_moves()
        if not legal:
            break
        move = legal[int(torch.randint(len(legal), (1,), generator=generator))]
        source.umpire(move)
        moves.append(move)

    reference = OthelloBoardState(n=8)
    fast = OthelloBoardState(n=8)
    for index, move in enumerate(moves):
        reference.umpire(move)
        _fast_umpire_known_legal_game(fast, move)
        assert np.array_equal(reference.state, fast.state)
        assert reference.next_hand_color == fast.next_hand_color
        if index + 1 < len(moves):
            assert _full_board_state_key_for_next_action(
                fast,
                moves[index + 1],
            ) == full_board_state_key(reference)


def test_fast_constructive_classification_matches_reference_engine(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(17)
    games = [_random_legal_game(rng, max_moves=14) for _ in range(12)]
    games = [game for game in games if len(game) >= 11]
    expected_positives = 0
    expected_hard_negatives = 0
    for game_idx, game in enumerate(games):
        for t in range(4, min(10, len(game) - 1) + 1):
            positives, hard_negatives = pair_index_module._enumerate_constructive_pairs(
                chunk_id=0,
                game_idx=game_idx,
                prefix=game[:t],
                next_action=game[t],
                board_size=8,
            )
            expected_positives += len(positives)
            expected_hard_negatives += len(hard_negatives)

    chunk = tmp_path / "constructive.pickle"
    _write_chunk(chunk, games)
    stats = build_pair_index(
        [chunk],
        tmp_path / "constructive.sqlite",
        board_size=8,
        t_min=4,
        t_max=10,
        positions_per_game=0,
        use_constructive_pairs=True,
    )
    assert stats["constructive_positive_count"] == expected_positives
    assert stats["constructive_hard_negative_count"] == expected_hard_negatives


def test_built_index_pairs_obey_full_state_and_occupancy_contract(
    tmp_path: Path,
) -> None:
    positive_chunk, hard_chunk, index_path, stats = _build_known_index(tmp_path)
    games = {
        positive_chunk.name: pickle.loads(positive_chunk.read_bytes()),
        hard_chunk.name: pickle.loads(hard_chunk.read_bytes()),
    }
    connection = sqlite3.connect(index_path)
    try:
        positive_rows = connection.execute(
            "SELECT anchor.name, pairs.anchor_game, pairs.t, "
            "partner.name, pairs.partner_game "
            "FROM positive_pairs AS pairs "
            "JOIN chunks AS anchor ON anchor.id = pairs.anchor_chunk_id "
            "JOIN chunks AS partner ON partner.id = pairs.partner_chunk_id"
        ).fetchall()
        hard_rows = connection.execute(
            "SELECT anchor.name, pairs.anchor_game, pairs.t, "
            "negative.name, pairs.negative_game "
            "FROM hard_negative_pairs AS pairs "
            "JOIN chunks AS anchor ON anchor.id = pairs.anchor_chunk_id "
            "JOIN chunks AS negative ON negative.id = pairs.negative_chunk_id"
        ).fetchall()
    finally:
        connection.close()

    assert stats["positive_count"] == len(positive_rows) == 2
    assert stats["hard_negative_count"] == len(hard_rows) == 2
    assert stats["positive_anchor_count"] == 2
    assert stats["hard_negative_anchor_count"] == 2
    assert stats["estimated_positive_fallback_rate"] == pytest.approx(5 / 7)
    assert stats["estimated_mean_hard_negatives_per_anchor"] == pytest.approx(2 / 7)
    assert stats["positive_t_histogram"] == {3: 2}
    assert stats["hard_negative_t_histogram"] == {4: 2}

    for anchor_chunk, anchor_game, t, partner_chunk, partner_game in positive_rows:
        anchor = replay_prefix(games[anchor_chunk][anchor_game][:t], 8)
        partner = replay_prefix(games[partner_chunk][partner_game][:t], 8)
        assert full_board_state_key(anchor) == full_board_state_key(partner)
    for anchor_chunk, anchor_game, t, negative_chunk, negative_game in hard_rows:
        anchor = replay_prefix(games[anchor_chunk][anchor_game][:t], 8)
        negative = replay_prefix(games[negative_chunk][negative_game][:t], 8)
        assert occupancy_key(anchor) == occupancy_key(negative)
        assert full_board_state_key(anchor) != full_board_state_key(negative)


def test_surface_hard_negative_mining_adds_near_same_board_negatives(
    tmp_path: Path,
) -> None:
    chunk = tmp_path / "surface.pickle"
    next_action = 42
    _write_chunk(
        chunk,
        [
            [*SURFACE_HARD_NEGATIVE_PREFIXES[0], next_action],
            [*SURFACE_HARD_NEGATIVE_PREFIXES[1], next_action],
        ],
    )
    index_path = tmp_path / "surface.sqlite"
    stats = build_pair_index(
        [chunk],
        index_path,
        board_size=8,
        t_min=8,
        t_max=8,
        positions_per_game=0,
        use_surface_hard_negatives=True,
        surface_jaccard_threshold=0.8,
        surface_max_pairs_per_anchor=1,
    )
    assert stats["natural_hard_negative_count"] == 0
    assert stats["surface_hard_negative_count"] == 2
    assert stats["hard_negative_count"] == 2

    games = pickle.loads(chunk.read_bytes())
    boards = [replay_prefix(game[:8], 8) for game in games]
    assert occupancy_key(boards[0]) != occupancy_key(boards[1])
    assert full_board_state_key(boards[0]) != full_board_state_key(boards[1])
    assert pair_index_module._occupancy_jaccard(
        occupancy_key(boards[0]), occupancy_key(boards[1])
    ) == pytest.approx(0.8461538461538461)

    connection = sqlite3.connect(index_path)
    try:
        rows = connection.execute(
            "SELECT anchor_game, negative_game FROM hard_negative_pairs"
        ).fetchall()
    finally:
        connection.close()
    assert set(rows) == {(0, 1), (1, 0)}

    pairs = OrderAwarePairIndex(index_path, tmp_path, board_size=8)
    try:
        assert pairs.use_surface_hard_negatives is True
        assert pairs.surface_jaccard_threshold == pytest.approx(0.8)
        assert pairs.surface_max_pairs_per_anchor == 1
        pairs.pairs_for_chunk(chunk.name)
        for game_idx in range(2):
            refs = pairs.hard_negative_refs(chunk.name, game_idx, 8)
            assert len(refs) == 1
            assert len(pairs.resolve_prefix_tokens(refs[0])) == 8
    finally:
        pairs.close()


def test_efficient_index_samples_one_balanced_position_per_game(
    tmp_path: Path,
) -> None:
    chunk = tmp_path / "sampled.pickle"
    games = [
        [*POSITIVE_ORDERS[0], 20],
        [*POSITIVE_ORDERS[1], 20],
        [*HARD_NEGATIVE_ORDERS[0], 9],
        [*HARD_NEGATIVE_ORDERS[1], 9],
    ]
    _write_chunk(chunk, games)
    index_path = tmp_path / "sampled.sqlite"
    stats = build_pair_index(
        [chunk],
        index_path,
        board_size=8,
        t_min=3,
        t_max=4,
        positions_per_game=1,
        sampling_seed=0,
    )
    assert stats["n_candidate_occurrences"] == 6
    assert stats["n_occurrences"] == len(games)
    assert stats["sampling_fraction"] == pytest.approx(4 / 6)
    assert sum(stats["indexed_t_histogram"].values()) == len(games)

    pairs = OrderAwarePairIndex(index_path, tmp_path, board_size=8)
    pairs.pairs_for_chunk(chunk.name)
    assert pairs.indexed_anchor_count == len(games)
    assert sum(
        pairs.state_key_or_none(chunk.name, game_idx, t) is not None
        for game_idx in range(len(games))
        for t in (3, 4)
    ) == len(games)
    pairs.close()


def test_sampler_uses_partner_order_and_never_calls_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positive_chunk, _, index_path, _ = _build_known_index(tmp_path)
    monkeypatch.setattr(
        OthelloBoardState,
        "umpire",
        lambda *_args, **_kwargs: pytest.fail("engine called in v8 batch path"),
    )
    batches = OrderAwareChunkBatches(
        positive_chunk,
        index_path,
        tmp_path,
        OrderAwareBatchConfig(
            board_size=8,
            t_min=3,
            t_max=3,
            groups_per_batch=1,
            samples_per_group=2,
            hard_negatives_per_anchor=0,
            seed=0,
        ),
    )
    batch = next(batches)
    group_state_keys = {
        batches.pairs.state_key(positive_chunk.name, int(game_idx), 3)
        for game_idx in batch["game_indices"]
    }
    batches.close()
    assert batch["x_context"].shape == (2, 3)
    assert batch["x_positive_target"].shape == (2, 4)
    assert batch["x_hard_negative_targets"].shape == (2, 0, 4)
    assert int(batch["positive_fallback_mask"].sum()) == 1
    assert len(group_state_keys) == 2
    fallback = batch["positive_fallback_mask"]
    assert torch.equal(
        batch["x_positive_target"][fallback, :-1],
        batch["x_context"][fallback],
    )
    assert not torch.equal(
        batch["x_positive_target"][~fallback, :-1],
        batch["x_context"][~fallback],
    )
    assert torch.equal(
        batch["x_positive_target"][:, -1],
        batch["x_positive_target"][:, -1].new_full((2,), batch["x_positive_target"][0, -1]),
    )


def test_sampler_includes_same_action_hard_negatives(tmp_path: Path) -> None:
    _, hard_chunk, index_path, _ = _build_known_index(tmp_path)
    batches = OrderAwareChunkBatches(
        hard_chunk,
        index_path,
        tmp_path,
        OrderAwareBatchConfig(
            board_size=8,
            t_min=4,
            t_max=4,
            groups_per_batch=1,
            samples_per_group=2,
            hard_negatives_per_anchor=1,
            seed=0,
        ),
    )
    batch = next(batches)
    batches.close()
    assert batch["hard_negative_mask"].all()
    assert batch["mean_hard_negatives"] == 1.0
    assert torch.equal(
        batch["x_hard_negative_targets"][:, 0, -1],
        batch["x_positive_target"][:, -1],
    )


def test_hard_negative_mining_does_not_require_same_observed_next_action(
    tmp_path: Path,
) -> None:
    chunk = tmp_path / "different-actions.pickle"
    _write_chunk(
        chunk,
        [
            [*HARD_NEGATIVE_ORDERS[0], 10],
            [*HARD_NEGATIVE_ORDERS[1], 11],
        ],
    )
    index_path = tmp_path / "different-actions.sqlite"
    stats = build_pair_index(
        [chunk],
        index_path,
        board_size=8,
        t_min=4,
        t_max=4,
    )
    assert stats["hard_negative_count"] == 2

    pairs = OrderAwarePairIndex(index_path, tmp_path, board_size=8)
    pairs.pairs_for_chunk(chunk.name)
    assert pairs.hard_negative_refs(chunk.name, 0, 4)
    assert pairs.hard_negative_refs(chunk.name, 1, 4)
    pairs.close()


def test_sampler_reserves_minimum_pair_anchors_per_group(tmp_path: Path) -> None:
    positive_chunk, _, index_path, _ = _build_known_index(tmp_path)
    batches = OrderAwareChunkBatches(
        positive_chunk,
        index_path,
        tmp_path,
        OrderAwareBatchConfig(
            board_size=8,
            t_min=3,
            t_max=3,
            groups_per_batch=1,
            samples_per_group=2,
            hard_negatives_per_anchor=0,
            min_pair_anchors_per_group=1,
            seed=0,
        ),
    )
    batch = next(batches)
    batches.close()
    assert int((~batch["positive_fallback_mask"]).sum()) >= 1


def test_legacy_index_resolves_pair_prefixes_without_loading_partner_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positive_chunk, _, index_path, _ = _build_known_index(tmp_path)
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("UPDATE positive_pairs SET partner_prefix = NULL")
        connection.execute("UPDATE hard_negative_pairs SET negative_prefix = NULL")
        connection.execute(
            "UPDATE metadata SET value = 'false' "
            "WHERE key = 'embedded_pair_prefixes'"
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(
        pair_index_module,
        "load_chunk",
        lambda *_args, **_kwargs: pytest.fail("partner chunk loaded at runtime"),
    )
    batches = OrderAwareChunkBatches(
        positive_chunk,
        index_path,
        tmp_path,
        OrderAwareBatchConfig(
            board_size=8,
            t_min=3,
            t_max=3,
            groups_per_batch=1,
            samples_per_group=2,
            hard_negatives_per_anchor=0,
            seed=0,
        ),
    )
    batch = next(batches)
    batches.close()
    assert int((~batch["positive_fallback_mask"]).sum()) == 1


def _random_legal_game(rng: np.random.Generator, max_moves: int = 40) -> list[int]:
    board = OthelloBoardState(n=8)
    moves: list[int] = []
    while len(moves) < max_moves:
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
        except Exception:  # noqa: BLE001 - terminal positions are fine to skip
            break
        moves.append(move)
    return moves


def _build_synthetic_chunk_and_index(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    games = [_random_legal_game(rng, max_moves=40) for _ in range(80)]
    games = [g for g in games if len(g) >= 12]
    chunk_path = tmp_path / "synthetic.pickle"
    with open(chunk_path, "wb") as handle:
        pickle.dump(games, handle)
    index_path = tmp_path / "synthetic_index.sqlite"
    build_pair_index(
        [chunk_path],
        index_path,
        board_size=8,
        t_min=4,
        t_max=10,
        positions_per_game=0,  # exhaustive indexing -> many anchors
    )
    return chunk_path, index_path


def _iterate_batches(
    chunk_path: Path,
    index_path: Path,
    data_dir: Path,
    *,
    fast: bool,
    max_batches: int = 6,
) -> list[dict[str, object]]:
    batches = OrderAwareChunkBatches(
        chunk_path,
        index_path,
        data_dir,
        OrderAwareBatchConfig(
            board_size=8,
            t_min=4,
            t_max=10,
            groups_per_batch=2,
            samples_per_group=2,
            hard_negatives_per_anchor=1,
            chunk_sample_multiplier=2.0,
            seed=123,
        ),
    )
    batches._use_fast_viability = fast  # type: ignore[attr-defined]
    collected: list[dict[str, object]] = []
    for _ in range(max_batches):
        try:
            collected.append(next(batches))
        except StopIteration:
            break
    batches.close()
    return collected


def test_incremental_viability_is_deterministic_and_matches_naive_budget(
    tmp_path: Path,
) -> None:
    """Fast viability bookkeeping preserves the sampler contract.

    Active state buckets use O(1) swap-removal, so their internal order can
    differ from the naive full-scan implementation. The same seed must remain
    deterministic, while fast and slow paths must emit the same number and
    shapes of batches.
    """
    chunk_path, index_path = _build_synthetic_chunk_and_index(tmp_path)
    fast_batches = _iterate_batches(
        chunk_path, index_path, tmp_path, fast=True, max_batches=6
    )
    repeated_fast_batches = _iterate_batches(
        chunk_path, index_path, tmp_path, fast=True, max_batches=6
    )
    slow_batches = _iterate_batches(
        chunk_path, index_path, tmp_path, fast=False, max_batches=6
    )
    assert len(fast_batches) == len(slow_batches) >= 1, (
        len(fast_batches),
        len(slow_batches),
    )
    for fast_batch, repeated_fast, slow_batch in zip(
        fast_batches,
        repeated_fast_batches,
        slow_batches,
    ):
        for key in (
            "x_context",
            "x_positive_target",
            "x_hard_negative_targets",
            "hard_negative_mask",
            "group_ids",
            "positive_fallback_mask",
            "game_indices",
        ):
            assert torch.equal(fast_batch[key], repeated_fast[key]), key
            assert fast_batch[key].shape == slow_batch[key].shape, key


def test_v8_gradients_ema_and_action_conditioning() -> None:
    torch.manual_seed(0)
    model = OthelloJEPAOrderAware(_tiny_cfg(lambda_ce=0.05))
    batch = _objective_batch(model)
    out = model(**batch)
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.action_embedding.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert any(p.grad is not None for p in model.next_action_head.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())

    repeated_context = out["context_latent"][0:1].detach().expand(2, -1)
    predictions = model.predict(repeated_context, torch.tensor([1, 2]))
    assert not torch.allclose(predictions[0], predictions[1])

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(0.1)
    model.update_target_encoder(momentum=0.5)
    assert not torch.allclose(
        before,
        next(model.target_encoder.parameters()).detach(),
    )


def test_v8_target_encoder_skips_masked_hard_negative_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = OthelloJEPAOrderAware(_tiny_cfg())
    batch = _objective_batch(model, batch_size=4, hard_count=2)
    batch["hard_negative_mask"][:] = False
    batch["hard_negative_mask"][0, 0] = True
    target_batch_sizes: list[int] = []
    original_encode_hidden = model.encode_hidden

    def recording_encode_hidden(encoder, idx):
        if encoder is model.target_encoder:
            target_batch_sizes.append(int(idx.size(0)))
        return original_encode_hidden(encoder, idx)

    monkeypatch.setattr(model, "encode_hidden", recording_encode_hidden)
    out = model(**batch)
    assert torch.isfinite(out["loss"])
    assert target_batch_sizes == [5]


def test_order_aware_infonce_random_and_perfect_cases() -> None:
    torch.manual_seed(0)
    groups, members, dim, hard_count = 2, 4, 16, 2
    batch_size = groups * members
    group_ids = torch.arange(groups).repeat_interleave(members)
    predictions = torch.randn(batch_size, dim)
    positives = torch.randn(batch_size, dim)
    hard = torch.randn(batch_size, hard_count, dim)
    hard_mask = torch.ones(batch_size, hard_count, dtype=torch.bool)
    random_loss, _, mean_hard = order_aware_infonce_loss(
        predictions,
        positives,
        hard,
        hard_mask,
        group_ids,
        temperature=100.0,
    )
    assert abs(float(random_loss) - math.log(members + hard_count)) < 0.02
    assert float(mean_hard) == hard_count

    perfect = torch.eye(members).repeat(groups, 1)
    no_hard = torch.empty(batch_size, 0, members)
    no_hard_mask = torch.empty(batch_size, 0, dtype=torch.bool)
    perfect_loss, perfect_accuracy, _ = order_aware_infonce_loss(
        perfect,
        perfect,
        no_hard,
        no_hard_mask,
        group_ids,
        temperature=0.01,
    )
    assert float(perfect_loss) < 1e-4
    assert float(perfect_accuracy) == 1.0


def test_lambda_ce_zero_does_not_train_ce_head() -> None:
    model = OthelloJEPAOrderAware(_tiny_cfg(lambda_ce=0.0))
    out = model(**_objective_batch(model))
    out["loss"].backward()
    assert float(out["hybrid_ce_loss"]) == 0.0
    assert all(parameter.grad is None for parameter in model.next_action_head.parameters())


@pytest.mark.parametrize(
    ("config_name", "lambda_ce"),
    [
        ("jepa_v8_order_aware.yml", 0.0),
        ("jepa_v8_improvement.yml", 0.0),
        ("jepa_v8_improvement_hard_negatives.yml", 0.0),
        ("jepa_v8_order_aware_hybrid.yml", 0.05),
    ],
)
def test_v8_yaml_configs_resolve_and_build(
    config_name: str,
    lambda_ce: float,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data = tjm.load_yaml_config(repo_root / "configs" / config_name)
    parser = tjm.build_parser()
    args = parser.parse_args(
        [
            *tjm.config_to_cli_args(data, parser),
            "--out_dir",
            "__unused__",
            "--n_layers",
            "1",
            "--n_heads",
            "4",
            "--d_model",
            "32",
            "--dropout",
            "0.0",
            "--action_dim",
            "16",
            "--predictor_hidden_mult",
            "2",
            "--predictor_n_layers",
            "2",
            "--batch_size",
            "4",
            "--order_groups_per_batch",
            "2",
            "--order_samples_per_group",
            "2",
            "--order_min_pair_anchors_per_group",
            "1",
            "--precision",
            "fp32",
            "--probe_interval_steps",
            "0",
            "--drive_sync_dir",
            "",
        ]
    )
    cfg, _ = tjm.namespace_to_train_config(args)
    tjm.validate_train_config(cfg)
    model, objective_cfg = tjm.build_objective_model(cfg)
    assert isinstance(model, OthelloJEPAOrderAware)
    assert cfg.variant == "v8"
    assert cfg.view_mode == "order_aware"
    assert objective_cfg.lambda_ce == lambda_ce


def test_v8_training_dispatch_runs_optimizer_and_ema_step(tmp_path: Path) -> None:
    positive_chunk, _, index_path, _ = _build_known_index(tmp_path)
    cfg = tjm.TrainConfig(
        out_dir="__unused__",
        data_dir=str(tmp_path),
        pair_index_path=str(index_path),
        objective_class="jepa_order_aware",
        variant="v8",
        predictor_type="action_conditioned_mlp",
        n_layers=1,
        n_heads=4,
        d_model=32,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        predictor_dropout=0.0,
        action_dim=16,
        loss_type="order_aware_infonce",
        prediction_horizon=1,
        view_mode="order_aware",
        order_t_min=3,
        order_t_max=3,
        order_groups_per_batch=1,
        order_samples_per_group=2,
        order_hard_negatives_per_anchor=0,
        order_index_positions_per_game=0,
        batch_size=2,
        warmup_steps=1,
        precision="fp32",
        probe_interval_steps=0,
    )
    model, _ = tjm.build_objective_model(cfg)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    target_before = next(model.target_encoder.parameters()).detach().clone()
    result = tjm.dispatch_train_one_chunk(
        model,
        optimizer,
        scaler,
        positive_chunk,
        cfg,
        model.config,
        torch.device("cpu"),
        0,
        10,
    )
    train_loss, metrics, step = result[:3]
    assert step == 1
    assert math.isfinite(train_loss)
    assert metrics["positive_fallback_rate"] == 0.5
    assert not torch.allclose(
        target_before,
        next(model.target_encoder.parameters()).detach(),
    )


EXISTING_V1_TO_V7_CONFIGS = [
    "jepa_linear_k1_ema_b8_hard_disjoint.yml",
    "jepa_no_predictor_k1_vicreg_b8_hard_disjoint.yml",
    "jepa_v1_vicreg_b8.yml",
    "jepa_v1_vicreg_b8_hard_disjoint.yml",
    "jepa_v2_ema_b8.yml",
    "jepa_v2_ema_b8_hard_disjoint.yml",
    "jepa_v3_mlp_k8_b8.yml",
    "jepa_v4_multi_pos_mlp_k4_b8.yml",
    "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint.yml",
    "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint_vicreg.yml",
    "jepa_v5_contrastive.yml",
    "jepa_v5_contrastive_hard_disjoint.yml",
    "jepa_v6_hard_disjoint_action.yml",
    "jepa_v7_action_smooth_l1.yml",
    "jepa_v7_action_infonce.yml",
    "jepa_v7_action_hybrid.yml",
]


@pytest.mark.parametrize("config_name", EXISTING_V1_TO_V7_CONFIGS)
def test_every_existing_v1_to_v7_config_still_runs_one_step(
    config_name: str,
    synthetic_game_batch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data = tjm.load_yaml_config(repo_root / "configs" / config_name)
    parser = tjm.build_parser()
    args = parser.parse_args(
        [
            *tjm.config_to_cli_args(data, parser),
            "--out_dir",
            "__unused__",
            "--n_layers",
            "1",
            "--n_heads",
            "4",
            "--d_model",
            "32",
            "--dropout",
            "0.0",
            "--predictor_hidden_dim",
            "32",
            "--predictor_hidden_mult",
            "2",
            "--predictor_n_layers",
            "2",
            "--predictor_n_heads",
            "4",
            "--predictor_dropout",
            "0.0",
            "--action_dim",
            "16",
            "--batch_size",
            "4",
            "--action_groups_per_batch",
            "2",
            "--action_samples_per_group",
            "2",
            "--precision",
            "fp32",
            "--eval_chunks",
            "0",
            "--probe_interval_steps",
            "0",
            "--drive_sync_dir",
            "",
        ]
    )
    cfg, _ = tjm.namespace_to_train_config(args)
    model, _ = tjm.build_objective_model(cfg)
    batch = synthetic_game_batch(B=4, T=24, board_size=8, seed=1)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)

    cls_name = tjm.objective_class(cfg)
    if cls_name in {"jepa_hard_disjoint_action", "jepa_hard_disjoint_infonce"}:
        prefix_ids = torch.tensor([0, 0, 1, 1])
        context = batch[:, :6].clone()
        context[1] = context[0]
        context[3] = context[2]
        out = model(context, batch[:, 6], prefix_ids)
    elif cls_name == "jepa_action_conditioned":
        context = batch[:, :6]
        target = batch[:, :7]
        group_ids = (
            torch.tensor([0, 0, 1, 1])
            if cfg.action_loss_mode in {"grouped_infonce", "hybrid"}
            else None
        )
        out = model(context, target, group_ids=group_ids)
    else:
        context, target, target_positions = tjm.build_jepa_views(
            cfg,
            batch,
            context_length=6,
            horizon=cfg.prediction_horizon,
        )
        out = model(context, target, target_positions)
    out["loss"].backward()
    optimizer.step()
    assert torch.isfinite(out["loss"])
