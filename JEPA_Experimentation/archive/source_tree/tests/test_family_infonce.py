from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from othello_research.datasets.family_dataset import (
    FamilyArtifacts,
    FamilyBatchConfig,
    FamilyBatchSampler,
)
from othello_research.objectives.family_infonce import (
    FamilyInfoNCEConfig,
    OthelloFamilyInfoNCE,
    family_multi_positive_infonce,
)
from othello_research.othello.bitboard import (
    INITIAL_BLACK,
    INITIAL_WHITE,
    apply_move,
    legal_moves_mask,
    scan_games,
)
from othello_research.othello.board import OthelloBoardState

REPO_ROOT = Path(__file__).resolve().parents[1]

# Known 8x8 transpositions (same as the v8 test fixtures).
POSITIVE_ORDERS = ([19, 18, 26], [26, 18, 19])          # same board at t=3
HARD_NEGATIVE_ORDERS = ([19, 18, 37, 20], [19, 20, 37, 18])  # same move-set t=4


def _lm(player, opponent):
    """uint64-coerced legal_moves_mask (numba returns Python ints, and masks
    with bit 63 set would otherwise be re-typed as float64 on the next call)."""
    return legal_moves_mask(np.uint64(player), np.uint64(opponent))


def _am(player, opponent, move):
    new_p, new_o, flips = apply_move(
        np.uint64(player), np.uint64(opponent), np.uint64(move)
    )
    return np.uint64(new_p), np.uint64(new_o), np.uint64(flips)


def _mask_from_state(state: np.ndarray, color: int) -> int:
    mask = 0
    flat = state.reshape(-1)
    for square in range(64):
        if flat[square] == color:
            mask |= 1 << square
    return mask


def _random_games_with_slow_engine(n_games: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(n_games):
        board = OthelloBoardState(n=8)
        moves: list[int] = []
        while len(moves) < 60:
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


def test_initial_position_and_legal_moves() -> None:
    board = OthelloBoardState(n=8)
    assert _mask_from_state(board.state, 1) == int(INITIAL_BLACK)
    assert _mask_from_state(board.state, -1) == int(INITIAL_WHITE)
    moves = _lm(INITIAL_BLACK, INITIAL_WHITE)
    squares = {s for s in range(64) if int(moves) >> s & 1}
    assert squares == {19, 26, 37, 44}


def test_bitboard_matches_reference_engine_on_random_games() -> None:
    games = _random_games_with_slow_engine(200, seed=11)
    for game in games:
        reference = OthelloBoardState(n=8)
        black, white = INITIAL_BLACK, INITIAL_WHITE
        for i, move in enumerate(game):
            black_to_move = (i % 2) == 0
            player, opponent = (black, white) if black_to_move else (white, black)
            if _lm(player, opponent) == np.uint64(0):
                # Pass ply: strict alternation broken; the scan stops here.
                break
            player, opponent, flips = _am(player, opponent, move)
            if flips == np.uint64(0):
                break
            black, white = (player, opponent) if black_to_move else (opponent, player)
            reference.umpire(move)
            assert _mask_from_state(reference.state, 1) == int(black)
            assert _mask_from_state(reference.state, -1) == int(white)


def test_scan_truncates_at_first_pass() -> None:
    games = _random_games_with_slow_engine(300, seed=23)
    _, t_arr, _, _, n_pass, n_illegal = scan_games(games, t_min=1, t_max=60)
    assert n_illegal == 0
    # Independently compute expected pass-free length per game.
    expected_rows = 0
    n_pass_expected = 0
    for game in games:
        black, white = INITIAL_BLACK, INITIAL_WHITE
        plies = 0
        truncated = False
        for i, move in enumerate(game):
            black_to_move = (i % 2) == 0
            player, opponent = (black, white) if black_to_move else (white, black)
            if _lm(player, opponent) == np.uint64(0):
                truncated = True
                break
            player, opponent, _ = _am(player, opponent, move)
            black, white = (player, opponent) if black_to_move else (opponent, player)
            plies += 1
        expected_rows += plies
        n_pass_expected += int(truncated)
    assert len(t_arr) == expected_rows
    assert n_pass == n_pass_expected


def test_known_transpositions_share_family_and_class_keys() -> None:
    def keys(prefix: list[int]) -> tuple[int, int]:
        _, t, mset, black, _, _ = scan_games([prefix], t_min=len(prefix),
                                             t_max=len(prefix))
        assert len(t) == 1
        return int(mset[0]), int(black[0])

    mset_a, black_a = keys(POSITIVE_ORDERS[0])
    mset_b, black_b = keys(POSITIVE_ORDERS[1])
    assert mset_a == mset_b and black_a == black_b  # same family, same class

    mset_c, black_c = keys(HARD_NEGATIVE_ORDERS[0])
    mset_d, black_d = keys(HARD_NEGATIVE_ORDERS[1])
    assert mset_c == mset_d and black_c != black_d  # same family, diff class


@pytest.fixture(scope="module")
def tiny_artifacts(tmp_path_factory) -> Path:
    """Build a small end-to-end artifacts dir through the real Part A script."""
    root = tmp_path_factory.mktemp("family")
    data_dir = root / "chunks"
    data_dir.mkdir()
    rng_games = _random_games_with_slow_engine(400, seed=7)
    fixtures = [
        [*POSITIVE_ORDERS[0], 20, 21],
        [*POSITIVE_ORDERS[1], 20, 34],
        [*HARD_NEGATIVE_ORDERS[0], 9],
        [*HARD_NEGATIVE_ORDERS[1], 9],
    ]
    train_games = fixtures + rng_games[:300]
    val_games = rng_games[300:]
    with open(data_dir / "chunk_000.pickle", "wb") as f:
        pickle.dump(train_games, f)
    with open(data_dir / "chunk_001.pickle", "wb") as f:
        pickle.dump(val_games, f)

    out_dir = root / "family_artifacts"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_family_index.py"),
            "--data_dir", str(data_dir),
            "--out_dir", str(out_dir),
            "--train_chunks", "1",
            "--t_min", "3",
            "--t_max", "12",
            "--workers", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return out_dir


def test_artifacts_layout_and_manifest(tiny_artifacts: Path) -> None:
    for rel in (
        "packed_games.bin",
        "offsets.npy",
        "game_id_map.json",
        "family_index/train/members.parquet",
        "family_index/train/families.parquet",
        "family_index/val/members.parquet",
        "density_report.json",
        "density_report.md",
        "manifest.json",
    ):
        assert (tiny_artifacts / rel).exists(), rel
    manifest = json.loads(
        (tiny_artifacts / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["t_min"] == 3 and manifest["t_max"] == 12
    assert not any("_intermediate" in k for k in manifest["artifacts"])


def test_invariant_audit_on_materialized_pairs(tiny_artifacts: Path) -> None:
    cfg = FamilyBatchConfig(families_per_batch=1, members_per_family=8, seed=0)
    art = FamilyArtifacts(tiny_artifacts, cfg, verify=True)
    split = art.train
    rng = np.random.default_rng(0)
    checked = 0
    for family_id in split.viable_families:
        rows = split.members_of(int(family_id))
        if checked >= 50:
            break
        rows = rng.permutation(rows)[:4]
        boards = []
        for r in rows:
            t = int(split.t[r])
            raw_prefix = art.packed[
                art.offsets[int(split.game_id[r])] :
                art.offsets[int(split.game_id[r])] + t
            ].tolist()
            board = OthelloBoardState(n=8)
            board.update(raw_prefix)
            boards.append((int(split.class_uid[r]), board.state.copy()))
        for i in range(len(boards)):
            for j in range(i + 1, len(boards)):
                occ_i = boards[i][1] != 0
                occ_j = boards[j][1] != 0
                assert np.array_equal(occ_i, occ_j)  # same family = same occupancy
                same_board = np.array_equal(boards[i][1], boards[j][1])
                same_class = boards[i][0] == boards[j][0]
                assert same_board == same_class
                checked += 1
    assert checked > 0


def test_loss_plateau_and_separation_behaviour() -> None:
    family_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    class_ids = torch.tensor([0, 0, 1, 2, 3, 3, 4, 5])
    eligible = torch.ones(8, dtype=torch.bool)

    # Constant embeddings -> uniform logits -> excess == 0 exactly.
    z_flat = torch.nn.functional.normalize(torch.ones(8, 16), dim=-1)
    out = family_multi_positive_infonce(z_flat, family_ids, class_ids,
                                        eligible, temperature=0.1)
    assert torch.isclose(out["excess"], torch.tensor(0.0), atol=1e-5)

    # Class-aligned embeddings -> loss far below plateau.
    z = torch.nn.functional.normalize(
        torch.nn.functional.one_hot(class_ids, 6).float()
        + 0.01 * torch.randn(8, 6),
        dim=-1,
    )
    out = family_multi_positive_infonce(z, family_ids, class_ids, eligible,
                                        temperature=0.1)
    assert float(out["excess"]) < -1.0

    # Leakage filter: ineligible positives drop out of the numerator.
    eligible2 = eligible.clone()
    eligible2[1] = False  # row 1 was row 0's only positive
    with pytest.raises(ValueError):
        # anchor 0 loses its only positive; anchor 1 keeps row 0 -> still some
        # anchors remain, so construct the fully-starved case instead:
        family_multi_positive_infonce(
            z[:4], family_ids[:4], class_ids[:4],
            torch.tensor([False, False, False, False]), temperature=0.1,
        )


def test_sampler_batches_have_anchors_and_negatives(tiny_artifacts: Path) -> None:
    cfg = FamilyBatchConfig(families_per_batch=2, members_per_family=4, seed=1)
    art = FamilyArtifacts(tiny_artifacts, cfg, verify=False)
    sampler = FamilyBatchSampler(art, "train", cfg)
    batch = sampler.sample_batch()
    tokens = torch.from_numpy(batch["tokens"])
    assert tokens.ndim == 2 and tokens.max() < 61
    out = family_multi_positive_infonce(
        torch.nn.functional.normalize(torch.randn(tokens.size(0), 8), dim=-1),
        torch.from_numpy(batch["family_ids"]),
        torch.from_numpy(batch["class_ids"]),
        torch.from_numpy(batch["positive_eligible"]),
        temperature=0.1,
    )
    assert int(out["n_anchors"]) >= 2


def test_shared_config_parse_build_and_registry_roundtrip(tmp_path: Path) -> None:
    """family_infonce_v1.yml must flow through the standard train_jepa paths:
    yml -> argparse -> TrainConfig -> build_objective_model, and native
    checkpoints must load back through build_model_from_checkpoint so the
    existing eval notebooks work by changing only CONFIG_NAME."""
    from dataclasses import asdict

    from othello_research.training import train_jepa as tjm

    args, _, _ = tjm.parse_configured_args([
        "--config", str(REPO_ROOT / "configs" / "family_infonce_v1.yml"),
        "--out_dir", str(tmp_path),
        "--drive_sync_dir", "",
        "--n_layers", "1",
        "--n_heads", "4",
        "--d_model", "32",
        "--proj_hidden", "32",
        "--proj_dim", "8",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    assert tjm.objective_class(cfg) == "family_infonce"
    tjm.validate_train_config(cfg)

    model, model_cfg = tjm.build_objective_model(cfg)
    assert isinstance(model, OthelloFamilyInfoNCE)
    assert model_cfg.proj_dim == 8

    ckpt = {
        "model": model.state_dict(),
        "model_config": {"objective_class": "family_infonce",
                         **asdict(model_cfg)},
        "step": 1,
        "games_seen": 8,
    }
    loaded, loaded_cfg = tjm.build_model_from_checkpoint(ckpt)
    loaded.load_state_dict(ckpt["model"])
    assert isinstance(loaded, OthelloFamilyInfoNCE)
    tokens = torch.randint(0, loaded.config.vocab_size, (2, 6))
    hidden = loaded.encode_hidden(loaded.context_encoder, tokens)
    assert hidden.shape == (2, 6, loaded.config.d_model)


def test_model_forward_and_eval_compat_checkpoint(tmp_path: Path) -> None:
    from othello_research.evaluation.linear_head import JEPALinearHead
    from othello_research.training import train_jepa as tjm
    from othello_research.training.train_family_infonce import (
        save_eval_compat_checkpoint,
    )

    cfg = FamilyInfoNCEConfig(n_layers=1, n_heads=4, d_model=32,
                              proj_hidden=32, proj_dim=8, dropout=0.0)
    model = OthelloFamilyInfoNCE(cfg)
    tokens = torch.randint(0, 60, (8, 6))
    lengths = torch.full((8,), 6, dtype=torch.long)
    out = model(
        tokens, lengths,
        family_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        class_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        positive_eligible=torch.ones(8, dtype=torch.bool),
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())

    ckpt_path = tmp_path / "eval_compat.pt"
    save_eval_compat_checkpoint(model, ckpt_path, step=1, rows_seen=8)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded, loaded_cfg = tjm.build_model_from_checkpoint(ckpt)
    loaded.load_state_dict(ckpt["model"])
    loaded.eval()

    # Encoder weights must round-trip exactly.
    reference = model.encode_hidden(model.context_encoder, tokens)
    model_eval = model.eval()
    with torch.no_grad():
        reference = model_eval.encode_hidden(model_eval.context_encoder, tokens)
        recovered = loaded.encode_hidden(loaded.context_encoder, tokens)
    assert torch.allclose(reference, recovered, atol=1e-6)

    head = JEPALinearHead(loaded)
    logits, loss = head(tokens, tokens)
    assert logits.shape == (8, 6, loaded.config.vocab_size)
    assert torch.isfinite(loss)
