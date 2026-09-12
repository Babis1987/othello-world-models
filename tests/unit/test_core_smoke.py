"""Small, source-independent smoke tests for the thesis runtime.

These tests intentionally import only :mod:`othello_thesis` and provide a
compact behavioural safety net for a standalone checkout of the repository.
"""

from __future__ import annotations

import pytest
import torch

from othello_thesis.data.move_vocabulary import (
    build_mappings,
    raw_to_token_list,
    token_to_raw_list,
)
from othello_thesis.game_engine.board import OthelloBoardState
from othello_thesis.models.mamba import (
    MambaARBlock,
    MambaARConfig,
    OthelloMambaAR,
)
from othello_thesis.models.transformer import (
    GPTConfig,
    OthelloGPT,
)
from othello_thesis.objectives.jepa import JEPAConfig, OthelloJEPA


BOARD_SIZES = (8, 12, 16)


def _next_move_with_implicit_pass(board: OthelloBoardState) -> int | None:
    valid = board.get_valid_moves()
    if valid:
        return valid[0]
    board.next_hand_color *= -1
    try:
        opponent_valid = board.get_valid_moves()
    finally:
        board.next_hand_color *= -1
    return opponent_valid[0] if opponent_valid else None


@pytest.mark.parametrize("board_size", BOARD_SIZES)
def test_board_engine_completes_a_deterministic_legal_game(board_size: int) -> None:
    board = OthelloBoardState(board_size)
    for _ in range(board_size * board_size - 4 + 1):
        move = _next_move_with_implicit_pass(board)
        if move is None:
            break
        board.umpire(move)
    else:  # pragma: no cover - protects against a non-terminating engine
        raise AssertionError("Deterministic game did not terminate")

    assert board.is_game_over()
    assert len(board.history) <= board_size * board_size - 4
    assert board.get_winner() in {-1, 0, 1}


@pytest.mark.parametrize("board_size", BOARD_SIZES)
def test_move_vocabulary_round_trip(board_size: int) -> None:
    raw_to_token, token_to_raw = build_mappings(board_size)
    assert len(raw_to_token) == board_size * board_size
    assert len(token_to_raw) == board_size * board_size - 4
    sample = token_to_raw[:12]
    tokens = raw_to_token_list(sample, board_size)
    assert token_to_raw_list(tokens, board_size) == sample


def _build_ar_model(architecture: str) -> torch.nn.Module:
    if architecture == "transformer":
        return OthelloGPT(
            GPTConfig(
                board_size=8,
                n_layers=1,
                n_heads=4,
                d_model=16,
                dropout=0.0,
            )
        )
    return OthelloMambaAR(
        MambaARConfig(
            board_size=8,
            n_layers=1,
            d_model=16,
            d_state=4,
            d_conv=3,
            expand=2,
            dropout=0.0,
            mlp_hidden_mult=0,
            mamba_backend="torch",
        )
    )


@pytest.mark.parametrize("architecture", ("transformer", "mamba"))
def test_ar_forward_backward_and_optimizer_step(architecture: str) -> None:
    torch.manual_seed(7)
    model = _build_ar_model(architecture).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    before = model.lm_head.weight.detach().clone()

    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 4, 61)
    assert loss is not None and torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)


def _build_jepa(architecture: str) -> OthelloJEPA:
    return OthelloJEPA(
        JEPAConfig(
            variant="jepa_v1",
            loss_type="infonce",
            view_mode="hard_disjoint_future",
            board_size=8,
            n_layers=1,
            n_heads=4,
            d_model=16,
            dropout=0.0,
            encoder_architecture=architecture,
            d_state=4,
            d_conv=3,
            expand=2,
            mamba_backend="torch",
            predictor_type="linear",
            prediction_horizon=1,
            ema_momentum=0.996,
            use_ema_target=True,
            contrastive_temperature=0.1,
        )
    )


@pytest.mark.parametrize("architecture", ("transformer", "mamba"))
def test_jepa_forward_backward_optimizer_and_ema_step(architecture: str) -> None:
    torch.manual_seed(11)
    model = _build_jepa(architecture).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    context = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    )
    target = torch.tensor([[12], [13], [14]])
    positions = torch.full((3, 1), 4, dtype=torch.long)
    target_before = next(model.target_encoder.parameters()).detach().clone()

    output = model(context, target, positions)
    assert torch.isfinite(output["loss"])
    optimizer.zero_grad(set_to_none=True)
    output["loss"].backward()
    optimizer.step()
    model.update_target_encoder()

    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )
    assert not torch.equal(
        target_before,
        next(model.target_encoder.parameters()),
    )


def test_mamba_length_one_fast_path_survives_parity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MambaARConfig(
        board_size=8,
        n_layers=1,
        d_model=16,
        d_state=4,
        d_conv=3,
        expand=2,
        dropout=0.0,
        mlp_hidden_mult=0,
        mamba_backend="torch",
    )
    block = MambaARBlock(config, layer_idx=0).eval()
    reference = block.mixer.forward

    def drifted_reference(x: torch.Tensor) -> torch.Tensor:
        return reference(x) + 0.25

    monkeypatch.setattr(block.mixer, "forward", drifted_reference)
    with pytest.warns(RuntimeWarning):
        output = block(torch.randn(3, 1, config.d_model))

    assert torch.isfinite(output).all()
    assert block._length_one_fast_path_checked is True
    assert block._length_one_fast_path_enabled is True
