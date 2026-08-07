"""Nanda-style causal edits for Othello residual-stream board directions.

The intervention probe in this module is deliberately a *raw* linear relative
board probe.  The normalized probes used for the headline decodability tables
contain a per-example LayerNorm, whose classifier weights are not fixed
directions in the encoder's raw residual space.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

from othello_thesis.data.chunk_dataset import load_chunk
from othello_thesis.data.move_vocabulary import build_mappings
from othello_thesis.evaluation.position_probe import PositionProbeBank
from othello_thesis.game_engine.board import OthelloBoardState


Operation = Literal["flip", "erase"]


@dataclass(frozen=True)
class InterventionCase:
    chunk_path: str
    game_index: int
    prefix_length: int
    square: int
    source_class: int
    target_class: int
    operation: Operation
    prefix_raw: tuple[int, ...]
    original_legal_tokens: tuple[int, ...]
    target_legal_tokens: tuple[int, ...]

    @property
    def case_id(self) -> str:
        payload = (
            f"{self.chunk_path}:{self.game_index}:{self.prefix_length}:"
            f"{self.square}:{self.operation}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "case_id": self.case_id}


def _effective_side(board: OthelloBoardState) -> int | None:
    if board.get_valid_moves():
        return int(board.next_hand_color)
    board.next_hand_color *= -1
    try:
        return int(board.next_hand_color) if board.get_valid_moves() else None
    finally:
        board.next_hand_color *= -1


def _legal_tokens(
    board: OthelloBoardState,
    raw_to_token: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            int(raw_to_token[move])
            for move in board.get_valid_moves()
            if int(raw_to_token[move]) >= 0
        )
    )


def _counterfactual_case(
    *,
    chunk_path: Path,
    game_index: int,
    game: Sequence[int],
    prefix_length: int,
    operation: Operation,
    board_size: int,
    rng: random.Random,
) -> InterventionCase | None:
    board = OthelloBoardState(board_size)
    board.update(game[:prefix_length])
    side = _effective_side(board)
    if side is None:
        return None
    board.next_hand_color = side
    raw_to_token, _ = build_mappings(board_size)
    original_legal = _legal_tokens(board, raw_to_token)
    if not original_legal:
        return None

    occupied = [
        square
        for square, value in enumerate(board.get_state())
        if int(value) != 0
    ]
    rng.shuffle(occupied)
    for square in occupied:
        value = int(board.state.flat[square])
        source_class = 0 if value == side else 1
        target_class = (1 - source_class) if operation == "flip" else 2
        target = OthelloBoardState(board_size)
        target.state = np.array(board.state, copy=True)
        target.next_hand_color = side
        target.history = list(game[:prefix_length])
        target.state.flat[square] = -value if operation == "flip" else 0
        target_legal = _legal_tokens(target, raw_to_token)
        if not target_legal or target_legal == original_legal:
            continue
        return InterventionCase(
            chunk_path=str(chunk_path),
            game_index=game_index,
            prefix_length=prefix_length,
            square=square,
            source_class=source_class,
            target_class=target_class,
            operation=operation,
            prefix_raw=tuple(int(move) for move in game[:prefix_length]),
            original_legal_tokens=original_legal,
            target_legal_tokens=target_legal,
        )
    return None


def build_intervention_cases(
    chunks: Sequence[str | Path],
    *,
    board_size: int,
    n_cases: int,
    seed: int,
) -> tuple[InterventionCase, ...]:
    """Build deterministic flip/erase cases whose legal set actually changes."""
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    cases: list[InterventionCase] = []
    for chunk_number, raw_path in enumerate(chunks):
        chunk_path = Path(raw_path)
        games = load_chunk(str(chunk_path))
        for game_index, game in enumerate(games):
            if len(game) < 8:
                continue
            local_seed = int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{chunk_number}:{game_index}".encode("utf-8"),
                    digest_size=8,
                ).digest(),
                "little",
            )
            rng = random.Random(local_seed)
            prefix_length = rng.randint(4, len(game) - 2)
            preferred: Operation = "flip" if len(cases) % 2 == 0 else "erase"
            case = _counterfactual_case(
                chunk_path=chunk_path,
                game_index=game_index,
                game=game,
                prefix_length=prefix_length,
                operation=preferred,
                board_size=board_size,
                rng=rng,
            )
            if case is None:
                fallback: Operation = "erase" if preferred == "flip" else "flip"
                case = _counterfactual_case(
                    chunk_path=chunk_path,
                    game_index=game_index,
                    game=game,
                    prefix_length=prefix_length,
                    operation=fallback,
                    board_size=board_size,
                    rng=rng,
                )
            if case is not None:
                cases.append(case)
            if len(cases) >= n_cases:
                return tuple(cases)
        del games
    raise ValueError(f"Built only {len(cases)} intervention cases; need {n_cases}")


def _token_batch(
    cases: Sequence[InterventionCase],
    *,
    board_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_to_token, _ = build_mappings(board_size)
    rows = [
        [int(raw_to_token[move]) for move in case.prefix_raw]
        for case in cases
    ]
    lengths = torch.tensor([len(row) for row in rows], device=device)
    pad_token = board_size * board_size - 4
    tokens = torch.full(
        (len(rows), int(lengths.max().item())),
        pad_token,
        dtype=torch.long,
        device=device,
    )
    for row_index, row in enumerate(rows):
        tokens[row_index, : len(row)] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
    return tokens, lengths - 1


def _probe_directions(
    probe_bank: PositionProbeBank,
    cases: Sequence[InterventionCase],
    *,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    probe = probe_bank.relative[str(layer)]
    if not isinstance(probe.input_norm, nn.Identity):
        raise TypeError("Causal intervention requires a raw, unnormalized probe")
    if not isinstance(probe.proj, nn.Linear):
        raise TypeError("Causal intervention requires a linear relative probe")
    d_model = int(probe.proj.weight.shape[1])
    weights = probe.proj.weight.view(probe.n_squares, 3, d_model)
    directions = torch.stack(
        [
            weights[case.square, case.target_class]
            - weights[case.square, case.source_class]
            for case in cases
        ]
    ).to(device=device, dtype=torch.float32)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _random_directions_like(
    reference: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_directions = torch.randn(
        reference.shape,
        generator=generator,
        dtype=torch.float32,
    ).to(reference.device)
    return random_directions / random_directions.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)


def _forward_last_logits(
    encoder: nn.Module,
    readout: nn.Module,
    probe_bank: PositionProbeBank,
    cases: Sequence[InterventionCase],
    *,
    board_size: int,
    device: torch.device,
    alpha: float,
    control: Literal["null", "probe", "random"],
    random_seed: int,
) -> torch.Tensor:
    tokens, last_indices = _token_batch(
        cases,
        board_size=board_size,
        device=device,
    )
    positions = torch.arange(tokens.shape[1], device=device)
    rows = torch.arange(tokens.shape[0], device=device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        hidden = encoder.drop(encoder.wte(tokens) + encoder.wpe(positions))
        for layer, block in enumerate(encoder.blocks, start=1):
            hidden = block(hidden)
            if control == "null":
                continue
            directions = _probe_directions(
                probe_bank,
                cases,
                layer=layer,
                device=device,
            )
            if control == "random":
                directions = _random_directions_like(
                    directions,
                    seed=random_seed + layer,
                )
            selected = hidden[rows, last_indices].float()
            residual_scale = selected.std(dim=-1, unbiased=False).clamp_min(1e-6)
            edited = selected + alpha * residual_scale[:, None] * directions
            hidden[rows, last_indices] = edited.to(hidden.dtype)
        final_hidden = encoder.ln_f(hidden)[rows, last_indices].float()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=False,
    ):
        logits = readout(final_hidden.float())
    if not torch.isfinite(logits).all():
        raise RuntimeError("Intervention produced non-finite logits")
    return logits.float()


def _score_logits(
    logits: torch.Tensor,
    cases: Sequence[InterventionCase],
) -> dict[str, float | int | dict[str, float]]:
    probs = torch.softmax(logits, dim=-1).cpu()
    top1_target = 0
    target_mass = 0.0
    original_mass = 0.0
    newly_legal_mass = 0.0
    removed_legal_mass = 0.0
    topn_errors = 0.0
    by_operation: dict[str, list[float]] = {"flip": [], "erase": []}
    for row, case in enumerate(cases):
        target = set(case.target_legal_tokens)
        original = set(case.original_legal_tokens)
        target_indices = torch.tensor(sorted(target), dtype=torch.long)
        original_indices = torch.tensor(sorted(original), dtype=torch.long)
        target_value = float(probs[row].index_select(0, target_indices).sum())
        original_value = float(probs[row].index_select(0, original_indices).sum())
        target_mass += target_value
        original_mass += original_value
        by_operation[case.operation].append(target_value)
        prediction = int(probs[row].argmax())
        top1_target += int(prediction in target)
        n = len(target)
        predicted_set = set(torch.topk(probs[row], k=n).indices.tolist())
        topn_errors += len(predicted_set - target) + len(target - predicted_set)
        newly = target - original
        removed = original - target
        if newly:
            newly_legal_mass += float(
                probs[row].index_select(
                    0, torch.tensor(sorted(newly), dtype=torch.long)
                ).sum()
            )
        if removed:
            removed_legal_mass += float(
                probs[row].index_select(
                    0, torch.tensor(sorted(removed), dtype=torch.long)
                ).sum()
            )
    count = len(cases)
    return {
        "n_cases": count,
        "target_top1_legal": top1_target / count,
        "target_legal_probability_mass": target_mass / count,
        "original_legal_probability_mass": original_mass / count,
        "newly_legal_probability_mass": newly_legal_mass / count,
        "removed_legal_probability_mass": removed_legal_mass / count,
        "mean_topn_false_positive_plus_false_negative": topn_errors / count,
        "target_mass_by_operation": {
            operation: sum(values) / len(values) if values else None
            for operation, values in by_operation.items()
        },
    }


def evaluate_causal_intervention(
    encoder: nn.Module,
    readout: nn.Module,
    probe_bank: PositionProbeBank,
    *,
    selection_cases: Sequence[InterventionCase],
    test_cases: Sequence[InterventionCase],
    board_size: int,
    device: torch.device | str,
    alpha_values: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Select intervention strength on selection cases and test it once."""
    device = torch.device(device)
    encoder.eval()
    readout.eval()
    probe_bank.eval()

    def run(
        cases: Sequence[InterventionCase],
        *,
        alpha: float,
        control: Literal["null", "probe", "random"],
    ) -> dict[str, Any]:
        logits: list[torch.Tensor] = []
        for start in range(0, len(cases), batch_size):
            logits.append(
                _forward_last_logits(
                    encoder,
                    readout,
                    probe_bank,
                    cases[start : start + batch_size],
                    board_size=board_size,
                    device=device,
                    alpha=alpha,
                    control=control,
                    random_seed=seed + start * 1000,
                ).cpu()
            )
        return _score_logits(torch.cat(logits), cases)

    selection_curve: list[dict[str, Any]] = []
    for alpha in alpha_values:
        metrics = run(selection_cases, alpha=float(alpha), control="probe")
        selection_curve.append({"alpha": float(alpha), **metrics})
        print(
            f"intervention selection alpha={float(alpha):g} "
            f"target_mass={metrics['target_legal_probability_mass']:.4f} "
            f"topN_error={metrics['mean_topn_false_positive_plus_false_negative']:.3f}",
            flush=True,
        )
    selected = min(
        selection_curve,
        key=lambda row: (
            float(row["mean_topn_false_positive_plus_false_negative"]),
            -float(row["target_legal_probability_mass"]),
            float(row["alpha"]),
        ),
    )
    alpha = float(selected["alpha"])
    null = run(test_cases, alpha=0.0, control="null")
    random_control = run(test_cases, alpha=alpha, control="random")
    probe = run(test_cases, alpha=alpha, control="probe")
    return {
        "protocol": "nanda_linear_residual_intervention_v1",
        "board_size": board_size,
        "alpha_values": [float(value) for value in alpha_values],
        "selected_alpha": alpha,
        "selection_curve": selection_curve,
        "selection_cases": [case.to_dict() for case in selection_cases],
        "test_cases": [case.to_dict() for case in test_cases],
        "test": {
            "null": null,
            "magnitude_matched_random": random_control,
            "probe_direction": probe,
        },
    }


__all__ = [
    "InterventionCase",
    "build_intervention_cases",
    "evaluate_causal_intervention",
]
