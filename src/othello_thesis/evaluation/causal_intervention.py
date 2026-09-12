"""Adapted causal edits for Othello residual-stream board directions.

The intervention probe in this module is deliberately a *raw* linear relative
board probe.  The normalized probes used for the headline decodability tables
contain a per-example LayerNorm, whose classifier weights are not fixed
directions in the encoder's raw residual space.

Why intervention at all
-----------------------
A probe shows that board state is *decodable* from the residual stream. That is
a correlational claim: the information is present, but the model might not be
using it. A high-accuracy probe is compatible with the board being an inert
by-product.

The intervention turns it into a causal claim. Take a real position, pick an
occupied square, and construct a counterfactual board in which that square is
flipped or erased -- chosen so that the *set of legal moves actually changes*.
Then edit the model's residual stream along the probe direction that would move
that square from its true class to the counterfactual class, and ask whether the
model's next-move distribution shifts toward the counterfactual legal set.

If it does, the board representation is being read by the computation that
produces moves. If it does not, decodability was incidental.

The counterfactual is defined on the *legal move set*, not on the board
picture, because that is the only thing the model's output can be scored
against.

Controls
--------
An edit of any sufficiently large magnitude perturbs the output. The question
is whether the *direction* matters, so the probe direction is compared against:

* ``null`` -- no edit, the unperturbed baseline.
* ``geometry_matched_random`` -- the primary control. One signed permutation per
  case, shared across all layers, applied to the probe directions. This keeps
  every per-layer norm and every within-case cross-layer angle identical, and
  changes only the alignment with the model's own feature coordinates. Anything
  the probe direction achieves over this control cannot be explained by edit
  magnitude or by the geometry of the edit sequence.
* ``independent_layer_random`` -- a fresh random direction per layer. Weaker as
  a control (it destroys the cross-layer structure too) but it separates
  "direction matters" from "coherence across layers matters".
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
    """One counterfactual edit: this square, on this position, changes the legal set.

    Carries both legal-move sets so scoring never has to replay the board, and
    both class ids so the edit direction can be reconstructed at any layer.
    The frozen dataclass plus the hashed ``case_id`` make a case suite
    quotable: the same seed and chunks regenerate exactly the same cases.
    """

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
    """Which colour actually moves next, honouring the pass rule.

    If the nominal side to move has no legal move it must pass, so the position
    the model is really being asked about belongs to the opponent. Returns None
    when neither side can move -- a finished game, which cannot be a case.
    The temporary mutation is undone in the ``finally``.
    """
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

    # Try occupied squares in a seeded random order and take the first one
    # whose edit changes the legal set. Most squares do not -- flipping a disc
    # in the middle of a settled cluster often leaves every legal move intact,
    # and such a case would score identically under every condition and
    # contribute only noise.
    occupied = [
        square
        for square, value in enumerate(board.get_state())
        if int(value) != 0
    ]
    rng.shuffle(occupied)
    for square in occupied:
        value = int(board.state.flat[square])
        # Classes are in the probe's relative (mine/yours/empty) encoding
        # relative to the side to move: 0 = mine, 1 = yours, 2 = empty.
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
    """Build deterministic flip/erase cases whose legal set actually changes.

    Every random choice is derived from ``hash(seed, chunk, game_index)``, so
    the suite depends on nothing but its inputs -- not on how many cases were
    already collected, nor on iteration order. The same suite is then used for
    every model in the grid, which is what makes the intervention numbers
    comparable across cells rather than merely similar.

    Flip and erase alternate so the suite stays balanced between the two edit
    types; if the preferred one yields no usable square for a game, the other
    is tried before giving up on it.
    """
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
    """Direction that moves square ``case.square`` from its source class to its target class.

    For a linear probe the class logit is ``w_class . h``, so ``w_target -
    w_source`` is exactly the direction along which the classifier's preference
    swaps, and nothing else about the square's decoding changes. That is the
    edit the intervention applies.

    Both guards are load-bearing rather than defensive. A probe with an input
    LayerNorm defines its weights in a *normalised* space, so its rows are not
    fixed directions of the raw residual stream and adding them would edit
    something other than the intended feature; a non-linear probe has no single
    direction at all.
    """
    probe = probe_bank.relative[str(layer)]
    if not isinstance(probe.input_norm, nn.Identity):
        raise TypeError("Causal intervention requires a raw, unnormalized probe")
    if not isinstance(probe.proj, nn.Linear):
        raise TypeError("Causal intervention requires a linear relative probe")
    d_model = int(probe.proj.weight.shape[1])
    # Flat (n_squares*3, d_model) weight, unpacked into per-square class rows.
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


def _geometry_matched_randomizer(
    reference: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one orthogonal signed permutation per case.

    Reusing the same transform at every layer preserves every layer-wise norm
    and all cross-layer angles of the targeted probe directions, while
    randomising their alignment with the model's feature coordinates.

    A signed permutation is orthogonal, so it is an isometry: lengths and angles
    survive exactly. That is what makes this the strict control -- it differs
    from the real edit in alignment and in nothing else, so any advantage the
    probe direction shows is attributable to *where* it points, not to how big
    the edit is or how the edits relate across layers.
    """
    if reference.ndim != 2:
        raise ValueError("reference directions must have shape [batch, d_model]")
    batch_size, d_model = reference.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(d_model, generator=generator) for _ in range(batch_size)]
    ).to(reference.device)
    signs = torch.randint(
        0,
        2,
        (batch_size, d_model),
        generator=generator,
        dtype=torch.int8,
    ).to(device=reference.device, dtype=reference.dtype)
    signs = signs.mul_(2).sub_(1)
    return permutations, signs


def _apply_geometry_matched_randomizer(
    directions: torch.Tensor,
    permutations: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Apply a shared-across-layers orthogonal transform to each case."""
    if directions.shape != permutations.shape or directions.shape != signs.shape:
        raise ValueError("randomizer tensors must match the directions shape")
    return directions.gather(dim=1, index=permutations) * signs


def _forward_last_logits(
    encoder: nn.Module,
    readout: nn.Module,
    probe_bank: PositionProbeBank,
    cases: Sequence[InterventionCase],
    *,
    board_size: int,
    device: torch.device,
    alpha: float,
    control: Literal["null", "probe", "random", "independent_random"],
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
        random_permutations: torch.Tensor | None = None
        random_signs: torch.Tensor | None = None
        # Sequential editing: the edit is re-applied after every block rather
        # than once, because a single edit at one layer is largely undone by
        # the layers that follow. The model has to be held at the
        # counterfactual board all the way to the output for the readout to
        # reflect it.
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
                if random_permutations is None or random_signs is None:
                    random_permutations, random_signs = _geometry_matched_randomizer(
                        directions,
                        seed=random_seed,
                    )
                directions = _apply_geometry_matched_randomizer(
                    directions,
                    random_permutations,
                    random_signs,
                )
            elif control == "independent_random":
                directions = _random_directions_like(
                    directions,
                    seed=random_seed + layer,
                )
            # Only the last real position is edited: it is the one the readout
            # reads, and the only one whose board is the position in question.
            selected = hidden[rows, last_indices].float()
            # alpha is measured in units of the local residual standard
            # deviation, so one alpha means the same relative perturbation at
            # every layer, in every model, at every board size. An absolute
            # alpha would be a different-sized intervention in each of them and
            # the comparison across the grid would be meaningless.
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
    """Score a batch of intervened logits against the counterfactual legal sets.

    The headline figure is ``target_legal_probability_mass``: how much of the
    model's next-move distribution now sits on moves that are legal on the
    *counterfactual* board. ``original_legal_probability_mass`` is its mirror
    and should fall as the first rises.

    ``newly_legal`` / ``removed_legal`` isolate the moves that the edit actually
    changed status for -- the shared majority of the two legal sets is
    uninformative, and these two numbers exclude it.

    ``mean_topn_false_positive_plus_false_negative`` takes the model's top-N
    moves, with N the size of the counterfactual legal set, and counts the
    symmetric difference against it. Unlike probability mass it cannot be
    inflated by a diffuse distribution, which is why it is used as the primary
    criterion when selecting alpha.
    """
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
    """Select intervention strength on selection cases and test it once.

    Alpha has to be tuned -- too small and the edit does nothing, too large and
    it destroys the representation and the model outputs noise -- but tuning it
    on the cases the result is reported from would be selection on the test set.
    So alpha is swept on a disjoint ``selection_cases`` suite, fixed, and then
    applied exactly once to ``test_cases`` under all four conditions.

    The selection criterion is lexicographic: minimise top-N error first, break
    ties on higher target mass, then prefer the smaller alpha. Top-N error
    leads because it is the metric that a diffuse, damaged distribution cannot
    game.
    """
    device = torch.device(device)
    encoder.eval()
    readout.eval()
    probe_bank.eval()

    def run(
        cases: Sequence[InterventionCase],
        *,
        alpha: float,
        control: Literal["null", "probe", "random", "independent_random"],
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
    # The single test pass: one unperturbed baseline, two random controls and
    # the probe direction, all at the same fixed alpha on the same cases.
    null = run(test_cases, alpha=0.0, control="null")
    random_control = run(test_cases, alpha=alpha, control="random")
    independent_random_control = run(
        test_cases,
        alpha=alpha,
        control="independent_random",
    )
    probe = run(test_cases, alpha=alpha, control="probe")
    return {
        "protocol": "adapted_linear_sequential_residual_v2",
        "board_size": board_size,
        "alpha_values": [float(value) for value in alpha_values],
        "selected_alpha": alpha,
        "selection_curve": selection_curve,
        "selection_cases": [case.to_dict() for case in selection_cases],
        "test_cases": [case.to_dict() for case in test_cases],
        "test": {
            "null": null,
            "geometry_matched_random": random_control,
            "independent_layer_random": independent_random_control,
            "probe_direction": probe,
        },
        "random_control": {
            "primary": "geometry_matched_random",
            "construction": "shared per-case signed permutation across layers",
            "preserves": [
                "per-layer direction norm",
                "all within-case cross-layer direction angles",
            ],
            "sensitivity_control": "independent_layer_random",
        },
    }


__all__ = [
    "InterventionCase",
    "build_intervention_cases",
    "evaluate_causal_intervention",
]
