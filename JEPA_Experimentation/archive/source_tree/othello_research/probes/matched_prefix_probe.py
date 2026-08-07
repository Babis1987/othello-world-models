"""Matched-conditions board probes: pass-free prefixes, last position only.

Diagnostic companion to the standard all-position probes in
``probes/board_state.py``. The family-InfoNCE objective (and the hard-disjoint
JEPA family generally) supervises ONLY the hidden state at position t-1 of
pass-free prefixes with t in a bounded range. The standard probe protocol
averages over every position of every game — including positions the
objective never shaped — so it cannot distinguish

    (a) "no probe-decodable board state anywhere" from
    (b) "board state concentrated at the trained readout positions".

This module trains the same per-layer absolute/relative probes on a bounded,
pass-free, last-prefix-position slice. That slice exactly matches the original
family-InfoNCE protocol. For v1 random-boundary VICReg it is a controlled
diagnostic rather than the complete training distribution, because v1 samples
valid boundaries across the full sequence and does not require pass-free
prefixes. Label conventions are identical to ``board_state.py``
(ABS_BLACK/ABS_WHITE/EMPTY; REL mine/yours via the effective side to play),
verified by parity tests on both 8x8 and larger boards.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.datasets.move_mapping import build_mappings
from othello_research.othello.board import OthelloBoardState
from othello_research.othello.bitboard import (
    INITIAL_BLACK,
    INITIAL_WHITE,
    apply_move,
    legal_moves_mask,
)
from othello_research.probes.board_state import (
    ABS_BLACK,
    ABS_WHITE,
    EMPTY,
    REL_MINE,
    REL_YOURS,
    ActivationCache,
)

U0 = np.uint64(0)


@dataclass
class MatchedPrefixSample:
    tokens: np.ndarray  # (t,) int64 token ids
    t: int
    y_abs: np.ndarray  # (n_squares,) int64
    y_rel: np.ndarray  # (n_squares,) int64


def _replay_passfree_generic(
    game: Sequence[int], board_size: int
) -> list[tuple[int, int]]:
    """Board-size-aware fallback for boards that do not fit in a uint64."""
    board = OthelloBoardState(n=board_size)
    states: list[tuple[int, int]] = []
    for i, move_value in enumerate(game):
        move = int(move_value)
        expected_color = 1 if (i % 2) == 0 else -1
        if board.next_hand_color != expected_color:
            break
        # This diagnostic intentionally stops at a pass rather than letting
        # OthelloBoardState.umpire perform its implicit-forfeit handling.
        if not board.get_valid_moves() or not board.tentative_move(move):
            break
        board.umpire(move)

        black = 0
        white = 0
        for square, value in enumerate(board.state.reshape(-1)):
            if value == 1:
                black |= 1 << square
            elif value == -1:
                white |= 1 << square
        states.append((black, white))
    return states


def _replay_passfree(
    game: Sequence[int], board_size: int = 8
) -> list[tuple[int, int]]:
    """Replay under strict alternation; return (black, white) masks per ply.

    Stops at the first pass or illegal-for-expected-player move, mirroring the
    family-index scan rule. Element i is the position AFTER move i.
    """
    if board_size != 8:
        return _replay_passfree_generic(game, board_size)

    black, white = INITIAL_BLACK, INITIAL_WHITE
    states: list[tuple[int, int]] = []
    for i, move in enumerate(game):
        black_to_move = (i % 2) == 0
        player, opponent = (black, white) if black_to_move else (white, black)
        if legal_moves_mask(np.uint64(player), np.uint64(opponent)) == U0:
            break
        player, opponent, flips = apply_move(
            np.uint64(player), np.uint64(opponent), np.uint64(move)
        )
        if flips == U0:
            break
        if black_to_move:
            black, white = int(player), int(opponent)
        else:
            white, black = int(player), int(opponent)
        states.append((black, white))
    return states


def _labels_from_masks(
    black: int, white: int, t: int, board_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute + relative labels for the position after t pass-free moves.

    Matches ``board_state.board_state_labels_after_moves``: relative "mine" is
    the EFFECTIVE side to play (if the by-parity player has no legal move but
    the opponent does, the opponent is to play).
    """
    n_squares = board_size * board_size
    y_abs = np.full(n_squares, EMPTY, dtype=np.int64)
    y_rel = np.full(n_squares, EMPTY, dtype=np.int64)

    expected_black = (t % 2) == 0  # Black plays ply t (0-indexed) next.
    side_is_black = expected_black
    if board_size == 8:
        black_u, white_u = np.uint64(black), np.uint64(white)
        player, opponent = (
            (black_u, white_u) if expected_black else (white_u, black_u)
        )
        if legal_moves_mask(player, opponent) == U0 and (
            legal_moves_mask(opponent, player) != U0
        ):
            side_is_black = not expected_black
    else:
        # uint64 bitboards are intrinsically limited to 8x8. Reconstruct the
        # dynamic board to determine whether the parity player must pass.
        board = OthelloBoardState(n=board_size)
        for square in range(n_squares):
            bit = 1 << square
            row, col = divmod(square, board_size)
            if black & bit:
                board.state[row, col] = 1
            elif white & bit:
                board.state[row, col] = -1
            else:
                board.state[row, col] = 0
        board.next_hand_color = 1 if expected_black else -1
        expected_has_move = bool(board.get_valid_moves())
        board.next_hand_color *= -1
        opponent_has_move = bool(board.get_valid_moves())
        if not expected_has_move and opponent_has_move:
            side_is_black = not expected_black
    mine_mask = black if side_is_black else white
    yours_mask = white if side_is_black else black

    for square in range(n_squares):
        bit = 1 << square
        if black & bit:
            y_abs[square] = ABS_BLACK
        elif white & bit:
            y_abs[square] = ABS_WHITE
        if mine_mask & bit:
            y_rel[square] = REL_MINE
        elif yours_mask & bit:
            y_rel[square] = REL_YOURS
    return y_abs, y_rel


def sample_matched_prefixes(
    games: Sequence[Sequence[int]],
    *,
    board_size: int = 8,
    t_min: int = 4,
    t_max: int = 44,
    prefixes_per_game: int = 4,
    seed: int = 42,
) -> list[MatchedPrefixSample]:
    """Sample pass-free prefixes with t in [t_min, t_max], labels at position t."""
    raw_to_token, _ = build_mappings(board_size)
    lut = np.full(board_size * board_size, -1, dtype=np.int64)
    for raw, token in enumerate(raw_to_token):
        lut[raw] = token
    rng = np.random.default_rng(seed)
    samples: list[MatchedPrefixSample] = []
    for game in games:
        states = _replay_passfree(game, board_size)
        last_t = min(len(states), t_max)
        if last_t < t_min:
            continue
        n = min(prefixes_per_game, last_t - t_min + 1)
        ts = rng.choice(np.arange(t_min, last_t + 1), size=n, replace=False)
        for t in sorted(int(x) for x in ts):
            tokens = lut[np.asarray(game[:t], dtype=np.int64)]
            if (tokens < 0).any():
                continue
            black, white = states[t - 1]
            y_abs, y_rel = _labels_from_masks(black, white, t, board_size)
            samples.append(MatchedPrefixSample(tokens, t, y_abs, y_rel))
    return samples


@torch.no_grad()
def extract_last_position_features(
    context_encoder: nn.Module,
    samples: Sequence[MatchedPrefixSample],
    *,
    layers: Sequence[int],
    device: torch.device,
    batch_size: int = 256,
    precision: str = "bf16",
    pad_token: int | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-layer features at position t-1 plus label/t tensors (CPU)."""
    config = context_encoder.config
    if pad_token is None:
        pad_token = int(config.vocab_size) - 1
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        amp = lambda: torch.autocast("cuda", dtype=dtype, enabled=True)  # noqa: E731
    else:
        amp = nullcontext
    context_encoder.eval()

    features: dict[int, list[torch.Tensor]] = {int(l): [] for l in layers}
    y_abs_all: list[torch.Tensor] = []
    y_rel_all: list[torch.Tensor] = []
    t_all: list[torch.Tensor] = []
    with ActivationCache(context_encoder, layers) as cache:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            max_t = max(s.t for s in batch)
            x = torch.full((len(batch), max_t), pad_token, dtype=torch.long)
            for i, s in enumerate(batch):
                x[i, : s.t] = torch.from_numpy(s.tokens)
            lengths = torch.tensor([s.t for s in batch], dtype=torch.long)
            cache.clear()
            with amp():
                context_encoder(x.to(device))
            gather = (lengths - 1).to(device)
            rows = torch.arange(len(batch), device=device)
            for layer in features:
                acts = cache.activations[int(layer)].to(device)
                features[layer].append(acts[rows, gather].float().cpu())
            y_abs_all.append(torch.from_numpy(np.stack([s.y_abs for s in batch])))
            y_rel_all.append(torch.from_numpy(np.stack([s.y_rel for s in batch])))
            t_all.append(lengths)
    return (
        {layer: torch.cat(parts) for layer, parts in features.items()},
        torch.cat(y_abs_all),
        torch.cat(y_rel_all),
        torch.cat(t_all),
    )


class _MLPProbe(nn.Module):
    """Same shape as the smoke-eval notebook's MLPProbe for comparability."""

    def __init__(self, d_in: int, d_out: int, hidden_dim: int, n_layers: int,
                 dropout: float):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(d_in, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        ]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _InputNormalizedProbe(nn.Module):
    """Apply a parameter-free LayerNorm before a linear or MLP readout.

    Mamba residual-stream scales vary sharply with depth.  A shared,
    parameter-free normalization keeps the downstream optimization problem
    comparable across layers and encoder architectures without allowing the
    probe to learn an architecture-specific affine rescaling.
    """

    def __init__(self, d_model: int, readout: nn.Module) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.readout = readout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.input_norm(x.float()))


def _make_probe(probe_type: str, d_model: int, n_squares: int,
                hidden_dim: int, n_probe_layers: int, dropout: float,
                input_layernorm: bool = False) -> nn.Module:
    if probe_type == "linear":
        readout: nn.Module = nn.Linear(d_model, n_squares * 3)
    elif probe_type == "mlp":
        readout = _MLPProbe(d_model, n_squares * 3, hidden_dim, n_probe_layers,
                            dropout)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type!r}")
    if input_layernorm:
        return _InputNormalizedProbe(d_model, readout)
    return readout


def train_matched_probe_bank(
    context_encoder: nn.Module,
    train_games: Sequence[Sequence[int]],
    val_games: Sequence[Sequence[int]],
    *,
    layers: Sequence[int],
    device: torch.device,
    probe_type: str = "mlp",
    hidden_dim: int = 512,
    n_probe_layers: int = 1,
    dropout: float = 0.1,
    epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    precision: str = "bf16",
    board_size: int = 8,
    t_min: int = 4,
    t_max: int = 44,
    prefixes_per_game: int = 4,
    seed: int = 42,
    input_layernorm: bool = False,
    gradient_clip: float | None = None,
) -> dict:
    """Train per-layer abs/rel probes on matched-conditions features.

    The encoder is frozen, so features are extracted once and the probes train
    on cached tensors (fast). Returns a result dict shaped like the standard
    probe results: ``{'layers': [...], 'metrics': {layer: {mode: {accuracy}}}}``.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_samples = sample_matched_prefixes(
        train_games, board_size=board_size, t_min=t_min, t_max=t_max,
        prefixes_per_game=prefixes_per_game, seed=seed,
    )
    val_samples = sample_matched_prefixes(
        val_games, board_size=board_size, t_min=t_min, t_max=t_max,
        prefixes_per_game=prefixes_per_game, seed=seed + 1,
    )
    if not train_samples or not val_samples:
        raise ValueError("No matched prefixes sampled; check t range / games.")

    train_feats, train_abs, train_rel, _ = extract_last_position_features(
        context_encoder, train_samples, layers=layers, device=device,
        batch_size=batch_size, precision=precision,
    )
    val_feats, val_abs, val_rel, val_t = extract_last_position_features(
        context_encoder, val_samples, layers=layers, device=device,
        batch_size=batch_size, precision=precision,
    )

    d_model = int(context_encoder.config.d_model)
    n_squares = board_size * board_size
    generator = torch.Generator().manual_seed(seed)
    metrics: dict[str, dict] = {}
    for layer in layers:
        layer_metrics: dict[str, dict] = {}
        for mode, y_train, y_val in (
            ("absolute", train_abs, val_abs),
            ("relative", train_rel, val_rel),
        ):
            probe = _make_probe(
                probe_type,
                d_model,
                n_squares,
                hidden_dim,
                n_probe_layers,
                dropout,
                input_layernorm=input_layernorm,
            ).to(device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=lr,
                                          weight_decay=weight_decay)
            x_train = train_feats[int(layer)]
            for _ in range(epochs):
                order = torch.randperm(len(x_train), generator=generator)
                for start in range(0, len(order), batch_size):
                    idx = order[start : start + batch_size]
                    xb = x_train[idx].to(device)
                    yb = y_train[idx].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = probe(xb).view(len(idx), n_squares, 3)
                    loss = F.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1))
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite {probe_type} matched-probe loss at "
                            f"layer={layer} mode={mode}"
                        )
                    loss.backward()
                    if gradient_clip is not None and gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            probe.parameters(), gradient_clip
                        )
                    optimizer.step()
            probe.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                x_val = val_feats[int(layer)]
                for start in range(0, len(x_val), batch_size):
                    xb = x_val[start : start + batch_size].to(device)
                    yb = y_val[start : start + batch_size].to(device)
                    preds = probe(xb).view(len(xb), n_squares, 3).argmax(-1)
                    correct += int((preds == yb).sum())
                    total += int(yb.numel())
            layer_metrics[mode] = {
                "accuracy": correct / max(1, total),
                "n_labels": total,
            }
            del probe
        metrics[str(int(layer))] = layer_metrics

    return {
        "probe_type": probe_type,
        "layers": [int(l) for l in layers],
        "metrics": metrics,
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "t_min": t_min,
        "t_max": t_max,
        "prefixes_per_game": prefixes_per_game,
        "val_t_mean": float(val_t.float().mean()),
        "seed": seed,
        "precision": precision,
        "input_layernorm": input_layernorm,
        "gradient_clip": gradient_clip,
        "protocol": "pass-free prefixes, hidden state at position t-1 only",
    }


def print_matched_table(label: str, result: dict) -> None:
    print(label)
    print(f"  samples: train={result['n_train_samples']:,} "
          f"val={result['n_val_samples']:,} "
          f"(t in [{result['t_min']},{result['t_max']}])")
    print("layer | absolute | relative")
    print("----- | -------- | --------")
    for layer in result["layers"]:
        m = result["metrics"][str(layer)]
        print(f"{layer:>5} | {m['absolute']['accuracy']:>8.2%} | "
              f"{m['relative']['accuracy']:>8.2%}")
