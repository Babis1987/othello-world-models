"""Board-state probing utilities for OthelloGPT.

The legal-move evaluator checks whether the model obeys the Othello rules at
the output layer. Board-state probes ask a different question: whether the
model's residual stream linearly encodes the board it is implicitly reasoning
over. This module trains lightweight linear probes on frozen model activations
for two label spaces:

    - absolute: black, white, empty
    - relative: mine, yours, empty, where "mine" is the side to play next

Layer indexing is explicit and notebook-friendly: layer 0 is the token+position
embedding stream after dropout, and layers 1..N are the residual stream after
each transformer block.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_thesis.data.chunk_dataset import load_chunk
from othello_thesis.data.move_vocabulary import build_mappings
from othello_thesis.game_engine.board import OthelloBoardState

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is available on Colab by default
    def tqdm(x=None, **_):
        return x if x is not None else _NoOpBar()


class _NoOpBar:
    """Minimal stand-in for tqdm when it is not installed."""

    def update(self, _n: int = 1) -> None: ...
    def set_postfix(self, **_: object) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *_): ...


ABS_BLACK = 0
ABS_WHITE = 1
EMPTY = 2

REL_MINE = 0
REL_YOURS = 1

LABEL_PAD = -100


# === Label construction ===

def _effective_side_to_play(board: OthelloBoardState) -> int:
    """Return the color that will make the next real move.

    The dataset has no explicit pass token. If ``board.next_hand_color`` has no
    legal moves, the next raw move belongs to the opponent, matching the rule
    engine's implicit forfeit handling in ``umpire``.
    """
    side = board.next_hand_color
    if board.get_valid_moves():
        return side

    board.next_hand_color *= -1
    opponent_has_move = bool(board.get_valid_moves())
    board.next_hand_color *= -1
    return -side if opponent_has_move else side


def board_state_labels_after_moves(
    game_raw: Sequence[int],
    board_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return absolute and relative board labels after each raw move.

    Args:
        game_raw: Raw moves to replay. The output has one label row per move.
        board_size: Side length of the Othello board.

    Returns:
        ``(absolute_labels, relative_labels)``. Each outer list has length
        ``len(game_raw)`` and each inner list has length ``board_size**2``.
    """
    board = OthelloBoardState(n=board_size)
    absolute_labels: list[list[int]] = []
    relative_labels: list[list[int]] = []

    for move in game_raw:
        board.umpire(move)
        side_to_play = _effective_side_to_play(board)

        abs_row: list[int] = []
        rel_row: list[int] = []
        for value in board.get_state():
            if value == 1:
                abs_row.append(ABS_BLACK)
            elif value == -1:
                abs_row.append(ABS_WHITE)
            else:
                abs_row.append(EMPTY)

            if value == 0:
                rel_row.append(EMPTY)
            elif value == side_to_play:
                rel_row.append(REL_MINE)
            else:
                rel_row.append(REL_YOURS)

        absolute_labels.append(abs_row)
        relative_labels.append(rel_row)

    return absolute_labels, relative_labels


# === Probe model ===

class BoardStateProbe(nn.Module):
    """One 3-way classifier per board square.

    With ``hidden_dim=None`` this is a linear probe. Passing ``hidden_dim``
    creates a small 2-layer MLP probe, matching the nonlinear probe family used
    for absolute board-color labels in the original OthelloGPT work.
    """

    def __init__(self, d_model: int, n_squares: int, hidden_dim: int | None = None):
        super().__init__()
        self.n_squares = n_squares
        if hidden_dim is None:
            self.proj = nn.Linear(d_model, n_squares * 3)
        else:
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_squares * 3),
            )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activations to ``(..., n_squares, 3)`` logits."""
        logits = self.proj(activations)
        return logits.view(*logits.shape[:-1], self.n_squares, 3)


class ProbeBank(nn.Module):
    """Absolute and relative board-state probes for selected layers."""

    def __init__(
        self,
        layers: Sequence[int],
        d_model: int,
        n_squares: int,
        *,
        absolute_hidden_dim: int | None = None,
        relative_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.absolute = nn.ModuleDict({
            str(layer): BoardStateProbe(d_model, n_squares, absolute_hidden_dim)
            for layer in layers
        })
        self.relative = nn.ModuleDict({
            str(layer): BoardStateProbe(d_model, n_squares, relative_hidden_dim)
            for layer in layers
        })


# === Activation capture ===

def _base_model(model: nn.Module) -> nn.Module:
    """Unwrap a torch.compile wrapper if present."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _model_config_value(model: nn.Module, name: str, default: int) -> int:
    config = getattr(_base_model(model), "config", None)
    return int(getattr(config, name, default))


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


class ActivationCache:
    """Forward-hook based residual stream collector."""

    def __init__(self, model: nn.Module, layers: Sequence[int]):
        self.model = _base_model(model)
        self.layers = tuple(int(layer) for layer in layers)
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActivationCache":
        for layer in self.layers:
            module = self._module_for_layer(layer)
            handle = module.register_forward_hook(self._make_hook(layer))
            self.handles.append(handle)
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.activations.clear()

    def _module_for_layer(self, layer: int) -> nn.Module:
        if layer == 0:
            return self.model.drop
        if 1 <= layer <= len(self.model.blocks):
            return self.model.blocks[layer - 1]
        raise ValueError(
            f"Layer {layer} is invalid; expected 0..{len(self.model.blocks)}"
        )

    def _make_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations[layer] = output.detach().float()

        return hook


# === Data loading and batches ===

def load_games_from_chunks(
    chunks: Sequence[str | Path],
    n_games: int,
    *,
    skip_games: int = 0,
) -> list[list[int]]:
    """Load a deterministic slice of games from chunk files."""
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games}")
    if skip_games < 0:
        raise ValueError(f"skip_games must be non-negative, got {skip_games}")

    games: list[list[int]] = []
    skipped = 0
    for chunk_path in chunks:
        for game in load_chunk(chunk_path):
            if skipped < skip_games:
                skipped += 1
                continue
            games.append(list(game))
            if len(games) >= n_games:
                return games
    return games


def _tokenize_game(
    game_raw: Sequence[int],
    raw_to_token: Sequence[int],
    board_size: int,
) -> list[int]:
    tokens: list[int] = []
    for move in game_raw:
        if not (0 <= move < board_size * board_size):
            raise ValueError(
                f"Raw move {move} is outside [0, {board_size * board_size})"
            )
        token = raw_to_token[move]
        if token == -1:
            raise ValueError(
                f"Raw move {move} is a starting position and has no token id"
            )
        tokens.append(token)
    return tokens


def _iter_probe_batches(
    games_raw: Sequence[Sequence[int]],
    board_size: int,
    block_size: int,
    batch_size: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield padded input tokens and absolute/relative labels."""
    raw_to_token, _ = build_mappings(board_size)
    pad_token = board_size * board_size - 4
    n_squares = board_size * board_size

    for start in range(0, len(games_raw), batch_size):
        batch_raw = [list(game[: block_size + 1]) for game in games_raw[start : start + batch_size]]
        batch_raw = [game for game in batch_raw if len(game) >= 2]
        if not batch_raw:
            continue

        input_lengths = [len(game) - 1 for game in batch_raw]
        max_len = max(input_lengths)
        x = torch.full((len(batch_raw), max_len), pad_token, dtype=torch.long)
        y_abs = torch.full(
            (len(batch_raw), max_len, n_squares), LABEL_PAD, dtype=torch.long
        )
        y_rel = torch.full_like(y_abs, LABEL_PAD)

        for row, raw in enumerate(batch_raw):
            input_raw = raw[:-1]
            tokens = _tokenize_game(input_raw, raw_to_token, board_size)
            labels_abs, labels_rel = board_state_labels_after_moves(
                input_raw, board_size
            )
            input_len = len(tokens)
            x[row, :input_len] = torch.tensor(tokens, dtype=torch.long)
            y_abs[row, :input_len] = torch.tensor(labels_abs, dtype=torch.long)
            y_rel[row, :input_len] = torch.tensor(labels_rel, dtype=torch.long)

        yield x, y_abs, y_rel


# === Training and evaluation ===

def _probe_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, 3),
        labels.reshape(-1),
        ignore_index=LABEL_PAD,
    )


def _metric_counts(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    preds = logits.argmax(dim=-1)
    valid = labels != LABEL_PAD
    occupied = valid & (labels != EMPTY)
    empty = valid & (labels == EMPTY)
    return {
        "total": int(valid.sum().item()),
        "correct": int(((preds == labels) & valid).sum().item()),
        "occupied_total": int(occupied.sum().item()),
        "occupied_correct": int(((preds == labels) & occupied).sum().item()),
        "empty_total": int(empty.sum().item()),
        "empty_correct": int(((preds == labels) & empty).sum().item()),
    }


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _counts_to_metrics(counts: dict[str, int]) -> dict[str, float]:
    def ratio(num: str, den: str) -> float:
        total = counts.get(den, 0)
        if total == 0:
            return float("nan")
        return float(counts.get(num, 0) / total)

    return {
        "accuracy": ratio("correct", "total"),
        "occupied_accuracy": ratio("occupied_correct", "occupied_total"),
        "empty_accuracy": ratio("empty_correct", "empty_total"),
        "n_labels": int(counts.get("total", 0)),
        "n_occupied_labels": int(counts.get("occupied_total", 0)),
        "n_empty_labels": int(counts.get("empty_total", 0)),
    }


def _empty_counts_by_layer(layers: Sequence[int]) -> dict[str, dict[str, dict[str, int]]]:
    return {
        str(layer): {
            "absolute": {},
            "relative": {},
        }
        for layer in layers
    }


def _evaluate_probe_bank(
    model: nn.Module,
    probe_bank: ProbeBank,
    games_raw: Sequence[Sequence[int]],
    board_size: int,
    layers: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, dict[str, float]]]:
    block_size = _model_config_value(model, "block_size", board_size * board_size - 5)
    counts = _empty_counts_by_layer(layers)

    model.eval()
    probe_bank.eval()
    with ActivationCache(model, layers) as cache:
        for x_cpu, y_abs_cpu, y_rel_cpu in tqdm(
            _iter_probe_batches(games_raw, board_size, block_size, batch_size),
            desc="probe eval",
            leave=False,
        ):
            x = x_cpu.to(device)
            y_abs = y_abs_cpu.to(device)
            y_rel = y_rel_cpu.to(device)
            cache.clear()

            with torch.inference_mode():
                with _amp_context(device):
                    model(x)
                for layer in layers:
                    key = str(layer)
                    acts = cache.activations[layer].to(device)
                    abs_logits = probe_bank.absolute[key](acts)
                    rel_logits = probe_bank.relative[key](acts)
                    _merge_counts(
                        counts[key]["absolute"], _metric_counts(abs_logits, y_abs)
                    )
                    _merge_counts(
                        counts[key]["relative"], _metric_counts(rel_logits, y_rel)
                    )

    return {
        layer: {
            mode: _counts_to_metrics(mode_counts)
            for mode, mode_counts in layer_counts.items()
        }
        for layer, layer_counts in counts.items()
    }


def train_board_state_probes(
    model: nn.Module,
    train_chunks: Sequence[str | Path],
    val_chunks: Sequence[str | Path],
    board_size: int,
    n_train_games: int,
    n_val_games: int,
    device: torch.device | str,
    *,
    layers: Sequence[int] | None = None,
    batch_size: int = 64,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    absolute_hidden_dim: int | None = None,
    relative_hidden_dim: int | None = None,
) -> tuple[ProbeBank, dict[str, object]]:
    """Train absolute and relative linear board-state probes.

    Args:
        model: Frozen OthelloGPT model.
        train_chunks: Chunk files used to fit the probes.
        val_chunks: Held-out chunk files used to report probe accuracy.
        board_size: Side length of the Othello board.
        n_train_games: Number of deterministic training games to load.
        n_val_games: Number of deterministic validation games to load.
        device: Torch device for model/probe computation.
        layers: Layer ids to probe. Defaults to all layers, including layer 0.
        batch_size: Game batch size for activation extraction.
        epochs: Number of probe-training passes over the loaded games.
        lr: Probe optimizer learning rate.
        weight_decay: Probe optimizer weight decay.
        absolute_hidden_dim: Hidden width for absolute MLP probes. ``None``
            means a linear probe.
        relative_hidden_dim: Hidden width for relative MLP probes. ``None``
            means a linear probe.

    Returns:
        ``(probe_bank, result_dict)`` where ``result_dict`` contains validation
        metrics per layer and mode.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    device = torch.device(device)
    base_model = _base_model(model)
    n_layers = len(base_model.blocks)
    if layers is None:
        layers = tuple(range(n_layers + 1))
    else:
        layers = tuple(int(layer) for layer in layers)
    for layer in layers:
        if not (0 <= layer <= n_layers):
            raise ValueError(f"Invalid probe layer {layer}; expected 0..{n_layers}")

    d_model = _model_config_value(model, "d_model", 512)
    block_size = _model_config_value(model, "block_size", board_size * board_size - 5)
    n_squares = board_size * board_size

    train_games = load_games_from_chunks(train_chunks, n_train_games)
    val_games = load_games_from_chunks(val_chunks, n_val_games)
    if not train_games:
        raise ValueError("No training games were loaded for probing")
    if not val_games:
        raise ValueError("No validation games were loaded for probing")

    probe_bank = ProbeBank(
        layers,
        d_model,
        n_squares,
        absolute_hidden_dim=absolute_hidden_dim,
        relative_hidden_dim=relative_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe_bank.parameters(), lr=lr, weight_decay=weight_decay
    )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    history: list[dict[str, float]] = []
    with ActivationCache(model, layers) as cache:
        for epoch in range(1, epochs + 1):
            probe_bank.train()
            total_loss = 0.0
            total_batches = 0

            pbar = tqdm(
                _iter_probe_batches(train_games, board_size, block_size, batch_size),
                desc=f"probe train epoch {epoch}/{epochs}",
                leave=False,
            )
            for x_cpu, y_abs_cpu, y_rel_cpu in pbar:
                x = x_cpu.to(device)
                y_abs = y_abs_cpu.to(device)
                y_rel = y_rel_cpu.to(device)
                cache.clear()

                with torch.no_grad():
                    with _amp_context(device):
                        model(x)

                optimizer.zero_grad(set_to_none=True)
                loss = torch.zeros((), device=device)
                for layer in layers:
                    key = str(layer)
                    acts = cache.activations[layer].to(device)
                    loss = loss + _probe_loss(probe_bank.absolute[key](acts), y_abs)
                    loss = loss + _probe_loss(probe_bank.relative[key](acts), y_rel)
                loss = loss / (2 * len(layers))
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_batches += 1
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(loss=f"{total_loss / total_batches:.4f}")

            history.append({
                "epoch": float(epoch),
                "train_loss": total_loss / max(1, total_batches),
            })

    metrics = _evaluate_probe_bank(
        model=model,
        probe_bank=probe_bank,
        games_raw=val_games,
        board_size=board_size,
        layers=layers,
        batch_size=batch_size,
        device=device,
    )
    result: dict[str, object] = {
        "n_train_games": len(train_games),
        "n_val_games": len(val_games),
        "layers": list(layers),
        "board_size": board_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "absolute_hidden_dim": absolute_hidden_dim,
        "relative_hidden_dim": relative_hidden_dim,
        "history": history,
        "metrics": metrics,
    }
    return probe_bank, result
