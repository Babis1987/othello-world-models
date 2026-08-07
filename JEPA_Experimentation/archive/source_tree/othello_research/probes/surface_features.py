"""Surface-feature linear diagnostics for frozen Othello encoders.

These probes test whether an encoder linearly exposes simple sequence-level
statistics, such as ply count and recent-move position, alongside more
state-dependent features such as relative corner and edge occupancy.

Layer indexing matches ``othello_research.probes.board_state``: layer 0 is the
token+position stream after dropout, and layers 1..N are transformer block
outputs. The hook pattern is intentionally copied rather than refactored from
``board_state.py`` so this diagnostic can be added without modifying existing
probe code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from othello_research.datasets.dataset import load_chunk
from othello_research.datasets.move_mapping import build_mappings
from othello_research.othello.board import OthelloBoardState

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional UI nicety
    def tqdm(x=None, **_):
        return x if x is not None else _NoOpBar()


class _NoOpBar:
    def update(self, _n: int = 1) -> None: ...
    def set_postfix(self, **_: object) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *_): ...


OCC_EMPTY = 0
OCC_MINE = 1
OCC_YOURS = 2


FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "ply_index": {"type": "regression", "out_dim": 1},
    "parity": {"type": "binary", "out_dim": 2},
    "n_occupied": {"type": "regression", "out_dim": 1},
    "n_legal_moves": {"type": "regression", "out_dim": 1},
    "last_move_position": {"type": "classification", "out_dim": "n_squares"},
    "corner_occupancy": {
        "type": "multi_classification",
        "groups": 4,
        "classes_per_group": 3,
    },
    "edge_occupancy": {
        "type": "multi_classification",
        "groups": "n_edge_cells",
        "classes_per_group": 3,
    },
    "board_state_relative": {
        "type": "multi_classification",
        "groups": "n_squares",
        "classes_per_group": 3,
    },
}


@dataclass
class ProbeTrainConfig:
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"


def _validate_board_size(board_size: int) -> None:
    if board_size < 4 or board_size % 2 != 0:
        raise ValueError(f"board_size must be an even integer >= 4, got {board_size}")


def _n_squares(board_size: int) -> int:
    return board_size * board_size


def _n_playable(board_size: int) -> int:
    return _n_squares(board_size) - 4


def _block_size(board_size: int) -> int:
    return _n_playable(board_size) - 1


def _corner_indices(board_size: int) -> list[int]:
    last = board_size - 1
    return [0, last, last * board_size, board_size * board_size - 1]


def _edge_indices(board_size: int) -> list[int]:
    last = board_size - 1
    top = [col for col in range(1, last)]
    right = [row * board_size + last for row in range(1, last)]
    bottom = [last * board_size + col for col in range(1, last)]
    left = [row * board_size for row in range(1, last)]
    return top + right + bottom + left


def _effective_side_and_legal_moves(board: OthelloBoardState) -> tuple[int, list[int]]:
    """Return the side that would make the next real move and its legal moves."""
    side = board.next_hand_color
    legal_moves = board.get_valid_moves()
    if legal_moves:
        return side, legal_moves

    board.next_hand_color *= -1
    try:
        opponent_moves = board.get_valid_moves()
    finally:
        board.next_hand_color *= -1

    if opponent_moves:
        return -side, opponent_moves
    return side, []


def _is_effectively_game_over(board: OthelloBoardState) -> bool:
    _, legal_moves = _effective_side_and_legal_moves(board)
    return not legal_moves


def _replay_to_ply(
    game_raw: Sequence[int],
    ply: int,
    board_size: int,
) -> OthelloBoardState:
    _validate_board_size(board_size)
    if ply < 0 or ply >= len(game_raw):
        raise ValueError(f"ply={ply} is outside game length {len(game_raw)}")

    board = OthelloBoardState(n=board_size)
    for move in game_raw[: ply + 1]:
        board.umpire(int(move))
    return board


def _relative_occupancy(value: int, side_to_play: int) -> int:
    if value == 0:
        return OCC_EMPTY
    if value == side_to_play:
        return OCC_MINE
    return OCC_YOURS


def labels_ply_index(game_raw: Sequence[int], ply: int, board_size: int) -> torch.Tensor:
    """Return scalar ``float32`` tensor: zero-based move index at this position."""
    return compute_all_labels(game_raw, ply, board_size)["ply_index"]


def labels_parity(game_raw: Sequence[int], ply: int, board_size: int) -> torch.Tensor:
    """Return scalar ``long`` tensor: 0 if black is to move, 1 if white is to move."""
    return compute_all_labels(game_raw, ply, board_size)["parity"]


def labels_n_occupied(game_raw: Sequence[int], ply: int, board_size: int) -> torch.Tensor:
    """Return scalar ``float32`` tensor: number of non-empty board squares."""
    return compute_all_labels(game_raw, ply, board_size)["n_occupied"]


def labels_n_legal_moves(game_raw: Sequence[int], ply: int, board_size: int) -> torch.Tensor:
    """Return scalar ``float32`` tensor: legal moves for the next effective side."""
    return compute_all_labels(game_raw, ply, board_size)["n_legal_moves"]


def labels_last_move_position(
    game_raw: Sequence[int],
    ply: int,
    board_size: int,
) -> torch.Tensor:
    """Return scalar ``long`` tensor: raw board-cell index of the last move."""
    return compute_all_labels(game_raw, ply, board_size)["last_move_position"]


def labels_corner_occupancy(
    game_raw: Sequence[int],
    ply: int,
    board_size: int,
) -> torch.Tensor:
    """Return ``long`` tensor of shape ``(4,)`` with empty/mine/yours classes."""
    return compute_all_labels(game_raw, ply, board_size)["corner_occupancy"]


def labels_edge_occupancy(
    game_raw: Sequence[int],
    ply: int,
    board_size: int,
) -> torch.Tensor:
    """Return ``long`` tensor of shape ``(4 * (board_size - 2),)``.

    Edge cells exclude corners and are ordered top, right, bottom, left.
    Classes are 0=empty, 1=mine, 2=yours for the next effective side to move.
    """
    return compute_all_labels(game_raw, ply, board_size)["edge_occupancy"]


def labels_board_state_relative(
    game_raw: Sequence[int],
    ply: int,
    board_size: int,
) -> torch.Tensor:
    """Return ``long`` tensor of shape ``(board_size ** 2,)``.

    Classes are 0=empty, 1=mine, 2=yours for the next effective side to move.
    """
    return compute_all_labels(game_raw, ply, board_size)["board_state_relative"]


def compute_all_labels(
    game_raw: Sequence[int],
    ply: int,
    board_size: int,
) -> dict[str, torch.Tensor]:
    """Replay the game once and extract every diagnostic label.

    Args:
        game_raw: Raw board-cell indices, not model token ids.
        ply: Zero-based move position to label. The board is replayed through
            this move, matching the residual stream at token position ``ply``.
        board_size: Even board side length.

    Returns:
        Dict mapping feature name to label tensor.
    """
    board = _replay_to_ply(game_raw, ply, board_size)
    side_to_play, legal_moves = _effective_side_and_legal_moves(board)
    state = board.get_state()
    relative = torch.tensor(
        [_relative_occupancy(int(value), side_to_play) for value in state],
        dtype=torch.long,
    )

    return {
        "ply_index": torch.tensor(float(ply), dtype=torch.float32),
        "parity": torch.tensor(0 if side_to_play == 1 else 1, dtype=torch.long),
        "n_occupied": torch.tensor(
            float(sum(1 for value in state if value != 0)),
            dtype=torch.float32,
        ),
        "n_legal_moves": torch.tensor(float(len(legal_moves)), dtype=torch.float32),
        "last_move_position": torch.tensor(int(game_raw[ply]), dtype=torch.long),
        "corner_occupancy": relative[_corner_indices(board_size)].clone(),
        "edge_occupancy": relative[_edge_indices(board_size)].clone(),
        "board_state_relative": relative,
    }


def _base_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def extract_activations(
    encoder: nn.Module,
    token_batch: torch.Tensor,
    positions: torch.Tensor,
    layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Run encoder hooks and return sampled residual activations.

    Args:
        encoder: Inner OthelloGPT-like encoder with ``drop`` and ``blocks``.
        token_batch: Long tensor of shape ``(B, T)``.
        positions: Long tensor of shape ``(B,)`` giving the token position to
            extract for each row.
        layers: Layer indices. 0 means post-embedding-dropout; 1..N are block
            outputs.

    Returns:
        ``{layer: tensor}``, where each tensor has shape ``(B, d_model)`` and
        lives on CPU to keep GPU memory available for probe training.
    """
    model = _base_model(encoder)
    if token_batch.ndim != 2:
        raise ValueError(f"token_batch must have shape (B, T), got {token_batch.shape}")
    if positions.ndim != 1 or positions.numel() != token_batch.size(0):
        raise ValueError(
            "positions must have shape (B,), matching token_batch rows: "
            f"{positions.shape} vs B={token_batch.size(0)}"
        )
    if int(positions.min().item()) < 0 or int(positions.max().item()) >= token_batch.size(1):
        raise ValueError("positions must be within the token_batch sequence length")

    requested_layers = tuple(int(layer) for layer in layers)
    activations: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    was_training = model.training

    def module_for_layer(layer: int) -> nn.Module:
        if layer == 0:
            return model.drop
        if 1 <= layer <= len(model.blocks):
            return model.blocks[layer - 1]
        raise ValueError(f"Layer {layer} is invalid; expected 0..{len(model.blocks)}")

    def make_hook(layer: int):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            out = output[0] if isinstance(output, tuple) else output
            pos = positions.to(out.device)
            rows = torch.arange(out.size(0), device=out.device)
            activations[layer] = out[rows, pos].detach().float().cpu()

        return hook

    try:
        for layer in requested_layers:
            handles.append(module_for_layer(layer).register_forward_hook(make_hook(layer)))
        model.eval()
        with torch.no_grad():
            _ = model(token_batch)
        missing = sorted(set(requested_layers) - set(activations))
        if missing:
            raise RuntimeError(f"Missing activations for layers: {missing}")
        return {layer: activations[layer] for layer in requested_layers}
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _resolve_feature_spec(spec: dict[str, Any], board_size: int | None = None) -> dict[str, Any]:
    resolved = dict(spec)
    if resolved.get("out_dim") == "n_squares":
        if board_size is None:
            raise ValueError("board_size is required to resolve out_dim='n_squares'")
        resolved["out_dim"] = _n_squares(board_size)
    if resolved.get("groups") == "n_edge_cells":
        if board_size is None:
            raise ValueError("board_size is required to resolve groups='n_edge_cells'")
        resolved["groups"] = 4 * (board_size - 2)
    if resolved.get("groups") == "n_squares":
        if board_size is None:
            raise ValueError("board_size is required to resolve groups='n_squares'")
        resolved["groups"] = _n_squares(board_size)
    return resolved


def _classification_metrics(
    model: nn.Linear,
    activations: torch.Tensor,
    labels: torch.Tensor,
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    batch_size = 4096
    n_classes = int(spec["out_dim"])
    with torch.no_grad():
        for start in range(0, activations.size(0), batch_size):
            x = activations[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].long().view(-1).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            total_loss += float(loss.item()) * int(y.numel())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "n_labels": total,
        "chance": 1.0 / max(1, n_classes),
    }


def _multi_classification_metrics(
    model: nn.Linear,
    activations: torch.Tensor,
    labels: torch.Tensor,
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    groups = int(spec["groups"])
    classes = int(spec["classes_per_group"])
    correct = 0
    total = 0
    exact = 0
    examples = 0
    total_loss = 0.0
    batch_size = 2048
    with torch.no_grad():
        for start in range(0, activations.size(0), batch_size):
            x = activations[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].long().to(device)
            logits = model(x).view(-1, groups, classes)
            loss = F.cross_entropy(logits.reshape(-1, classes), y.reshape(-1))
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            exact += int((preds == y).all(dim=-1).sum().item())
            examples += int(y.size(0))
            total_loss += float(loss.item()) * int(y.numel())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "exact_match": exact / max(1, examples),
        "n_labels": total,
        "n_examples": examples,
        "chance": 1.0 / max(1, classes),
    }


def _regression_metrics(
    model: nn.Linear,
    activations: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float | int]:
    model.eval()
    preds: list[torch.Tensor] = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, activations.size(0), batch_size):
            x = activations[start : start + batch_size].to(device)
            pred = model(x).cpu() * target_std.cpu() + target_mean.cpu()
            preds.append(pred)

    pred_all = torch.cat(preds, dim=0).view(-1)
    y = labels.float().view(-1)
    residual = torch.sum((y - pred_all) ** 2)
    centered = torch.sum((y - y.mean()) ** 2)
    r2 = 0.0 if float(centered.item()) == 0.0 else 1.0 - float(residual.item() / centered.item())
    mae = torch.mean(torch.abs(y - pred_all))
    rmse = torch.sqrt(torch.mean((y - pred_all) ** 2))
    return {
        "r2": r2,
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "n_labels": int(y.numel()),
        "chance": 0.0,
    }


def train_linear_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    spec: dict[str, Any],
    cfg: ProbeTrainConfig,
    *,
    val_activations: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Train one linear probe and return train/validation metrics.

    Args:
        activations: Training activations with shape ``(N, d_model)``.
        labels: Training labels with shape ``(N,)`` or ``(N, n_groups)``.
        spec: Resolved feature spec. Placeholders such as ``"n_squares"`` must
            already be replaced by integers.
        cfg: Probe training hyperparameters.
        val_activations: Optional held-out activations.
        val_labels: Optional held-out labels.
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must have shape (N, d_model), got {activations.shape}")
    if activations.size(0) != labels.size(0):
        raise ValueError("activations and labels must have the same first dimension")

    device = _resolve_device(cfg.device)
    probe_type = str(spec["type"])
    d_model = int(activations.size(1))

    if probe_type == "regression":
        out_dim = int(spec.get("out_dim", 1))
        model = nn.Linear(d_model, out_dim).to(device)
        y_train = labels.float().view(-1, out_dim)
        target_mean = y_train.mean(dim=0, keepdim=True)
        target_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        train_targets = (y_train - target_mean) / target_std
    elif probe_type in {"binary", "classification"}:
        out_dim = int(spec["out_dim"])
        model = nn.Linear(d_model, out_dim).to(device)
        train_targets = labels.long().view(-1)
    elif probe_type == "multi_classification":
        groups = int(spec["groups"])
        classes = int(spec["classes_per_group"])
        model = nn.Linear(d_model, groups * classes).to(device)
        train_targets = labels.long()
    else:
        raise ValueError(f"Unsupported probe type: {probe_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dataset = TensorDataset(activations.float(), train_targets)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    for _epoch in range(cfg.epochs):
        model.train()
        for x_cpu, y_cpu in loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if probe_type == "regression":
                loss = F.mse_loss(logits, y)
            elif probe_type in {"binary", "classification"}:
                loss = F.cross_entropy(logits, y)
            else:
                groups = int(spec["groups"])
                classes = int(spec["classes_per_group"])
                loss = F.cross_entropy(logits.view(-1, groups, classes).reshape(-1, classes), y.reshape(-1))
            loss.backward()
            optimizer.step()

    if probe_type == "regression":
        train_metrics = _regression_metrics(
            model, activations.float(), labels, device, target_mean, target_std
        )
        val_metrics = (
            _regression_metrics(
                model,
                val_activations.float(),
                val_labels,
                device,
                target_mean,
                target_std,
            )
            if val_activations is not None and val_labels is not None
            else None
        )
        metric_name = "r2"
    elif probe_type in {"binary", "classification"}:
        train_metrics = _classification_metrics(
            model, activations.float(), labels, spec, device
        )
        val_metrics = (
            _classification_metrics(
                model, val_activations.float(), val_labels, spec, device
            )
            if val_activations is not None and val_labels is not None
            else None
        )
        metric_name = "accuracy"
    else:
        train_metrics = _multi_classification_metrics(
            model, activations.float(), labels, spec, device
        )
        val_metrics = (
            _multi_classification_metrics(
                model, val_activations.float(), val_labels, spec, device
            )
            if val_activations is not None and val_labels is not None
            else None
        )
        metric_name = "accuracy"

    selected = val_metrics if val_metrics is not None else train_metrics
    return {
        "metric": metric_name,
        metric_name: selected[metric_name],
        "train": train_metrics,
        "val": val_metrics,
        "spec": dict(spec),
    }


def _load_games_from_chunks(chunks: Sequence[str | Path], n_games: int) -> list[list[int]]:
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games}")
    games: list[list[int]] = []
    for chunk_path in chunks:
        for game in load_chunk(str(chunk_path)):
            games.append(list(game))
            if len(games) >= n_games:
                return games
    if not games:
        raise ValueError("No games loaded; check chunk paths")
    return games


def _valid_ply_indices(game_raw: Sequence[int], board_size: int) -> list[int]:
    max_positions = min(len(game_raw), _block_size(board_size))
    if max_positions <= 0:
        return []

    board = OthelloBoardState(n=board_size)
    valid: list[int] = []
    for ply, move in enumerate(game_raw[:max_positions]):
        board.umpire(int(move))
        if not _is_effectively_game_over(board):
            valid.append(ply)
    return valid


def _select_samples(
    games_raw: Sequence[Sequence[int]],
    board_size: int,
    sample_positions: str,
    *,
    seed: int,
) -> list[tuple[list[int], int]]:
    rng = Random(seed)
    samples: list[tuple[list[int], int]] = []
    for game in games_raw:
        game_list = list(game)
        valid_plys = _valid_ply_indices(game_list, board_size)
        if not valid_plys:
            continue
        if sample_positions == "one_per_game":
            samples.append((game_list, rng.choice(valid_plys)))
        elif sample_positions == "all_positions":
            samples.extend((game_list, ply) for ply in valid_plys)
        else:
            raise ValueError(
                "sample_positions must be 'one_per_game' or 'all_positions', "
                f"got {sample_positions!r}"
            )
    if not samples:
        raise ValueError("No valid probe positions were sampled")
    return samples


def _tokenize_prefix(
    game_raw: Sequence[int],
    ply: int,
    raw_to_token: Sequence[int],
    board_size: int,
) -> list[int]:
    tokens: list[int] = []
    for move in game_raw[: ply + 1]:
        if not (0 <= int(move) < _n_squares(board_size)):
            raise ValueError(f"Raw move {move} is outside the board")
        token = raw_to_token[int(move)]
        if token == -1:
            raise ValueError(f"Raw move {move} is a starting square with no token id")
        tokens.append(token)
    return tokens


def _stack_labels(label_chunks: list[torch.Tensor]) -> torch.Tensor:
    if not label_chunks:
        raise ValueError("No labels collected")
    return torch.cat(label_chunks, dim=0)


def _collect_probe_dataset(
    encoder: nn.Module,
    samples: Sequence[tuple[list[int], int]],
    board_size: int,
    layers: tuple[int, ...],
    features: tuple[str, ...],
    *,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    raw_to_token, _ = build_mappings(board_size)
    pad_token = _n_playable(board_size)
    activation_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    label_chunks: dict[str, list[torch.Tensor]] = {feature: [] for feature in features}

    for start in tqdm(range(0, len(samples), batch_size), desc=desc, unit="batch"):
        batch = list(samples[start : start + batch_size])
        max_len = max(ply + 1 for _, ply in batch)
        token_batch = torch.full((len(batch), max_len), pad_token, dtype=torch.long)
        positions = torch.empty(len(batch), dtype=torch.long)
        batch_labels: dict[str, list[torch.Tensor]] = {feature: [] for feature in features}

        for row, (game_raw, ply) in enumerate(batch):
            token_ids = _tokenize_prefix(game_raw, ply, raw_to_token, board_size)
            token_batch[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            positions[row] = ply
            labels = compute_all_labels(game_raw, ply, board_size)
            for feature in features:
                batch_labels[feature].append(labels[feature])

        acts = extract_activations(
            encoder,
            token_batch.to(device),
            positions.to(device),
            layers,
        )
        for layer, tensor in acts.items():
            activation_chunks[layer].append(tensor)
        for feature in features:
            label_chunks[feature].append(torch.stack(batch_labels[feature], dim=0))

    activations = {
        layer: torch.cat(chunks, dim=0).contiguous()
        for layer, chunks in activation_chunks.items()
    }
    labels = {feature: _stack_labels(chunks).contiguous() for feature, chunks in label_chunks.items()}
    return activations, labels


def run_surface_diagnostics(
    encoder: nn.Module,
    train_chunks: list[Path],
    val_chunks: list[Path],
    board_size: int,
    n_train_games: int,
    n_val_games: int,
    layers: tuple[int, ...],
    features: tuple[str, ...] | None = None,
    sample_positions: str = "one_per_game",
    probe_cfg: ProbeTrainConfig | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the full surface-features diagnostic.

    Returns:
        ``{"features", "layers", "metrics", "config"}``, where metrics are
        indexed by feature name and then layer string. Regression probes report
        R2; classification probes report accuracy.
    """
    _validate_board_size(board_size)
    selected_features = tuple(FEATURE_SPECS if features is None else features)
    unknown = sorted(set(selected_features) - set(FEATURE_SPECS))
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}")

    selected_layers = tuple(int(layer) for layer in layers)
    max_layer = len(_base_model(encoder).blocks)
    invalid_layers = [layer for layer in selected_layers if layer < 0 or layer > max_layer]
    if invalid_layers:
        raise ValueError(f"Invalid layers {invalid_layers}; expected 0..{max_layer}")

    device_obj = _resolve_device(device)
    encoder = _base_model(encoder).to(device_obj)
    encoder.eval()

    probe_cfg = probe_cfg or ProbeTrainConfig(device=str(device_obj))
    if probe_cfg.device != str(device_obj):
        probe_cfg = ProbeTrainConfig(
            epochs=probe_cfg.epochs,
            batch_size=probe_cfg.batch_size,
            lr=probe_cfg.lr,
            weight_decay=probe_cfg.weight_decay,
            device=str(device_obj),
        )

    train_games = _load_games_from_chunks(train_chunks, n_train_games)
    val_games = _load_games_from_chunks(val_chunks, n_val_games)
    train_samples = _select_samples(
        train_games, board_size, sample_positions, seed=17
    )
    val_samples = _select_samples(
        val_games, board_size, sample_positions, seed=29
    )

    collection_batch_size = max(1, min(probe_cfg.batch_size, 512))
    train_acts, train_labels = _collect_probe_dataset(
        encoder,
        train_samples,
        board_size,
        selected_layers,
        selected_features,
        batch_size=collection_batch_size,
        device=device_obj,
        desc="collect train activations",
    )
    val_acts, val_labels = _collect_probe_dataset(
        encoder,
        val_samples,
        board_size,
        selected_layers,
        selected_features,
        batch_size=collection_batch_size,
        device=device_obj,
        desc="collect val activations",
    )

    metrics: dict[str, dict[str, Any]] = {feature: {} for feature in selected_features}
    probe_jobs = [
        (feature, layer)
        for feature in selected_features
        for layer in selected_layers
    ]
    for feature, layer in tqdm(probe_jobs, desc="train linear probes", unit="probe"):
        spec = _resolve_feature_spec(FEATURE_SPECS[feature], board_size)
        result = train_linear_probe(
            train_acts[layer],
            train_labels[feature],
            spec,
            probe_cfg,
            val_activations=val_acts[layer],
            val_labels=val_labels[feature],
        )
        metrics[feature][str(layer)] = result

    return {
        "features": list(selected_features),
        "layers": list(selected_layers),
        "metrics": metrics,
        "config": {
            "board_size": board_size,
            "n_squares": _n_squares(board_size),
            "n_playable": _n_playable(board_size),
            "block_size": _block_size(board_size),
            "n_train_games_requested": n_train_games,
            "n_val_games_requested": n_val_games,
            "n_train_games_loaded": len(train_games),
            "n_val_games_loaded": len(val_games),
            "n_train_samples": len(train_samples),
            "n_val_samples": len(val_samples),
            "sample_positions": sample_positions,
            "probe_cfg": asdict(probe_cfg),
            "train_chunks": [str(path) for path in train_chunks],
            "val_chunks": [str(path) for path in val_chunks],
        },
    }


def _find_smoke_data_dir(board_size: int) -> Path:
    name = f"othello_{board_size}x{board_size}"
    candidates = [
        Path("data") / name,
        Path.cwd() / "data" / name,
        Path(r"G:\My Drive\Master_Thesis_Artifacts\data") / name,
        Path("/content/drive/MyDrive/Master_Thesis_Artifacts/data") / name,
        Path("/content/drive/MyDrive/Master_Thesis_Code/data") / name,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.pickle")):
            return candidate
    raise FileNotFoundError(
        "Could not find smoke-test data. Set the working directory to the "
        "project root or pass a data directory by editing _find_smoke_data_dir."
    )


def _smoke_test() -> None:
    from othello_research.models.gpt import GPTConfig, OthelloGPT

    board_size = 8
    data_dir = _find_smoke_data_dir(board_size)
    chunks = sorted(data_dir.glob("*.pickle"))
    train_chunks = chunks[:1]
    val_chunks = chunks[200:201] if len(chunks) > 200 else chunks[:1]

    cfg = GPTConfig(board_size=board_size, n_layers=2, n_heads=4, d_model=64, dropout=0.1)
    model = OthelloGPT(cfg)
    result = run_surface_diagnostics(
        encoder=model,
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        board_size=board_size,
        n_train_games=100,
        n_val_games=50,
        layers=(0, 1, 2),
        features=("ply_index", "parity"),
        sample_positions="one_per_game",
        probe_cfg=ProbeTrainConfig(epochs=1, batch_size=64, lr=1e-3, device="cuda"),
        device="cuda",
    )
    print(result)


if __name__ == "__main__":
    _smoke_test()
