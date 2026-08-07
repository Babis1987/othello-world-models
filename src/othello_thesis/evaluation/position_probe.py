"""Deterministic, architecture-neutral board-position probes.

The public API in this module deliberately separates three concerns:

1. :func:`build_position_probe_splits` fixes exact train/validation/test
   positions from explicit ``(chunk_path, game_index)`` records.
2. :func:`extract_position_features` runs any encoder exposing the shared
   Othello encoder surface (``wte``, ``wpe``, ``drop``, ``blocks``, ``ln_f``)
   once and caches fp32 last-position activations on CPU.
3. :func:`train_position_probe_bank` fits independent linear or small MLP
   probes, selects a layer on validation data, and reports it once on a
   disjoint test split.

Layer 0 is the residual stream after token/position embeddings and dropout;
layer ``i`` is the residual stream after block ``i``.  Headline probe inputs
use parameter-free LayerNorm; raw linear probes can disable it when fixed
residual-space directions are required for causal interventions. Encoder
extraction uses bf16 autocast on CUDA and fp32 on CPU; probe optimization
itself is fp32.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_thesis.data.chunk_dataset import load_chunk
from othello_thesis.data.move_vocabulary import build_mappings
from othello_thesis.probes.board_state import (
    EMPTY,
    board_state_labels_after_moves,
)


from othello_thesis.evaluation._position_protocol import (
    ABSOLUTE_CLASS_NAMES,
    PROTOCOL_TAG,
    RELATIVE_CLASS_NAMES,
    ChunkLoader,
    GameRecord,
    PositionFeatureSet,
    PositionProbeSplits,
    PositionSample,
    ProbeType,
    _assert_disjoint_records,
    _canonical_path,
    _game_sha256,
    _phase_interval,
    _record_key,
    _require_finite,
    _stable_seed,
    build_position_probe_splits,
    build_position_samples,
)


def _encoder_amp_context(device: torch.device):
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA position-probe extraction requires bf16 support")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def _shared_encoder_states(
    encoder: nn.Module,
    tokens: torch.Tensor,
    requested_layers: set[int],
) -> dict[int, torch.Tensor]:
    """Forward the common Transformer/Mamba encoder surface."""
    required = ("wte", "wpe", "drop", "blocks", "ln_f", "config")
    missing = [name for name in required if not hasattr(encoder, name)]
    if missing:
        raise TypeError(f"Encoder is missing shared-surface attributes: {missing}")

    seq_len = int(tokens.shape[1])
    block_size = int(encoder.config.block_size)
    if seq_len > block_size:
        raise ValueError(f"Input length {seq_len} exceeds encoder block_size={block_size}")

    positions = torch.arange(seq_len, device=tokens.device)
    hidden = encoder.drop(encoder.wte(tokens) + encoder.wpe(positions))
    _require_finite("encoder layer 0", hidden)
    states: dict[int, torch.Tensor] = {}
    if 0 in requested_layers:
        states[0] = hidden

    for layer, block in enumerate(encoder.blocks, start=1):
        hidden = block(hidden)
        _require_finite(f"encoder layer {layer}", hidden)
        if layer in requested_layers:
            states[layer] = hidden

    # Exercise the complete shared surface and fail if final normalization is
    # unhealthy, even though the historical per-block layer convention stores
    # pre-ln_f residual streams.
    final_hidden = encoder.ln_f(hidden)
    _require_finite("encoder final LayerNorm", final_hidden)
    return states


def _prepare_samples(
    samples: Sequence[PositionSample],
    *,
    board_size: int,
    load_chunk_fn: ChunkLoader,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    raw_to_token, _ = build_mappings(board_size)
    n_squares = board_size * board_size
    tokens: list[torch.Tensor | None] = [None] * len(samples)
    absolute = torch.empty((len(samples), n_squares), dtype=torch.long)
    relative = torch.empty_like(absolute)

    indices_by_chunk: dict[str, list[int]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        indices_by_chunk[_canonical_path(sample.chunk_path)].append(sample_index)

    for chunk_path, sample_indices in indices_by_chunk.items():
        games = load_chunk_fn(chunk_path)
        indices_by_game: dict[int, list[int]] = defaultdict(list)
        for sample_index in sample_indices:
            indices_by_game[samples[sample_index].game_index].append(sample_index)

        for game_index, game_sample_indices in indices_by_game.items():
            if not (0 <= game_index < len(games)):
                raise IndexError(
                    f"game_index={game_index} is outside chunk {chunk_path!r} "
                    f"with {len(games)} games"
                )
            game = list(games[game_index])
            first_sample = samples[game_sample_indices[0]]
            max_eligible = min(len(game) - 1, board_size * board_size - 5)
            if len(game) != first_sample.game_length or max_eligible != first_sample.max_eligible_position:
                raise RuntimeError(
                    f"Corpus record changed after manifest creation: "
                    f"{chunk_path!r}[{game_index}]"
                )
            if (
                first_sample.game_sha256
                and _game_sha256(game) != first_sample.game_sha256
            ):
                raise RuntimeError(
                    f"Corpus game content changed after manifest creation: "
                    f"{chunk_path!r}[{game_index}]"
                )

            max_t = max(samples[index].position for index in game_sample_indices)
            label_abs_rows, label_rel_rows = board_state_labels_after_moves(
                game[:max_t], board_size
            )
            token_rows: list[int] = []
            for raw_move in game[:max_t]:
                if not (0 <= raw_move < n_squares):
                    raise ValueError(
                        f"Raw move {raw_move} in {chunk_path!r}[{game_index}] "
                        f"is outside [0, {n_squares})"
                    )
                token = raw_to_token[raw_move]
                if token < 0:
                    raise ValueError(
                        f"Raw move {raw_move} is a starting square and has no token"
                    )
                token_rows.append(int(token))

            for sample_index in game_sample_indices:
                sample = samples[sample_index]
                t = sample.position
                if not (1 <= t <= max_eligible):
                    raise ValueError(f"Invalid sample position t={t} for {sample.sample_id}")
                tokens[sample_index] = torch.tensor(token_rows[:t], dtype=torch.long)
                absolute[sample_index] = torch.tensor(label_abs_rows[t - 1], dtype=torch.long)
                relative[sample_index] = torch.tensor(label_rel_rows[t - 1], dtype=torch.long)
        del games

    if any(value is None for value in tokens):
        raise AssertionError("Internal error: not every PositionSample was prepared")
    return [value for value in tokens if value is not None], absolute, relative


def extract_position_features(
    encoder: nn.Module,
    samples: Sequence[PositionSample],
    *,
    board_size: int,
    layers: Sequence[int],
    batch_size: int = 64,
    device: torch.device | str | None = None,
    load_chunk_fn: ChunkLoader = load_chunk,
) -> PositionFeatureSet:
    """Cache requested last-prefix-position activations as CPU fp32 tensors."""
    if not samples:
        raise ValueError("samples must not be empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    layer_tuple = tuple(int(layer) for layer in layers)
    if not layer_tuple or len(set(layer_tuple)) != len(layer_tuple):
        raise ValueError("layers must be non-empty and unique")
    n_blocks = len(encoder.blocks)
    for layer in layer_tuple:
        if not (0 <= layer <= n_blocks):
            raise ValueError(f"Invalid layer {layer}; expected 0..{n_blocks}")

    try:
        first_parameter = next(encoder.parameters())
    except StopIteration as exc:  # pragma: no cover - real encoders have parameters
        raise ValueError("encoder has no parameters") from exc
    target_device = torch.device(device) if device is not None else first_parameter.device
    if any(parameter.device != target_device for parameter in encoder.parameters()):
        raise ValueError(
            f"All encoder parameters must already be on device={target_device}"
        )
    d_model = int(encoder.config.d_model)

    prepared, absolute, relative = _prepare_samples(
        samples,
        board_size=board_size,
        load_chunk_fn=load_chunk_fn,
    )
    feature_cache = {
        layer: torch.empty((len(samples), d_model), dtype=torch.float32)
        for layer in layer_tuple
    }
    input_pad_token = board_size * board_size - 4
    was_training = encoder.training
    encoder.eval()
    try:
        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            batch_tokens = prepared[start:end]
            lengths = torch.tensor([len(value) for value in batch_tokens], dtype=torch.long)
            max_len = int(lengths.max().item())
            x_cpu = torch.full(
                (len(batch_tokens), max_len), input_pad_token, dtype=torch.long
            )
            for row, value in enumerate(batch_tokens):
                x_cpu[row, : len(value)] = value
            x = x_cpu.to(target_device)
            last_indices = (lengths - 1).to(target_device)
            rows = torch.arange(len(batch_tokens), device=target_device)

            with torch.inference_mode():
                with _encoder_amp_context(target_device):
                    states = _shared_encoder_states(encoder, x, set(layer_tuple))
                for layer in layer_tuple:
                    selected = states[layer][rows, last_indices].float()
                    _require_finite(f"cached layer {layer} features", selected)
                    feature_cache[layer][start:end].copy_(selected.cpu())
    finally:
        encoder.train(was_training)

    phase_bins = torch.tensor([sample.phase_bin for sample in samples], dtype=torch.long)
    feature_set = PositionFeatureSet(
        samples=tuple(samples),
        features=feature_cache,
        absolute_labels=absolute,
        relative_labels=relative,
        phase_bins=phase_bins,
        board_size=board_size,
    )
    _validate_feature_set(feature_set)
    return feature_set


class PositionProbe(nn.Module):
    """Parameter-free input normalization plus linear or one-hidden MLP head."""

    def __init__(
        self,
        d_model: int,
        n_squares: int,
        *,
        probe_type: ProbeType,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        input_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if probe_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown probe_type={probe_type!r}")
        self.n_squares = int(n_squares)
        self.input_norm: nn.Module = (
            nn.LayerNorm(d_model, elementwise_affine=False)
            if input_layernorm
            else nn.Identity()
        )
        output_dim = self.n_squares * 3
        if probe_type == "linear":
            self.proj = nn.Linear(d_model, output_dim)
        else:
            if hidden_dim <= 0:
                raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.proj(self.input_norm(features.float()))
        return logits.view(features.shape[0], self.n_squares, 3)


class PositionProbeBank(nn.Module):
    """Independent absolute and relative probes at every requested layer."""

    def __init__(
        self,
        layers: Sequence[int],
        d_model: int,
        n_squares: int,
        *,
        probe_type: ProbeType,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        input_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        kwargs = {
            "d_model": d_model,
            "n_squares": n_squares,
            "probe_type": probe_type,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "input_layernorm": input_layernorm,
        }
        self.absolute = nn.ModuleDict({
            str(layer): PositionProbe(**kwargs) for layer in self.layers
        })
        self.relative = nn.ModuleDict({
            str(layer): PositionProbe(**kwargs) for layer in self.layers
        })

    def iter_independent_probes(self) -> Iterable[tuple[str, int, PositionProbe]]:
        for mode, bank in (("absolute", self.absolute), ("relative", self.relative)):
            for layer in self.layers:
                yield mode, layer, bank[str(layer)]


def _validate_feature_set(features: PositionFeatureSet) -> None:
    n_samples = len(features.samples)
    if n_samples <= 0:
        raise ValueError("PositionFeatureSet must contain at least one sample")
    if len(set(features.sample_ids)) != n_samples:
        raise ValueError("PositionFeatureSet contains duplicate sample ids")
    expected_squares = features.board_size * features.board_size
    for name, labels in (
        ("absolute_labels", features.absolute_labels),
        ("relative_labels", features.relative_labels),
    ):
        if labels.shape != (n_samples, expected_squares):
            raise ValueError(
                f"{name} has shape {tuple(labels.shape)}; expected "
                f"{(n_samples, expected_squares)}"
            )
        if labels.dtype != torch.long:
            raise TypeError(f"{name} must use torch.long labels")
        if int(labels.min().item()) < 0 or int(labels.max().item()) > 2:
            raise ValueError(f"{name} contains labels outside [0, 2]")
    if features.phase_bins.shape != (n_samples,):
        raise ValueError("phase_bins must have shape (n_samples,)")
    if not features.features:
        raise ValueError("features must contain at least one layer")
    d_model: int | None = None
    for layer, tensor in features.features.items():
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise TypeError(f"Layer {layer} features must be CPU fp32")
        if tensor.ndim != 2 or tensor.shape[0] != n_samples:
            raise ValueError(
                f"Layer {layer} features have invalid shape {tuple(tensor.shape)}"
            )
        if d_model is None:
            d_model = int(tensor.shape[1])
        elif int(tensor.shape[1]) != d_model:
            raise ValueError("All cached layers must use the same d_model")
        _require_finite(f"layer {layer} feature cache", tensor)


def _assert_feature_splits(
    train: PositionFeatureSet,
    validation: PositionFeatureSet,
    test: PositionFeatureSet,
) -> None:
    for feature_set in (train, validation, test):
        _validate_feature_set(feature_set)
    if not (train.board_size == validation.board_size == test.board_size):
        raise ValueError("Feature splits have different board sizes")
    if not (train.layers == validation.layers == test.layers):
        raise ValueError("Feature splits have different layer sets")
    if not (train.d_model == validation.d_model == test.d_model):
        raise ValueError("Feature splits have different d_model values")
    split_ids = {
        "train": set(train.sample_ids),
        "validation": set(validation.sample_ids),
        "test": set(test.sample_ids),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(
                f"{left} and {right} feature sets overlap; first id is "
                f"{sorted(overlap)[0]}"
            )


def _confusion_from_predictions(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> torch.Tensor:
    flat_labels = labels.reshape(-1).to(torch.long).cpu()
    flat_predictions = predictions.reshape(-1).to(torch.long).cpu()
    encoded = flat_labels * 3 + flat_predictions
    return torch.bincount(encoded, minlength=9).reshape(3, 3)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics_from_confusion(
    confusion: torch.Tensor,
    *,
    class_names: Sequence[str],
) -> dict[str, Any]:
    confusion = confusion.to(torch.long).cpu()
    counts = confusion.sum(dim=1)
    total = int(counts.sum().item())
    correct = int(confusion.diag().sum().item())
    recalls = [
        _safe_ratio(int(confusion[index, index].item()), int(counts[index].item()))
        for index in range(3)
    ]
    occupied_total = int((counts[0] + counts[1]).item())
    occupied_correct = int((confusion[0, 0] + confusion[1, 1]).item())
    fractions = [_safe_ratio(int(count.item()), total) for count in counts]
    majority_index = max(range(3), key=lambda index: (int(counts[index]), -index))
    occupied_majority = max(int(counts[0]), int(counts[1]))
    return {
        "accuracy": _safe_ratio(correct, total),
        "occupied_accuracy": _safe_ratio(occupied_correct, occupied_total),
        "empty_accuracy": recalls[EMPTY],
        "per_class_recall": {
            class_names[index]: recalls[index] for index in range(3)
        },
        "macro_balanced_accuracy": float(sum(recalls) / 3.0),
        "class_counts": {
            class_names[index]: int(counts[index].item()) for index in range(3)
        },
        "class_fractions": {
            class_names[index]: fractions[index] for index in range(3)
        },
        "baselines": {
            "majority_class": class_names[majority_index],
            "majority_accuracy": fractions[majority_index],
            "empirical_random_accuracy": float(sum(value * value for value in fractions)),
            "uniform_random_accuracy": 1.0 / 3.0,
            "macro_balanced_random_accuracy": 1.0 / 3.0,
            "occupied_majority_accuracy": _safe_ratio(
                occupied_majority, occupied_total
            ),
        },
        "n_labels": total,
        "n_occupied_labels": occupied_total,
        "n_empty_labels": int(counts[EMPTY].item()),
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_position_probe_bank(
    probe_bank: PositionProbeBank,
    features: PositionFeatureSet,
    *,
    device: torch.device | str,
    batch_size: int = 256,
) -> dict[str, dict[str, Any]]:
    """Evaluate every independent probe, including per-phase metrics."""
    _validate_feature_set(features)
    if tuple(probe_bank.layers) != features.layers:
        raise ValueError("probe_bank layers do not match the feature cache")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    target_device = torch.device(device)
    probe_bank.eval()
    phase_values = sorted(int(value) for value in features.phase_bins.unique().tolist())
    sample_positions = torch.tensor(
        [sample.position for sample in features.samples],
        dtype=torch.long,
    )
    position_values = sorted(
        int(value) for value in sample_positions.unique().tolist()
    )

    aggregate: dict[str, dict[str, torch.Tensor]] = {
        str(layer): {
            "absolute": torch.zeros((3, 3), dtype=torch.long),
            "relative": torch.zeros((3, 3), dtype=torch.long),
        }
        for layer in features.layers
    }
    by_phase: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        str(layer): {
            mode: {
                phase: torch.zeros((3, 3), dtype=torch.long)
                for phase in phase_values
            }
            for mode in ("absolute", "relative")
        }
        for layer in features.layers
    }
    by_position: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        str(layer): {
            mode: {
                position: torch.zeros((3, 3), dtype=torch.long)
                for position in position_values
            }
            for mode in ("absolute", "relative")
        }
        for layer in features.layers
    }

    for start in range(0, features.n_samples, batch_size):
        end = min(start + batch_size, features.n_samples)
        labels_by_mode = {
            "absolute": features.absolute_labels[start:end],
            "relative": features.relative_labels[start:end],
        }
        batch_phase = features.phase_bins[start:end]
        batch_position = sample_positions[start:end]
        for layer in features.layers:
            layer_features = features.features[layer][start:end].to(target_device)
            for mode, bank in (
                ("absolute", probe_bank.absolute),
                ("relative", probe_bank.relative),
            ):
                logits = bank[str(layer)](layer_features)
                _require_finite(f"{mode} layer {layer} logits", logits)
                predictions = logits.argmax(dim=-1).cpu()
                labels = labels_by_mode[mode]
                aggregate[str(layer)][mode] += _confusion_from_predictions(
                    labels, predictions
                )
                for phase in phase_values:
                    mask = batch_phase == phase
                    if bool(mask.any()):
                        by_phase[str(layer)][mode][phase] += _confusion_from_predictions(
                            labels[mask], predictions[mask]
                        )
                for position in batch_position.unique().tolist():
                    position = int(position)
                    mask = batch_position == position
                    if bool(mask.any()):
                        by_position[str(layer)][mode][position] += (
                            _confusion_from_predictions(
                                labels[mask], predictions[mask]
                            )
                        )

    result: dict[str, dict[str, Any]] = {}
    for layer in features.layers:
        layer_key = str(layer)
        result[layer_key] = {}
        for mode, class_names in (
            ("absolute", ABSOLUTE_CLASS_NAMES),
            ("relative", RELATIVE_CLASS_NAMES),
        ):
            metrics = _metrics_from_confusion(
                aggregate[layer_key][mode], class_names=class_names
            )
            metrics["per_phase_bin"] = {
                str(phase): _metrics_from_confusion(
                    by_phase[layer_key][mode][phase], class_names=class_names
                )
                for phase in phase_values
            }
            metrics["per_position"] = {
                str(position): _metrics_from_confusion(
                    by_position[layer_key][mode][position],
                    class_names=class_names,
                )
                for position in position_values
            }
            result[layer_key][mode] = metrics
    return result


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))


def _select_layer(
    validation_metrics: dict[str, dict[str, Any]],
    *,
    layers: Sequence[int],
    mode: str,
    metric: str,
) -> int:
    values: list[tuple[float, int]] = []
    for layer in layers:
        if metric not in validation_metrics[str(layer)][mode]:
            raise KeyError(f"Unknown selection metric {metric!r}")
        value = float(validation_metrics[str(layer)][mode][metric])
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite validation metric for {mode} layer {layer}: {value}"
            )
        values.append((value, int(layer)))
    # Prefer the shallower layer on an exact tie, making selection stable.
    return max(values, key=lambda item: (item[0], -item[1]))[1]


def train_position_probe_bank(
    train_features: PositionFeatureSet,
    validation_features: PositionFeatureSet,
    test_features: PositionFeatureSet,
    *,
    probe_type: ProbeType,
    device: torch.device | str,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float | None = None,
    weight_decay: float = 0.0,
    gradient_clip: float = 1.0,
    seed: int = 42,
    selection_metric: str = "macro_balanced_accuracy",
    input_layernorm: bool = True,
    patience: int | None = None,
    min_delta: float = 0.0,
    early_stopping_modes: Sequence[str] = ("absolute", "relative"),
) -> tuple[PositionProbeBank, dict[str, Any]]:
    """Train independent probes and report validation-selected test metrics.

    Each layer/mode probe has disjoint parameters.  Its own CE contributes
    unscaled to the summed backward loss, while the mean across probes is used
    only for logging.  Gradient clipping is applied separately to every probe,
    avoiding architecture-depth-dependent coupling.
    """
    _assert_feature_splits(train_features, validation_features, test_features)
    if probe_type not in {"linear", "mlp"}:
        raise ValueError(f"Unknown probe_type={probe_type!r}")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive for per-probe clipping")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be positive when provided")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    stopping_modes = tuple(str(mode) for mode in early_stopping_modes)
    if not stopping_modes or any(
        mode not in {"absolute", "relative"} for mode in stopping_modes
    ):
        raise ValueError(
            "early_stopping_modes must contain absolute and/or relative"
        )
    if lr is None:
        lr = 1e-3 if probe_type == "linear" else 3e-4
    if lr <= 0:
        raise ValueError("lr must be positive")

    target_device = torch.device(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    bank = PositionProbeBank(
        train_features.layers,
        train_features.d_model,
        train_features.n_squares,
        probe_type=probe_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        input_layernorm=input_layernorm,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=float(lr), weight_decay=weight_decay
    )

    history: list[dict[str, Any]] = []
    best_validation_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_evaluations = 0
    for epoch in range(1, epochs + 1):
        bank.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch - 1)
        order = torch.randperm(train_features.n_samples, generator=generator)
        logged_loss_sum = 0.0
        seen_samples = 0

        for start in range(0, train_features.n_samples, batch_size):
            batch_indices = order[start : start + batch_size]
            labels_by_mode = {
                "absolute": train_features.absolute_labels[batch_indices].to(target_device),
                "relative": train_features.relative_labels[batch_indices].to(target_device),
            }
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            for layer in train_features.layers:
                layer_features = train_features.features[layer][batch_indices].to(
                    target_device
                )
                for mode, probe_dict in (
                    ("absolute", bank.absolute),
                    ("relative", bank.relative),
                ):
                    logits = probe_dict[str(layer)](layer_features)
                    _require_finite(f"training {mode} layer {layer} logits", logits)
                    loss = _classification_loss(logits, labels_by_mode[mode])
                    _require_finite(f"training {mode} layer {layer} loss", loss)
                    losses.append(loss)

            backward_loss = torch.stack(losses).sum()
            logging_loss = torch.stack([loss.detach() for loss in losses]).mean()
            backward_loss.backward()

            grad_norms: dict[str, float] = {}
            for mode, layer, probe in bank.iter_independent_probes():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    probe.parameters(),
                    max_norm=gradient_clip,
                    error_if_nonfinite=True,
                )
                grad_norms[f"{mode}/{layer}"] = float(grad_norm.detach().cpu())
            optimizer.step()
            for name, parameter in bank.named_parameters():
                _require_finite(f"probe parameter {name}", parameter)

            n_batch = int(batch_indices.numel())
            logged_loss_sum += float(logging_loss.cpu()) * n_batch
            seen_samples += n_batch

        validation_metrics = evaluate_position_probe_bank(
            bank,
            validation_features,
            device=target_device,
            batch_size=batch_size,
        )
        validation_values = [
            float(validation_metrics[str(layer)][mode][selection_metric])
            for layer in train_features.layers
            for mode in stopping_modes
        ]
        mean_validation_score = sum(validation_values) / len(validation_values)
        improved = mean_validation_score > best_validation_score + min_delta
        if improved:
            best_validation_score = mean_validation_score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in bank.state_dict().items()
            }
            stale_evaluations = 0
        else:
            stale_evaluations += 1

        history.append({
            "epoch": epoch,
            "mean_probe_loss": logged_loss_sum / max(1, seen_samples),
            "n_samples": seen_samples,
            "mean_validation_score": mean_validation_score,
            "improved": improved,
            "stale_evaluations": stale_evaluations,
        })
        print(
            f"{probe_type} board probe epoch={epoch}/{epochs} "
            f"loss={history[-1]['mean_probe_loss']:.4f} "
            f"val_mean={mean_validation_score:.4%} "
            f"best_epoch={best_epoch} stale={stale_evaluations}",
            flush=True,
        )
        if patience is not None and stale_evaluations >= patience:
            print(
                f"{probe_type} board probe early stop at epoch={epoch}; "
                f"restoring epoch={best_epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("Board probe training produced no valid checkpoint")
    bank.load_state_dict(best_state)
    validation_metrics = evaluate_position_probe_bank(
        bank,
        validation_features,
        device=target_device,
        batch_size=batch_size,
    )
    test_metrics = evaluate_position_probe_bank(
        bank,
        test_features,
        device=target_device,
        batch_size=batch_size,
    )
    selected: dict[str, Any] = {}
    for mode in ("absolute", "relative"):
        selected_layer = _select_layer(
            validation_metrics,
            layers=train_features.layers,
            mode=mode,
            metric=selection_metric,
        )
        selected[mode] = {
            "layer": selected_layer,
            "selection_metric": selection_metric,
            "validation": validation_metrics[str(selected_layer)][mode],
            "test": test_metrics[str(selected_layer)][mode],
        }

    result = {
        "protocol_tag": PROTOCOL_TAG,
        "probe_type": probe_type,
        "input_layernorm": bool(input_layernorm),
        "input_layernorm_affine": False,
        "encoder_feature_precision": "bf16_cuda_fp32_cache",
        "probe_optimization_precision": "fp32",
        "layers": list(train_features.layers),
        "board_size": train_features.board_size,
        "d_model": train_features.d_model,
        "n_squares": train_features.n_squares,
        "hidden_dim": hidden_dim if probe_type == "mlp" else None,
        "mlp_activation": "gelu" if probe_type == "mlp" else None,
        "dropout": dropout if probe_type == "mlp" else 0.0,
        "epochs": epochs,
        "max_epochs": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "early_stopping_modes": list(stopping_modes),
        "batch_size": batch_size,
        "lr": float(lr),
        "weight_decay": weight_decay,
        "gradient_clip_per_probe": gradient_clip,
        "seed": seed,
        "selection_metric": selection_metric,
        "sample_counts": {
            "train": train_features.n_samples,
            "validation": validation_features.n_samples,
            "test": test_features.n_samples,
        },
        "sample_ids": {
            "train": list(train_features.sample_ids),
            "validation": list(validation_features.sample_ids),
            "test": list(test_features.sample_ids),
        },
        "history": history,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selected": selected,
    }
    return bank, result


__all__ = [
    "ABSOLUTE_CLASS_NAMES",
    "RELATIVE_CLASS_NAMES",
    "PROTOCOL_TAG",
    "GameRecord",
    "PositionSample",
    "PositionProbeSplits",
    "PositionFeatureSet",
    "PositionProbe",
    "PositionProbeBank",
    "build_position_samples",
    "build_position_probe_splits",
    "extract_position_features",
    "evaluate_position_probe_bank",
    "train_position_probe_bank",
]
