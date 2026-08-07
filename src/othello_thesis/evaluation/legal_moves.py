"""Legal-move evaluation utilities for trained OthelloGPT models.

Cross-entropy on uniformly sampled legal Othello games bottoms out at the
entropy of the legal-move distribution, so loss alone is not a direct test of
whether the model has learned the rules. This module evaluates model outputs
against the rule engine by measuring top-1 legality, top-k legal concentration,
and probability mass assigned to legal moves.

The model uses compact token ids, while the rule engine uses raw board
positions. All legality checks are replayed on CPU with raw moves, then mapped
back to token ids before comparing against model logits.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch

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


# === Helpers ===

def _topk_indices(logits: torch.Tensor, k: int) -> list[int]:
    """Return the top-k token indices from a 1-D logits/probability tensor."""
    if logits.ndim != 1:
        raise ValueError(f"Expected a 1-D tensor, got shape {tuple(logits.shape)}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    actual_k = min(k, int(logits.numel()))
    return torch.topk(logits, k=actual_k).indices.detach().cpu().tolist()


def _legal_token_set(
    board: OthelloBoardState,
    raw_to_token: Sequence[int],
    token_to_raw: Sequence[int],
) -> set[int]:
    """Return the current legal moves as compact token ids."""
    legal: set[int] = set()
    for raw_move in board.get_valid_moves():
        token = raw_to_token[raw_move]
        if token != -1:
            if token >= len(token_to_raw):
                raise ValueError(f"Token id {token} is outside token_to_raw")
            legal.add(token)
    return legal


def _amp_context(device: torch.device):
    """Use bf16 autocast on CUDA; keep CPU evaluation in fp32.

    bf16 is required for the Mamba/SSM checkpoints: their finite residual
    streams can exceed fp16's maximum value (65,504), so fp16 evaluation can
    overflow even when the bf16-trained model is numerically healthy.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    """Fail fast instead of turning non-finite model outputs into valid-looking metrics."""
    if not torch.isfinite(tensor).all():
        nonfinite = int((~torch.isfinite(tensor)).sum().item())
        raise RuntimeError(
            f"Legal-move evaluation produced {nonfinite} non-finite values in {name} "
            f"(shape={tuple(tensor.shape)}, dtype={tensor.dtype})."
        )


def _model_block_size(model: torch.nn.Module, board_size: int) -> int:
    """Infer the model context length, falling back to the Othello default."""
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    config = getattr(base_model, "config", None)
    block_size = getattr(config, "block_size", None)
    if block_size is None:
        block_size = board_size * board_size - 5
    return int(block_size)


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or NaN for an empty sequence."""
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _position_means(sums: Sequence[float], counts: Sequence[int]) -> list[float]:
    """Convert per-position sums/counts into means with NaN for empty bins."""
    return [
        float(total / count) if count else float("nan")
        for total, count in zip(sums, counts)
    ]


def _tokenize_game(
    game_raw: Sequence[int],
    raw_to_token: Sequence[int],
    board_size: int,
) -> list[int]:
    """Convert one raw game to token ids with explicit data validation."""
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


def _empty_game_result(k_values: Sequence[int]) -> dict[str, object]:
    """Return the public single-game result shape with no scored positions."""
    return {
        "top1_correct": [],
        "topk_legal_fraction": {int(k): [] for k in k_values},
        "legal_prob_mass": [],
        "n_legal": [],
    }


def _score_probability_rows(
    probs: torch.Tensor,
    game_raw: Sequence[int],
    board_size: int,
    k_values: Sequence[int],
) -> tuple[dict[str, object], list[int]]:
    """Score real probability rows for one game.

    ``probs[t]`` is produced from the prefix ending at move ``t`` and predicts
    move ``t + 1``. We replay the truncated raw game up to the last available
    target so ``legal_sets[t + 1]`` is the legal set for the move being
    predicted. The empty-prefix prediction for move 0 is not scored.
    """
    result = _empty_game_result(k_values)
    position_indices: list[int] = []
    if probs.ndim != 2:
        raise ValueError(f"Expected probs shape (T, vocab), got {tuple(probs.shape)}")
    if len(game_raw) < 2:
        return result, position_indices

    legal_sets = replay_game_legality(game_raw, board_size)
    n_positions = min(int(probs.shape[0]), len(legal_sets) - 1)
    top1_correct = result["top1_correct"]
    topk_legal_fraction = result["topk_legal_fraction"]
    legal_prob_mass = result["legal_prob_mass"]
    n_legal = result["n_legal"]

    assert isinstance(top1_correct, list)
    assert isinstance(topk_legal_fraction, dict)
    assert isinstance(legal_prob_mass, list)
    assert isinstance(n_legal, list)

    for t in range(n_positions):
        target_move_index = t + 1
        legal = legal_sets[target_move_index]
        if not legal:
            continue

        row = probs[t]
        top1 = int(torch.argmax(row).item())
        top1_correct.append(1 if top1 in legal else 0)

        for k in k_values:
            k_int = int(k)
            topk = _topk_indices(row, k_int)
            topk_legal = sum(1 for token in topk if token in legal)
            topk_legal_fraction[k_int].append(topk_legal / len(topk))

        legal_indices = torch.tensor(sorted(legal), dtype=torch.long, device=row.device)
        legal_prob_mass.append(float(row.index_select(0, legal_indices).sum().item()))
        n_legal.append(len(legal))
        position_indices.append(t)

    return result, position_indices


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return a deterministic percentile bootstrap interval over games."""
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")

    data = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    means: list[torch.Tensor] = []
    remaining = n_resamples
    while remaining:
        count = min(100, remaining)
        indices = torch.randint(
            len(data),
            (count, len(data)),
            generator=generator,
        )
        means.append(data[indices].mean(dim=1))
        remaining -= count
    samples = torch.cat(means)
    alpha = (1.0 - confidence) / 2.0
    quantiles = torch.quantile(
        samples,
        torch.tensor([alpha, 1.0 - alpha], dtype=samples.dtype),
    )
    return {"low": float(quantiles[0]), "high": float(quantiles[1])}


def _pad_token_batch(
    batch_tokens: Sequence[Sequence[int]],
    pad_token: int,
) -> tuple[torch.Tensor, list[int]]:
    """Pad tokenized games for a batched next-token model forward pass."""
    input_lengths = [len(tokens) - 1 for tokens in batch_tokens]
    max_len = max(input_lengths)
    x = torch.full((len(batch_tokens), max_len), pad_token, dtype=torch.long)
    for row, tokens in enumerate(batch_tokens):
        input_len = input_lengths[row]
        x[row, :input_len] = torch.tensor(tokens[:-1], dtype=torch.long)
    return x, input_lengths


# === Per-game replay ===

def replay_game_legality(game_raw: Sequence[int], board_size: int) -> list[set[int]]:
    """Replay one raw game and collect legal token sets before each move.

    Corpus games omit pass markers. If the side to move has no legal move, the
    following raw move therefore belongs to the opponent. In that case this
    function records the opponent's legal set, matching the move that the model
    is asked to predict, while leaving :meth:`OthelloBoardState.umpire` to apply
    the implicit pass during replay.

    Args:
        game_raw: Raw board positions, each in ``0..board_size**2 - 1``.
        board_size: Side length of the Othello board.

    Returns:
        A list with one token-id set per move in ``game_raw``. Element ``i`` is
        the legal move set immediately before ``game_raw[i]`` was played.
    """
    raw_to_token, token_to_raw = build_mappings(board_size)
    board = OthelloBoardState(n=board_size)
    legal_sets: list[set[int]] = []

    for move in game_raw:
        legal = _legal_token_set(board, raw_to_token, token_to_raw)
        if not legal:
            # Games contain board moves only, not explicit pass tokens. Inspect
            # the opponent's turn without changing the board state that umpire
            # expects; umpire will perform the same implicit pass before playing
            # ``move``. If neither player can move, ``legal`` remains empty and
            # umpire raises for the invalid trailing move as before.
            board.next_hand_color *= -1
            try:
                legal = _legal_token_set(board, raw_to_token, token_to_raw)
            finally:
                board.next_hand_color *= -1

        legal_sets.append(legal)
        board.umpire(move)

    return legal_sets


# === Single-game evaluation ===

def evaluate_one_game(
    model: torch.nn.Module,
    game_tokens: torch.Tensor,
    game_raw: Sequence[int],
    board_size: int,
    device: torch.device | str,
    k_values: Sequence[int],
) -> dict[str, object]:
    """Evaluate legal-move metrics for one game with a single model forward."""
    device = torch.device(device)
    tokens = game_tokens.detach().cpu().long()
    raw = list(game_raw)
    if len(tokens) != len(raw):
        raise ValueError(
            f"game_tokens and game_raw length mismatch: {len(tokens)} != {len(raw)}"
        )

    max_len = _model_block_size(model, board_size) + 1
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        raw = raw[:max_len]
    if len(tokens) < 2:
        return _empty_game_result(k_values)

    model.eval()
    idx_batch = tokens[:-1].unsqueeze(0).to(device)
    with torch.inference_mode():
        with _amp_context(device):
            logits, _ = model(idx_batch)
        scored_logits = logits[0, : idx_batch.shape[1]].float()
        _require_finite("logits", scored_logits)
        probs = torch.softmax(scored_logits, dim=-1)
        _require_finite("probabilities", probs)
        probs = probs.cpu()

    result, _ = _score_probability_rows(probs, raw, board_size, k_values)
    return result


# === Aggregate evaluation ===

def evaluate_legal_moves(
    model: torch.nn.Module,
    val_chunks: Sequence[str | Path],
    board_size: int,
    n_games: int,
    device: torch.device | str,
    k_values: Sequence[int] = (1, 2, 3, 5),
    batch_size: int = 64,
) -> dict[str, object]:
    """Evaluate OthelloGPT legality metrics over validation chunks.

    Games are consumed deterministically in chunk order. Model forwards are
    batched on the requested device, while rule-engine replay stays on CPU.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive, got {n_games}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    device = torch.device(device)
    raw_to_token, _ = build_mappings(board_size)
    block_size = _model_block_size(model, board_size)
    max_game_len = block_size + 1
    input_pad_token = board_size * board_size - 4

    games_raw: list[list[int]] = []
    for chunk_path in val_chunks:
        if len(games_raw) >= n_games:
            break
        chunk_games = load_chunk(chunk_path)
        for game in chunk_games:
            if len(games_raw) >= n_games:
                break
            games_raw.append(list(game))

    if not games_raw:
        raise ValueError("No games were loaded from val_chunks")

    game_records: list[tuple[list[int], list[int]]] = []
    for game in games_raw:
        raw = game[:max_game_len]
        if len(raw) < 2:
            continue
        tokens = _tokenize_game(raw, raw_to_token, board_size)
        game_records.append((tokens, raw))

    if not game_records:
        raise ValueError("No games with at least 2 moves were available to evaluate")

    k_values = tuple(int(k) for k in k_values)
    model.eval()

    all_top1: list[float] = []
    all_mass: list[float] = []
    all_n_legal: list[float] = []
    all_topk: dict[int, list[float]] = {k: [] for k in k_values}
    per_game_top1: list[float] = []
    per_game_mass: list[float] = []

    position_count = board_size * board_size - 5
    top1_pos_sum = [0.0] * position_count
    mass_pos_sum = [0.0] * position_count
    n_legal_pos_sum = [0.0] * position_count
    pos_count = [0] * position_count

    pbar = tqdm(total=len(game_records), desc="legal-move eval", unit="game")
    for start in range(0, len(game_records), batch_size):
        batch = game_records[start : start + batch_size]
        batch_tokens = [tokens for tokens, _ in batch]
        batch_raw = [raw for _, raw in batch]
        x_cpu, input_lengths = _pad_token_batch(batch_tokens, input_pad_token)
        x = x_cpu.to(device, non_blocking=device.type == "cuda")

        with torch.inference_mode():
            with _amp_context(device):
                logits, _ = model(x)
            scored_logits = logits.float()
            _require_finite("logits", scored_logits)
            probs = torch.softmax(scored_logits, dim=-1)
            _require_finite("probabilities", probs)
            probs = probs.cpu()

        for row, raw in enumerate(batch_raw):
            input_len = input_lengths[row]
            game_probs = probs[row, :input_len]
            scores, position_indices = _score_probability_rows(
                game_probs, raw, board_size, k_values
            )

            top1 = scores["top1_correct"]
            mass = scores["legal_prob_mass"]
            topk = scores["topk_legal_fraction"]
            n_legal = scores["n_legal"]
            assert isinstance(top1, list)
            assert isinstance(mass, list)
            assert isinstance(topk, dict)
            assert isinstance(n_legal, list)

            if top1:
                all_top1.extend(float(x) for x in top1)
                all_mass.extend(float(x) for x in mass)
                all_n_legal.extend(float(x) for x in n_legal)
                per_game_top1.append(_mean([float(x) for x in top1]))
                per_game_mass.append(_mean([float(x) for x in mass]))
                for k in k_values:
                    all_topk[k].extend(float(x) for x in topk[k])

                for i, pos in enumerate(position_indices):
                    if pos >= position_count:
                        continue
                    top1_pos_sum[pos] += float(top1[i])
                    mass_pos_sum[pos] += float(mass[i])
                    n_legal_pos_sum[pos] += float(n_legal[i])
                    pos_count[pos] += 1

            pbar.update(1)
    pbar.close()

    n_tokens = len(all_top1)
    if n_tokens == 0:
        raise ValueError("Evaluation produced zero scorable positions")

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    config = getattr(base_model, "config", None)
    vocab_size = int(
        getattr(config, "vocab_size", board_size * board_size - 3)
    )
    random_vocab_legal_baseline = _mean(
        [n_legal / vocab_size for n_legal in all_n_legal]
    )
    top1_mean = _mean(all_top1)
    normalized_legal_lift = (
        (top1_mean - random_vocab_legal_baseline)
        / max(1e-12, 1.0 - random_vocab_legal_baseline)
    )

    return {
        "n_games": len(per_game_top1),
        "n_tokens": n_tokens,
        "top1_legal_per_token": top1_mean,
        "top1_legal_per_game": _mean(per_game_top1),
        "top1_legal_per_position": _position_means(top1_pos_sum, pos_count),
        "topk_legal": {k: _mean(values) for k, values in all_topk.items()},
        "legal_prob_mass_per_token": _mean(all_mass),
        "legal_prob_mass_per_game": _mean(per_game_mass),
        "legal_prob_mass_per_position": _position_means(mass_pos_sum, pos_count),
        "n_tokens_per_position": list(pos_count),
        "mean_legal_moves_per_token": _mean(all_n_legal),
        "random_vocab_top1_legal_baseline": random_vocab_legal_baseline,
        "top1_legal_normalized_lift": normalized_legal_lift,
        "bootstrap_ci_95_per_game": {
            "top1_legal": _bootstrap_mean_ci(per_game_top1, seed=42),
            "legal_probability_mass": _bootstrap_mean_ci(per_game_mass, seed=43),
        },
        "n_legal_per_position_mean": _position_means(n_legal_pos_sum, pos_count),
    }


def _smoke_test() -> None:
    """Run a tiny untrained-model smoke test from the repository root data."""
    from othello_thesis.models.transformer import GPTConfig, OthelloGPT

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "othello_8x8"
    chunks = sorted(data_dir.glob("*.pickle"))
    if not chunks:
        raise FileNotFoundError(f"No .pickle files found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GPTConfig(board_size=8, n_layers=2, n_heads=4, d_model=64)
    model = OthelloGPT(cfg).to(device)
    result = evaluate_legal_moves(
        model=model,
        val_chunks=chunks[:1],
        board_size=8,
        n_games=50,
        device=device,
        k_values=(1, 2, 3, 5),
        batch_size=16,
    )
    compact = {
        key: value
        for key, value in result.items()
        if not key.endswith("_per_position") and key != "n_legal_per_position_mean"
    }
    print(compact)


if __name__ == "__main__":
    _smoke_test()
