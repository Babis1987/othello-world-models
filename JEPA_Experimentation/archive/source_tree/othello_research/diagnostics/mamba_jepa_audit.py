"""Diagnostics for the mamba_jepa_v1_hd_vicreg smoke-eval anomaly.

Companion module for ``notebooks/mamba_jepa_diagnostics.ipynb``. Everything
here reuses the existing metric implementations (``evaluate_legal_moves``,
``_score_probability_rows``, ``ActivationCache``, ``_iter_probe_batches``,
``train_matched_probe_bank``) so numbers stay comparable with the smoke eval.

Audit finding this module exists to demonstrate and quantify:
``othello_research/evaluation/legal_moves.py::_amp_context`` hardcodes fp16
autocast on CUDA, while the Mamba residual stream exceeds the fp16 range
(max |activation| ~5e5 vs fp16 max 65504), so every legal-move evaluation of
a Mamba encoder produced NaN logits from position 1 onward.
"""

from __future__ import annotations

import json
import platform
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_research.datasets.dataset import (
    TARGET_PAD,
    OthelloChunkDataset,
    load_chunk,
)
from othello_research.datasets.move_mapping import build_mappings
from othello_research.evaluation.legal_moves import (
    _score_probability_rows,
    _tokenize_game,
    evaluate_legal_moves,
)
from othello_research.probes.board_state import (
    LABEL_PAD,
    ActivationCache,
    _iter_probe_batches,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **_):
        return x if x is not None else iter(())


# ============================================================================
# Environment / reproducibility
# ============================================================================

def collect_versions() -> dict[str, str]:
    """Torch / CUDA / mamba-ssm versions for the audit report."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }
    try:
        import mamba_ssm  # type: ignore

        info["mamba_ssm"] = getattr(mamba_ssm, "__version__", "installed")
    except Exception:
        info["mamba_ssm"] = "not installed"
    return info


def _autocast(device: torch.device, precision: str):
    """Explicit-precision autocast context. ``fp32`` disables autocast."""
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


# ============================================================================
# Precision sweep (mechanism check for the fp16 overflow)
# ============================================================================

@torch.no_grad()
def precision_sweep(
    encoder: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    precisions: Sequence[str] = ("fp32", "bf16", "fp16"),
) -> dict[str, list[dict[str, float | bool]]]:
    """Forward one batch under each precision; report per-layer max|act| and finiteness.

    Layer indices follow the board-probe convention: 0 = embedding stream,
    1..N = residual stream after each block.
    """
    encoder.eval()
    layers = tuple(range(len(encoder.blocks) + 1))
    results: dict[str, list[dict[str, float | bool]]] = {}
    for precision in precisions:
        rows: list[dict[str, float | bool]] = []
        with ActivationCache(encoder, layers) as cache:
            with _autocast(device, precision):
                encoder(x.to(device))
            for layer in layers:
                acts = cache.activations[layer]
                finite = bool(torch.isfinite(acts).all())
                max_abs = float(acts[torch.isfinite(acts)].abs().max()) if finite or torch.isfinite(acts).any() else float("nan")
                rows.append({
                    "layer": layer,
                    "max_abs": max_abs,
                    "all_finite": finite,
                    "nan_fraction": float((~torch.isfinite(acts)).float().mean()),
                })
        results[precision] = rows
    return results


def print_precision_sweep(results: dict[str, list[dict[str, float | bool]]]) -> None:
    precisions = list(results)
    print("layer | " + " | ".join(f"{p}: max|act| (finite)" for p in precisions))
    n_layers = len(results[precisions[0]])
    for i in range(n_layers):
        cells = []
        for p in precisions:
            row = results[p][i]
            cells.append(f"{row['max_abs']:>12.1f} ({'ok' if row['all_finite'] else 'NON-FINITE'})")
        print(f"{results[precisions[0]][i]['layer']:>5} | " + " | ".join(cells))
    print("fp16 representable max: 65504")


# ============================================================================
# Precision-forced next-move model (repro + fix for the legal-move evaluator)
# ============================================================================

class PrecisionForcedNextMoveModel(nn.Module):
    """Frozen encoder + trained head with an explicit internal compute precision.

    ``evaluate_legal_moves`` wraps ``model(x)`` in a hardcoded fp16 autocast.
    With ``precision='inherit'`` this wrapper obeys the outer context and
    reproduces the broken smoke-eval numbers. With ``'fp32'``/``'bf16'`` it
    overrides the outer autocast from inside ``forward``, so the unmodified
    evaluator produces valid numbers.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module, precision: str = "fp32"):
        super().__init__()
        if precision not in {"inherit", "fp32", "bf16", "fp16"}:
            raise ValueError(f"Unknown precision {precision!r}")
        self.encoder = encoder
        self.head = head
        self.precision = precision
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def config(self):
        return self.encoder.config

    def _encode(self, idx: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        pos = torch.arange(idx.size(1), device=idx.device)
        h = enc.drop(enc.wte(idx) + enc.wpe(pos))
        for block in enc.blocks:
            h = block(h)
        return enc.ln_f(h)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        with torch.no_grad():
            if self.precision == "inherit":
                h = self._encode(idx)
            else:
                with _autocast(idx.device, self.precision):
                    h = self._encode(idx)
            logits = self.head(h.float())
        return logits, None


class HookPathNextMoveModel(nn.Module):
    """Next-move head on board-probe hook-path features at one layer.

    Features come from ``ActivationCache`` (residual stream, pre-``ln_f``) at
    ``layer`` — the path that produced sane board-probe numbers. Optional
    input LayerNorm addresses the unnormalized-scale hypothesis for the MLP
    probes. Compatible with ``evaluate_legal_moves`` (precision-forced).
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        layer: int,
        *,
        input_norm: nn.Module | None = None,
        precision: str = "bf16",
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.layer = int(layer)
        self.input_norm = input_norm
        self.precision = precision
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def config(self):
        return self.encoder.config

    def extract(self, idx: torch.Tensor) -> torch.Tensor:
        """Hook-path features at ``self.layer`` for a token batch (float32)."""
        with torch.no_grad(), ActivationCache(self.encoder, (self.layer,)) as cache:
            with _autocast(idx.device, self.precision):
                self.encoder(idx)
            return cache.activations[self.layer].to(idx.device).float()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        feats = self.extract(idx)
        if self.input_norm is not None:
            feats = self.input_norm(feats)
        logits = self.head(feats)
        loss = None
        if targets is not None:
            flat = targets.reshape(-1)
            valid = (flat != TARGET_PAD).sum().clamp_min(1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                flat,
                ignore_index=TARGET_PAD,
                reduction="sum",
            ) / valid
        return logits, loss


def make_mlp_head(
    d_model: int,
    vocab_size: int,
    hidden_dim: int = 512,
    n_layers: int = 1,
    dropout: float = 0.1,
) -> nn.Module:
    """Same MLP shape as the smoke-eval notebook's MLPProbe head."""
    layers: list[nn.Module] = [nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, vocab_size))
    return nn.Sequential(*layers)


def load_saved_head(payload_path: str | Path, d_model: int, vocab_size: int) -> nn.Module:
    """Rebuild a smoke-eval head module from its saved artifact."""
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    state = payload["head"]
    if set(state) == {"weight", "bias"}:
        head = nn.Linear(d_model, vocab_size)
        head.load_state_dict(state)
    else:
        hidden_dim = int(payload.get("hidden_dim", 512))
        n_layers = int(payload.get("n_layers", 1))
        head = make_mlp_head(d_model, vocab_size, hidden_dim, n_layers, dropout=0.0)
        # The smoke-eval MLPProbe wrapper stores its Sequential under ``net``,
        # producing keys such as ``net.0.weight``.  The diagnostic helper
        # rebuilds the Sequential directly, whose corresponding key is
        # ``0.weight``.  Also accept already-unwrapped artifacts so this stays
        # compatible with summaries/checkpoints regenerated by diagnostics.
        prefix = "net."
        unwrapped_state = {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state.items()
        }
        head.load_state_dict(unwrapped_state)
    head.eval()
    return head


# ============================================================================
# Section B: objective-solved test (nearest target prototype)
# ============================================================================

@torch.no_grad()
def target_prototypes(
    jepa_model: nn.Module,
    device: torch.device,
    *,
    precision: str = "fp32",
) -> torch.Tensor:
    """Encode all move tokens as length-1 target sequences at every position.

    Returns ``(block_size, n_moves, d_model)``: prototypes[t] are the target
    embeddings the objective used when the context length was t (absolute
    position t, matching ``build_target_positions``).
    """
    encoder = jepa_model.context_encoder
    config = encoder.config
    n_moves = int(config.vocab_size) - 1
    block_size = int(config.block_size)
    idx = torch.arange(n_moves, device=device).unsqueeze(1)
    protos = []
    for t in range(block_size):
        positions = torch.full_like(idx, t)
        with _autocast(device, precision):
            hidden = jepa_model.encode_hidden(encoder, idx, positions)
        protos.append(hidden[:, -1, :].float())
    return torch.stack(protos)


@torch.no_grad()
def evaluate_objective_solved(
    jepa_model: nn.Module,
    games: Sequence[Sequence[int]],
    device: torch.device,
    *,
    board_size: int = 8,
    metric: str = "l2",
    precision: str = "fp32",
    batch_size: int = 32,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Classify each position by nearest length-1 target prototype.

    ``z_pred = predictor(h[:, t-1])`` is compared against the 60 prototypes at
    absolute position t; the negative distances (or cosine similarities) feed
    the existing legal-move scorer. Reports exact top-1 vs the sampled
    continuation AND top-1 legality.
    """
    encoder = jepa_model.context_encoder
    config = encoder.config
    raw_to_token, _ = build_mappings(board_size)
    block_size = int(config.block_size)
    protos = target_prototypes(jepa_model, device, precision=precision)

    exact, top1_legal, topk_acc = [], [], {int(k): [] for k in k_values}
    for start in tqdm(range(0, len(games), batch_size), desc=f"objective-solved ({metric})"):
        batch = [list(g)[: block_size + 1] for g in games[start : start + batch_size]]
        batch = [g for g in batch if len(g) >= 2]
        if not batch:
            continue
        pad = int(config.vocab_size) - 1
        lengths = [len(g) - 1 for g in batch]
        max_len = max(lengths)
        x = torch.full((len(batch), max_len), pad, dtype=torch.long)
        for i, g in enumerate(batch):
            x[i, : lengths[i]] = torch.tensor(
                _tokenize_game(g[:-1], raw_to_token, board_size), dtype=torch.long
            )
        x = x.to(device)
        with _autocast(device, precision):
            hidden = jepa_model.encode_hidden(encoder, x)
        z_pred = jepa_model.predictor(hidden.float())  # (B, T, d), linear predictor

        for i, g in enumerate(batch):
            # Absolute target positions must stay within wpe range (0..block_size-1),
            # matching build_target_positions during training.
            T = min(lengths[i], block_size - 1)
            zp = z_pred[i, :T]  # zp[t-1] predicts move t (context x[:t])
            pt = protos[1 : T + 1]  # prototypes at absolute positions 1..T
            if metric == "l2":
                logits = -torch.cdist(zp.unsqueeze(1), pt).squeeze(1)  # (T, 60)
            elif metric == "cosine":
                logits = F.cosine_similarity(zp.unsqueeze(1), pt, dim=-1)
            else:
                raise ValueError(f"Unknown metric {metric!r}")
            probs = torch.softmax(logits, dim=-1)
            probs = torch.cat([probs, probs.new_zeros(probs.size(0), 1)], dim=1)
            true_tokens = torch.tensor(
                _tokenize_game(g[1:], raw_to_token, board_size), device=device
            )[:T]
            if logits.size(0) != true_tokens.size(0):
                raise RuntimeError(
                    "Objective-solved alignment mismatch after truncation: "
                    f"logits={logits.size(0)}, targets={true_tokens.size(0)}, "
                    f"T={T}, game_len={len(g)}, block_size={block_size}."
                )
            exact.extend((logits.argmax(-1) == true_tokens).float().tolist())
            scores, _ = _score_probability_rows(probs.cpu(), g, board_size, k_values)
            top1_legal.extend(scores["top1_correct"])
            for k in k_values:
                topk_acc[int(k)].extend(scores["topk_legal_fraction"][int(k)])

    return {
        "metric": metric,
        "n_positions": len(exact),
        "exact_top1": float(torch.tensor(exact).mean()),
        "top1_legal": float(torch.tensor(top1_legal, dtype=torch.float32).mean()),
        **{f"top{k}_legal_fraction": float(torch.tensor(v, dtype=torch.float32).mean())
           for k, v in topk_acc.items()},
    }


# ============================================================================
# Section C: smoke-eval probe replica (for random-init controls)
# ============================================================================

class _BankProbe(nn.Module):
    """Same probe as the smoke-eval notebook's BoardProbe."""

    def __init__(self, d_model: int, n_squares: int, probe_type: str,
                 hidden_dim: int = 512, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_squares = n_squares
        if probe_type == "linear":
            self.proj: nn.Module = nn.Linear(d_model, n_squares * 3)
        elif probe_type == "mlp":
            self.proj = make_mlp_head(d_model, n_squares * 3, hidden_dim, n_layers, dropout)
        else:
            raise ValueError(f"Unknown probe_type {probe_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x)
        return logits.view(*logits.shape[:-1], self.n_squares, 3)


def train_probe_bank_replica(
    context_encoder: nn.Module,
    train_games: Sequence[Sequence[int]],
    val_games: Sequence[Sequence[int]],
    device: torch.device,
    *,
    probe_type: str = "linear",
    board_size: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 1,
    precision: str = "bf16",
    seed: int = 42,
) -> dict:
    """Replicate the smoke-eval notebook's board-probe pipeline exactly.

    One AdamW over per-layer absolute+relative probes, joint loss averaged over
    2*n_layers heads, trained on all positions, evaluated on all positions.
    Pass a random-init encoder for the reservoir control.
    """
    torch.manual_seed(seed)
    context_encoder.eval()
    layers = tuple(range(len(context_encoder.blocks) + 1))
    d_model = int(context_encoder.config.d_model)
    n_squares = board_size * board_size
    absolute = nn.ModuleDict({str(l): _BankProbe(d_model, n_squares, probe_type) for l in layers}).to(device)
    relative = nn.ModuleDict({str(l): _BankProbe(d_model, n_squares, probe_type) for l in layers}).to(device)
    optimizer = torch.optim.AdamW(
        [*absolute.parameters(), *relative.parameters()], lr=lr, weight_decay=weight_decay
    )

    def probe_loss(logits, labels):
        return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), ignore_index=LABEL_PAD)

    block_size = int(context_encoder.config.block_size)
    with ActivationCache(context_encoder, layers) as cache:
        for _ in range(epochs):
            batches = _iter_probe_batches(train_games, board_size, block_size, batch_size)
            for x_cpu, y_abs_cpu, y_rel_cpu in tqdm(batches, desc=f"{probe_type} probes (train)"):
                x, y_abs, y_rel = (t.to(device) for t in (x_cpu, y_abs_cpu, y_rel_cpu))
                cache.clear()
                with torch.no_grad(), _autocast(device, precision):
                    context_encoder(x)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.zeros((), device=device)
                for layer in layers:
                    acts = cache.activations[layer].to(device)
                    loss = loss + probe_loss(absolute[str(layer)](acts), y_abs)
                    loss = loss + probe_loss(relative[str(layer)](acts), y_rel)
                (loss / (2 * len(layers))).backward()
                optimizer.step()

        counts = {str(l): {"absolute": [0, 0], "relative": [0, 0]} for l in layers}
        with torch.no_grad():
            batches = _iter_probe_batches(val_games, board_size, block_size, batch_size)
            for x_cpu, y_abs_cpu, y_rel_cpu in tqdm(batches, desc=f"{probe_type} probes (val)"):
                x, y_abs, y_rel = (t.to(device) for t in (x_cpu, y_abs_cpu, y_rel_cpu))
                cache.clear()
                with _autocast(device, precision):
                    context_encoder(x)
                for layer in layers:
                    acts = cache.activations[layer].to(device)
                    for mode, bank, labels in (
                        ("absolute", absolute, y_abs),
                        ("relative", relative, y_rel),
                    ):
                        preds = bank[str(layer)](acts).argmax(-1)
                        valid = labels != LABEL_PAD
                        counts[str(layer)][mode][0] += int(((preds == labels) & valid).sum())
                        counts[str(layer)][mode][1] += int(valid.sum())

    metrics = {
        key: {
            mode: {"accuracy": (c[0] / c[1] if c[1] else float("nan")), "n_labels": c[1]}
            for mode, c in layer_counts.items()
        }
        for key, layer_counts in counts.items()
    }
    return {"probe_type": probe_type, "layers": list(layers), "metrics": metrics}


def print_probe_comparison(
    label: str,
    results_by_name: dict[str, dict],
    mode: str = "relative",
) -> None:
    """Side-by-side per-layer accuracy table for several probe runs."""
    names = list(results_by_name)
    print(f"{label} — {mode} accuracy")
    print("layer | " + " | ".join(f"{n:>18}" for n in names))
    # The compared encoders may have different depths (Transformer L0..L8,
    # Mamba L0..L15). Use the union and render missing stages explicitly.
    layers = sorted({
        int(layer)
        for result in results_by_name.values()
        for layer in result["layers"]
    })
    for layer in layers:
        cells = []
        for n in names:
            metrics = results_by_name[n]["metrics"].get(str(layer), {}).get(mode)
            if metrics is None:
                cells.append(f"{'n/a':>18}")
            else:
                acc = metrics["accuracy"]
                cells.append(f"{acc:>17.2%} ")
        print(f"{layer:>5} | " + " | ".join(cells))


# ============================================================================
# Section D: head retrain on the hook path
# ============================================================================

def train_hook_path_heads(
    encoder: nn.Module,
    layer: int,
    train_chunks: Sequence[str | Path],
    device: torch.device,
    variants: dict[str, dict],
    *,
    board_size: int = 8,
    batch_size: int = 512,
    precision: str = "bf16",
    seed: int = 42,
) -> dict[str, HookPathNextMoveModel]:
    """Train several fresh next-move heads on hook-path features in one stream.

    ``variants`` maps name -> {'head_type': 'linear'|'mlp', 'lr': float,
    'input_layernorm': bool}. One encoder forward per batch feeds every
    variant, so a full LR sweep costs one pass over the chunks.
    """
    torch.manual_seed(seed)
    encoder.eval()
    d_model = int(encoder.config.d_model)
    vocab_size = int(encoder.config.vocab_size)
    block_size = int(encoder.config.block_size)

    models: dict[str, HookPathNextMoveModel] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    for name, spec in variants.items():
        head = (
            nn.Linear(d_model, vocab_size)
            if spec["head_type"] == "linear"
            else make_mlp_head(d_model, vocab_size)
        )
        norm = nn.LayerNorm(d_model) if spec.get("input_layernorm") else None
        model = HookPathNextMoveModel(
            encoder, head.to(device), layer, input_norm=(norm.to(device) if norm else None),
            precision=precision,
        )
        trainable = list(model.head.parameters()) + (
            list(model.input_norm.parameters()) if model.input_norm else []
        )
        models[name] = model
        optimizers[name] = torch.optim.AdamW(trainable, lr=spec["lr"], weight_decay=0.0)

    from torch.utils.data import DataLoader

    for chunk_path in tqdm(list(train_chunks), desc=f"hook-path heads L{layer}", unit="chunk"):
        games = load_chunk(str(chunk_path))
        dataset = OthelloChunkDataset(games=games, block_size=block_size, board_size=board_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for x_cpu, y_cpu in loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            # One shared encoder forward per batch (hook features, chosen precision).
            feats = models[next(iter(models))].extract(x)
            flat_targets = y.reshape(-1)
            valid = (flat_targets != TARGET_PAD).sum().clamp_min(1)
            for name, model in models.items():
                f = model.input_norm(feats) if model.input_norm is not None else feats
                logits = model.head(f)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), flat_targets,
                    ignore_index=TARGET_PAD, reduction="sum",
                ) / valid
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[name].step()
        del games, dataset, loader
    for model in models.values():
        model.head.eval()
        if model.input_norm is not None:
            model.input_norm.eval()
    return models


def evaluate_next_move_model_legality(
    model: nn.Module,
    val_chunks: Sequence[str | Path],
    device: torch.device,
    *,
    board_size: int = 8,
    n_games: int = 5000,
    k_values: Sequence[int] = (1, 3, 5),
    batch_size: int = 64,
) -> dict[str, float]:
    """Standard legal-move evaluation, compacted to the smoke-eval headline metrics."""
    metrics = evaluate_legal_moves(
        model=model,
        val_chunks=val_chunks,
        board_size=board_size,
        n_games=n_games,
        device=device,
        k_values=k_values,
        batch_size=batch_size,
    )
    topk = metrics["topk_legal"]
    return {
        "top1_legal": metrics["top1_legal_per_token"],
        "top3_legal_fraction": topk[3],
        "top5_legal_fraction": topk[5],
        "legal_prob_mass": metrics["legal_prob_mass_per_token"],
        "n_games": metrics["n_games"],
        "n_tokens": metrics["n_tokens"],
    }


# ============================================================================
# Section E: summary regeneration from raw eval outputs
# ============================================================================

def regenerate_summary(results_json_path: str | Path) -> str:
    """Re-emit the smoke-eval headline tables straight from the raw JSON.

    Bypasses the original notebook summary generator to rule out a
    report-writing bug behind identical head rows.
    """
    data = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
    lines = [f"# Regenerated summary: {data['metadata']['run_name']}", ""]
    lines.append("## Next-move heads (from raw metrics dicts)")
    lines.append("")
    lines.append("| Head | Top-1 legal | Top-3 legal fr. | Top-5 legal fr. | Legal mass | dict id |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for head in ("linear", "mlp"):
        m = data["next_move_legal"][head]["metrics"]
        topk = m["topk_legal"]
        mass = m["legal_prob_mass_per_token"]

        def pct(v):
            return "NaN" if v is None else f"{100 * float(v):.2f}%"

        lines.append(
            f"| {head} | {pct(m['top1_legal_per_token'])} | {pct(topk.get('3'))} | "
            f"{pct(topk.get('5'))} | {pct(mass)} | {id(m)} |"
        )
    lines.append("")
    lin = data["next_move_legal"]["linear"]["metrics"]
    mlp = data["next_move_legal"]["mlp"]["metrics"]
    lines.append(
        "- per-position top1 arrays identical: "
        f"**{lin['top1_legal_per_position'] == mlp['top1_legal_per_position']}**"
    )
    lines.append(
        "- legal_prob_mass_per_token is None (= NaN serialized): "
        f"linear={lin['legal_prob_mass_per_token'] is None}, "
        f"mlp={mlp['legal_prob_mass_per_token'] is None} "
        "(NaN probability mass => non-finite logits during the legal-move eval)"
    )
    lines.append("")
    lines.append("## Linear board probes (relative), from raw JSON")
    lines.append("")
    lines.append("| Layer | All-position | Matched |")
    lines.append("|---:|---:|---:|")
    board = data["board_state"]["linear"]["results"]["metrics"]
    matched = data["board_state"].get("matched", {}).get("linear", {}).get("metrics", {})
    for layer in data["board_state"]["linear"]["results"]["layers"]:
        rel = board[str(layer)]["relative"]["accuracy"]
        m_rel = matched.get(str(layer), {}).get("relative", {}).get("accuracy")
        m_txt = "n/a" if m_rel is None else f"{100 * m_rel:.2f}%"
        lines.append(f"| L{layer} | {100 * rel:.2f}% | {m_txt} |")
    return "\n".join(lines) + "\n"
