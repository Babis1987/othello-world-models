"""Standalone Li, Nanda, and adapted causal-intervention protocols.

The two literature methods intentionally remain separate.  Li et al. optimize
an activation through a nonlinear absolute-board probe.  Nanda et al. add one
normalized linear relative-board direction to selected residual branches.  The
historical thesis diagnostic is retained as a third, explicitly adapted method;
it is not presented as either paper's intervention.

The published constants below come from the authors' released 8x8 code.  Runs
on JEPA, Mamba, or larger boards are labelled extensions in the output rather
than literal replications of the original model experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from othello_thesis.evaluation.causal_intervention import (
    InterventionCase,
    _counterfactual_case,
    _forward_last_logits,
    _score_logits,
    _token_batch,
    evaluate_causal_intervention,
)
from othello_thesis.data.chunk_dataset import load_chunk
from othello_thesis.evaluation.position_probe import (
    PositionFeatureSet,
    PositionProbeBank,
)
from othello_thesis.evaluation.unified import (
    PROTOCOL_ID,
    FrozenEncoderNextMoveHead,
    train_frozen_next_move_head,
)
from othello_thesis.game_engine.board import OthelloBoardState


CAUSAL_SUITE_ID = "causal_intervention_suite_v1"
LI_METHOD = "li"
NANDA_METHOD = "nanda"
ADAPTED_METHOD = "adapted"
METHODS = (LI_METHOD, NANDA_METHOD, ADAPTED_METHOD)

LI_PROTOCOL: dict[str, Any] = {
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "steps": 1000,
    "preservation_weight": 0.2,
    "layers_8x8": (4, 5, 6, 7, 8),
}
NANDA_PROTOCOL: dict[str, Any] = {
    "scale": 2.3,
    "probe_resid_post_index": 5,
    "attention_blocks_8x8": (2, 3, 4, 5, 6, 7),
}

LITERATURE_SOURCES = {
    "li": {
        "paper": "https://openreview.net/forum?id=DeG07_TcZvT",
        "code": (
            "https://github.com/likenneth/othello_world/blob/master/"
            "intervening_probe_interact_column.ipynb"
        ),
        "probe": (
            "https://github.com/likenneth/othello_world/blob/master/"
            "mingpt/probe_model.py"
        ),
    },
    "nanda": {
        "paper": "https://arxiv.org/abs/2309.00941",
        "code": "https://github.com/ajyl/mech_int_othelloGPT",
    },
}


@dataclass(frozen=True)
class CausalProfile:
    name: str
    selection_cases: int
    test_cases: int
    li_steps: int
    probe_epochs: int


PROFILES = {
    # A pipeline gate only.  It intentionally does not support a thesis claim.
    "smoke": CausalProfile("smoke", 8, 16, 10, 1),
    # Published Li optimization length and the historical held-out case budget.
    "full": CausalProfile("full", 200, 1000, 1000, 5),
}


def build_flip_intervention_cases(
    chunks: Sequence[str | Path],
    *,
    board_size: int,
    n_cases: int,
    seed: int,
) -> tuple[InterventionCase, ...]:
    """Build deterministic colour-flip cases shared by all three methods."""
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
            case = _counterfactual_case(
                chunk_path=chunk_path,
                game_index=game_index,
                game=game,
                prefix_length=prefix_length,
                operation="flip",
                board_size=board_size,
                rng=rng,
            )
            if case is not None:
                cases.append(case)
            if len(cases) >= n_cases:
                return tuple(cases)
        del games
    raise ValueError(f"Built only {len(cases)} flip cases; need {n_cases}")


def causal_claim_scope(architecture: str, objective: str, board_size: int) -> str:
    """Return the strongest honest description of the selected condition."""
    architecture = str(architecture).lower()
    objective = str(objective).lower()
    if architecture == "transformer" and objective == "ar" and board_size == 8:
        return "published_method_replication"
    parts: list[str] = []
    if architecture != "transformer":
        parts.append("architecture")
    if objective != "ar":
        parts.append("objective")
    if board_size != 8:
        parts.append("board")
    return "_".join(parts) + "_extension"


def li_layer_indices(n_layers: int) -> tuple[int, ...]:
    """Map Li's post-block layers 4..8 by normalized depth."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    return tuple(
        dict.fromkeys(
            max(1, min(n_layers, round(layer * n_layers / 8)))
            for layer in LI_PROTOCOL["layers_8x8"]
        )
    )


def nanda_branch_indices(n_layers: int) -> tuple[int, ...]:
    """Map Nanda's zero-based attention blocks 2..7 by normalized depth."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    return tuple(
        dict.fromkeys(
            max(0, min(n_layers - 1, round(block * (n_layers - 1) / 7)))
            for block in NANDA_PROTOCOL["attention_blocks_8x8"]
        )
    )


def nanda_probe_layer(n_layers: int) -> int:
    """Translate official resid_post index 5 to local post-block layer L6."""
    return max(1, min(n_layers, round(6 * n_layers / 8)))


class LiAbsoluteBoardProbe(nn.Module):
    """The released Li et al. two-layer battery-probe architecture."""

    def __init__(self, d_model: int, n_squares: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_squares = int(n_squares)
        self.proj = nn.Sequential(
            nn.Linear(int(d_model), int(hidden_dim), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.n_squares * 3, bias=True),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.proj(activation.float()).reshape(-1, self.n_squares, 3)


class LiProbeBank(nn.Module):
    """Independent Li-style nonlinear absolute-board probes by layer."""

    def __init__(
        self,
        layers: Sequence[int],
        d_model: int,
        n_squares: int,
        *,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        self.absolute = nn.ModuleDict(
            {
                str(layer): LiAbsoluteBoardProbe(
                    d_model, n_squares, hidden_dim=hidden_dim
                )
                for layer in self.layers
            }
        )


def li_optimize_activation(
    probe: LiAbsoluteBoardProbe,
    activation: torch.Tensor,
    desired_labels: torch.Tensor,
    *,
    changed_squares: torch.Tensor,
    steps: int = 1000,
    learning_rate: float = 1e-3,
    preservation_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply Li's Adam-on-activation intervention to a batch of states."""
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if not 0 <= preservation_weight <= 1:
        raise ValueError("preservation_weight must lie in [0, 1]")
    edited = activation.detach().float().clone().requires_grad_(True)
    desired_labels = desired_labels.to(edited.device, dtype=torch.long)
    changed_squares = changed_squares.to(edited.device, dtype=torch.long)
    if desired_labels.shape != (edited.shape[0], probe.n_squares):
        raise ValueError("desired_labels shape does not match activation batch")
    weights = torch.full(
        desired_labels.shape,
        float(preservation_weight),
        device=edited.device,
        dtype=torch.float32,
    )
    weights[torch.arange(edited.shape[0], device=edited.device), changed_squares] = 1.0
    optimizer = torch.optim.Adam([edited], lr=float(learning_rate))
    initial_loss: float | None = None
    final_loss = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(edited)
        losses = F.cross_entropy(
            logits.reshape(-1, 3),
            desired_labels.reshape(-1),
            reduction="none",
        ).reshape_as(weights)
        loss = (weights * losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Li activation optimization produced non-finite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    if not torch.isfinite(edited).all():
        raise RuntimeError("Li activation optimization produced non-finite state")
    return edited.detach(), {
        "steps": int(steps),
        "initial_weighted_loss": float(initial_loss or 0.0),
        "final_weighted_loss": final_loss,
    }


def nanda_target_directions(
    probe_bank: PositionProbeBank,
    cases: Sequence[InterventionCase],
    *,
    layer: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return Nanda's normalized target-class vector (not a class difference)."""
    probe = probe_bank.relative[str(layer)]
    if not isinstance(probe.input_norm, nn.Identity):
        raise TypeError("Nanda intervention requires an unnormalized probe")
    if not isinstance(probe.proj, nn.Linear):
        raise TypeError("Nanda intervention requires a linear probe")
    d_model = int(probe.proj.weight.shape[1])
    weights = probe.proj.weight.view(probe.n_squares, 3, d_model)
    directions = torch.stack(
        [weights[case.square, case.target_class] for case in cases]
    ).to(device=device, dtype=torch.float32)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _apply_last_position_edit(
    branch: torch.Tensor,
    *,
    rows: torch.Tensor,
    last_indices: torch.Tensor,
    directions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    edited = branch.clone()
    selected = edited[rows, last_indices].float()
    edited[rows, last_indices] = (selected + float(scale) * directions).to(branch.dtype)
    return edited


def _forward_nanda_logits(
    encoder: nn.Module,
    readout: nn.Module,
    probe_bank: PositionProbeBank,
    cases: Sequence[InterventionCase],
    *,
    board_size: int,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    tokens, last_indices = _token_batch(cases, board_size=board_size, device=device)
    rows = torch.arange(tokens.shape[0], device=device)
    positions = torch.arange(tokens.shape[1], device=device)
    patch_blocks = set(nanda_branch_indices(len(encoder.blocks)))
    probe_layer = nanda_probe_layer(len(encoder.blocks))
    directions = nanda_target_directions(
        probe_bank, cases, layer=probe_layer, device=device
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        hidden = encoder.drop(encoder.wte(tokens) + encoder.wpe(positions))
        for block_index, block in enumerate(encoder.blocks):
            if hasattr(block, "attn"):
                branch = block.attn(block.ln_1(hidden))
                if block_index in patch_blocks:
                    branch = _apply_last_position_edit(
                        branch,
                        rows=rows,
                        last_indices=last_indices,
                        directions=directions,
                        scale=scale,
                    )
                hidden = hidden + branch
                hidden = hidden + block.mlp(block.ln_2(hidden))
            elif hasattr(block, "mixer"):
                branch = block.resid_dropout(block._mix(block.ln_1(hidden)))
                if block_index in patch_blocks:
                    branch = _apply_last_position_edit(
                        branch,
                        rows=rows,
                        last_indices=last_indices,
                        directions=directions,
                        scale=scale,
                    )
                hidden = hidden + branch
                if block.use_mlp:
                    hidden = hidden + block.mlp(block.ln_2(hidden))
            else:
                raise TypeError(
                    f"Unsupported residual block for Nanda extension: {type(block)!r}"
                )
        final_hidden = encoder.ln_f(hidden)[rows, last_indices].float()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        logits = readout(final_hidden.float())
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nanda intervention produced non-finite logits")
    return logits.float()


def _forward_unedited_logits(
    encoder: nn.Module,
    readout: nn.Module,
    cases: Sequence[InterventionCase],
    *,
    board_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the shared frozen model without requiring either probe family."""
    tokens, last_indices = _token_batch(cases, board_size=board_size, device=device)
    rows = torch.arange(tokens.shape[0], device=device)
    positions = torch.arange(tokens.shape[1], device=device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        hidden = encoder.drop(encoder.wte(tokens) + encoder.wpe(positions))
        for block in encoder.blocks:
            hidden = block(hidden)
        final_hidden = encoder.ln_f(hidden)[rows, last_indices].float()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        logits = readout(final_hidden.float())
    if not torch.isfinite(logits).all():
        raise RuntimeError("Unedited causal baseline produced non-finite logits")
    return logits.float()


def _absolute_target_labels(
    cases: Sequence[InterventionCase], *, board_size: int, device: torch.device
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for case in cases:
        board = OthelloBoardState(board_size)
        board.update(case.prefix_raw)
        state = np.asarray(board.get_state(), dtype=np.int8).reshape(-1).copy()
        state[case.square] *= -1
        # Position-probe contract: black=0, white=1, empty=2.
        labels = np.where(state == 1, 0, np.where(state == -1, 1, 2))
        rows.append(torch.from_numpy(labels.astype(np.int64, copy=False)))
    return torch.stack(rows).to(device)


def _forward_li_logits(
    encoder: nn.Module,
    readout: nn.Module,
    probes: LiProbeBank,
    cases: Sequence[InterventionCase],
    *,
    board_size: int,
    device: torch.device,
    steps: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    tokens, last_indices = _token_batch(cases, board_size=board_size, device=device)
    rows = torch.arange(tokens.shape[0], device=device)
    positions = torch.arange(tokens.shape[1], device=device)
    layers = li_layer_indices(len(encoder.blocks))
    changed = torch.tensor([case.square for case in cases], device=device)
    target_labels = _absolute_target_labels(cases, board_size=board_size, device=device)
    diagnostics: list[dict[str, Any]] = []
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        hidden = encoder.drop(encoder.wte(tokens) + encoder.wpe(positions))

    completed_layers = 0
    for layer, block in enumerate(encoder.blocks, start=1):
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            hidden = block(hidden)
        if layer not in layers:
            continue
        selected = hidden[rows, last_indices].float()
        probe = probes.absolute[str(layer)]
        with torch.no_grad():
            desired = probe(selected).argmax(dim=-1)
            desired[rows, changed] = target_labels[rows, changed]
        optimized, layer_diagnostics = li_optimize_activation(
            probe,
            selected,
            desired,
            changed_squares=changed,
            steps=steps,
            learning_rate=float(LI_PROTOCOL["learning_rate"]),
            preservation_weight=float(LI_PROTOCOL["preservation_weight"]),
        )
        hidden = hidden.clone()
        hidden[rows, last_indices] = optimized.to(hidden.dtype)
        completed_layers += 1
        diagnostics.append({"layer": layer, **layer_diagnostics})

    if completed_layers != len(layers):
        raise RuntimeError("Li intervention did not visit every configured layer")
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        final_hidden = encoder.ln_f(hidden)[rows, last_indices].float()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        logits = readout(final_hidden.float())
    if not torch.isfinite(logits).all():
        raise RuntimeError("Li intervention produced non-finite logits")
    return logits.float(), diagnostics


def _evaluate_li_probe_accuracy(
    bank: LiProbeBank,
    features: PositionFeatureSet,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    correct = 0
    total = 0
    bank.eval()
    with torch.inference_mode():
        for start in range(0, features.n_samples, batch_size):
            stop = min(features.n_samples, start + batch_size)
            labels = features.absolute_labels[start:stop].to(device)
            for layer in bank.layers:
                logits = bank.absolute[str(layer)](
                    features.features[layer][start:stop].to(device)
                )
                correct += int((logits.argmax(dim=-1) == labels).sum())
                total += int(labels.numel())
    return correct / total


def train_li_probe_bank(
    train_features: PositionFeatureSet,
    validation_features: PositionFeatureSet,
    test_features: PositionFeatureSet,
    *,
    layers: Sequence[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[LiProbeBank, dict[str, Any]]:
    """Train the released Li probe shape on the thesis fixed split."""
    torch.manual_seed(seed)
    bank = LiProbeBank(
        layers,
        train_features.d_model,
        train_features.n_squares,
        hidden_dim=int(LI_PROTOCOL["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=1e-3)
    best_accuracy = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        bank.train()
        generator = torch.Generator().manual_seed(seed + epoch - 1)
        order = torch.randperm(train_features.n_samples, generator=generator)
        for start in range(0, train_features.n_samples, batch_size):
            indices = order[start : start + batch_size]
            labels = train_features.absolute_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for layer in bank.layers:
                logits = bank.absolute[str(layer)](
                    train_features.features[layer][indices].to(device)
                )
                losses.append(
                    F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
                )
            loss = torch.stack(losses).sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Li probe training produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        validation_accuracy = _evaluate_li_probe_accuracy(
            bank, validation_features, device=device, batch_size=batch_size
        )
        history.append({"epoch": epoch, "validation_accuracy": validation_accuracy})
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in bank.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Li probe training did not produce a checkpoint")
    bank.load_state_dict(best_state)
    bank.eval()
    test_accuracy = _evaluate_li_probe_accuracy(
        bank, test_features, device=device, batch_size=batch_size
    )
    return bank, {
        "architecture": "Linear-ReLU-Linear",
        "hidden_dim": int(LI_PROTOCOL["hidden_dim"]),
        "layers": list(bank.layers),
        "epochs": epochs,
        "history": history,
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "training_scope": "thesis fixed split; probe shape follows Li et al.",
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class CausalInterventionSuite:
    """Resumable external-artifact runner for the three causal methods."""

    def __init__(
        self,
        prepared: Any,
        *,
        profile: str,
        methods: Sequence[str] = METHODS,
        force: bool = False,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {tuple(PROFILES)}, got {profile!r}")
        unknown = set(methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown causal methods: {sorted(unknown)}")
        self.prepared = prepared
        self.evaluator = prepared.evaluator
        self.profile = PROFILES[profile]
        self.methods = tuple(dict.fromkeys(methods))
        self.force = bool(force)
        self.device = self.evaluator.device
        self.output_dir = Path(prepared.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / f"results__{CAUSAL_SUITE_ID}.json"
        condition_scope = causal_claim_scope(
            prepared.case.architecture,
            prepared.case.objective,
            prepared.case.board_size,
        )
        identity = {
            "suite": CAUSAL_SUITE_ID,
            "profile": profile,
            "architecture": prepared.case.architecture,
            "objective": prepared.case.objective,
            "board_size": prepared.case.board_size,
            "checkpoint_sha256": self.evaluator.results["metadata"][
                "checkpoint_sha256"
            ],
            "split_manifest_hash": self.evaluator.split.manifest_hash,
            "condition_scope": condition_scope,
            "claim_scope": (
                "pipeline_smoke_no_thesis_claim"
                if profile == "smoke"
                else condition_scope
            ),
        }
        if self.state_path.is_file() and not force:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("identity") != identity:
                raise RuntimeError(
                    "Existing causal state belongs to a different checkpoint, split, "
                    "condition, or profile. Use a new output path or --force."
                )
            self.state = state
        else:
            self.state = {
                "schema_version": CAUSAL_SUITE_ID,
                "identity": identity,
                "literature_sources": LITERATURE_SOURCES,
                "published_constants": {
                    "li": LI_PROTOCOL,
                    "nanda": NANDA_PROTOCOL,
                },
                "methods": {},
                "timing_seconds": {},
            }
            _atomic_json(self.state_path, self.state)
        self._readout_cache: tuple[nn.Module, str] | None = None
        self._raw_probe_cache: PositionProbeBank | None = None
        self._li_probe_cache: LiProbeBank | None = None

    def _save(self) -> None:
        _atomic_json(self.state_path, self.state)

    def _cases(self) -> tuple[tuple[InterventionCase, ...], tuple[InterventionCase, ...]]:
        selection = build_flip_intervention_cases(
            self.evaluator.split.selection,
            board_size=self.prepared.case.board_size,
            n_cases=self.profile.selection_cases,
            seed=self.evaluator.config.seed + 50_000,
        )
        test = build_flip_intervention_cases(
            self.evaluator.split.test,
            board_size=self.prepared.case.board_size,
            n_cases=self.profile.test_cases,
            seed=self.evaluator.config.seed + 60_000,
        )
        manifest = {
            "operation": "occupied-square colour flip",
            "selection": [case.to_dict() for case in selection],
            "test": [case.to_dict() for case in test],
        }
        _atomic_json(
            self.output_dir / f"case_manifest__{CAUSAL_SUITE_ID}.json", manifest
        )
        return selection, test

    def _readout(self) -> tuple[nn.Module, str]:
        if self._readout_cache is not None:
            return self._readout_cache
        if self.prepared.case.objective == "ar":
            readout = self.prepared.primary_model.lm_head
            readout.eval()
            self._readout_cache = (readout, "native_ar")
            return self._readout_cache

        common_dir = self.prepared.run_dir / "thesis_eval" / "final"
        dedicated_path = self.output_dir / f"mlp_next_move_head__{PROTOCOL_ID}.pt"
        candidates = (
            dedicated_path,
            common_dir / f"mlp_next_move_head__{PROTOCOL_ID}.pt",
        )
        artifact = next((path for path in candidates if path.is_file()), None)
        if artifact is not None:
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            saved_split = payload.get("split_manifest_hash")
            if saved_split not in (None, self.evaluator.split.manifest_hash):
                raise RuntimeError("Frozen JEPA readout has a different split identity")
            saved_checkpoint = (
                payload.get("checkpoint_sha256")
                or (payload.get("run_metadata") or {}).get("checkpoint_sha256")
            )
            current_checkpoint = self.evaluator.results["metadata"][
                "checkpoint_sha256"
            ]
            if saved_checkpoint not in (None, current_checkpoint):
                raise RuntimeError("Frozen JEPA readout has a different checkpoint identity")
            selected = (payload.get("tuning") or {}).get("selected") or {}
            wrapper = FrozenEncoderNextMoveHead(
                self.evaluator.encoder,
                "mlp",
                hidden_dim=int(
                    selected.get("hidden_dim", self.evaluator.config.mlp_hidden_dim)
                ),
                dropout=float(
                    selected.get("dropout", self.evaluator.config.mlp_dropout)
                ),
                encoder_precision=self.evaluator.config.encoder_precision,
            ).to(self.device)
            wrapper.head.load_state_dict(payload["head"])
            label = "reused_common_frozen_mlp" if artifact.parent == common_dir else "dedicated_frozen_mlp"
        else:
            wrapper, train_result = train_frozen_next_move_head(
                self.evaluator.encoder,
                self.evaluator.split.head_training_pool(
                    self.evaluator.config.head_max_shards
                ),
                selection_chunks=self.evaluator.split.selection,
                head_type="mlp",
                config=self.evaluator.config,
                device=self.device,
            )
            torch.save(
                {
                    "schema_version": CAUSAL_SUITE_ID,
                    "head_type": "mlp",
                    "head": wrapper.head.state_dict(),
                    "split_manifest_hash": self.evaluator.split.manifest_hash,
                    "checkpoint_sha256": self.evaluator.results["metadata"][
                        "checkpoint_sha256"
                    ],
                    "train": train_result,
                    "purpose": "causal suite readout only; no legal-eval stage",
                },
                dedicated_path,
            )
            label = "trained_causal_only_frozen_mlp"
        wrapper.eval()
        self._readout_cache = (wrapper.head, label)
        return self._readout_cache

    def _raw_probe_bank(self) -> PositionProbeBank:
        if self._raw_probe_cache is not None:
            return self._raw_probe_cache
        common_path = (
            self.prepared.run_dir
            / "thesis_eval"
            / "final"
            / f"raw_linear_intervention_probe__{PROTOCOL_ID}.pt"
        )
        if common_path.is_file():
            common_results_path = (
                self.prepared.run_dir
                / "thesis_eval"
                / "final"
                / f"results__{PROTOCOL_ID}.json"
            )
            if not common_results_path.is_file():
                raise RuntimeError("Common intervention probe lacks its results identity")
            common_results = json.loads(
                common_results_path.read_text(encoding="utf-8")
            )
            if common_results.get("metadata", {}).get("checkpoint_sha256") != (
                self.evaluator.results["metadata"]["checkpoint_sha256"]
            ):
                raise RuntimeError("Common intervention probe has a different checkpoint")
            if common_results.get("metadata", {}).get("split_manifest_hash") != (
                self.evaluator.split.manifest_hash
            ):
                raise RuntimeError("Common intervention probe has a different split")
            layers = tuple(range(len(self.evaluator.encoder.blocks) + 1))
            bank = PositionProbeBank(
                layers,
                int(self.evaluator.encoder.config.d_model),
                self.prepared.case.board_size**2,
                probe_type="linear",
                hidden_dim=self.evaluator.config.mlp_hidden_dim,
                dropout=0.0,
                input_layernorm=False,
            ).to(self.device)
            payload = torch.load(common_path, map_location="cpu", weights_only=False)
            bank.load_state_dict(payload["probe_bank"])
            bank.eval()
        else:
            bank, _ = self.evaluator._intervention_probe_bank()
        self._raw_probe_cache = bank
        return bank

    def _li_probe_bank(self) -> LiProbeBank:
        if self._li_probe_cache is not None:
            return self._li_probe_cache
        path = self.output_dir / f"li_absolute_probe__{CAUSAL_SUITE_ID}.pt"
        layers = li_layer_indices(len(self.evaluator.encoder.blocks))
        bank = LiProbeBank(
            layers,
            int(self.evaluator.encoder.config.d_model),
            self.prepared.case.board_size**2,
            hidden_dim=int(LI_PROTOCOL["hidden_dim"]),
        ).to(self.device)
        if path.is_file() and not self.force:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            bank.load_state_dict(payload["probe_bank"])
        else:
            train, validation, test = self.evaluator._prepare_position_features()
            bank, result = train_li_probe_bank(
                train,
                validation,
                test,
                layers=layers,
                device=self.device,
                epochs=self.profile.probe_epochs,
                batch_size=self.evaluator.config.board_probe_batch_size,
                seed=self.evaluator.config.seed + 80_000,
            )
            torch.save(
                {
                    "schema_version": CAUSAL_SUITE_ID,
                    "probe_bank": bank.state_dict(),
                    "result": result,
                    "checkpoint_sha256": self.evaluator.results["metadata"][
                        "checkpoint_sha256"
                    ],
                    "split_manifest_hash": self.evaluator.split.manifest_hash,
                },
                path,
            )
        bank.eval()
        self._li_probe_cache = bank
        return bank

    def _run_nanda(
        self, test_cases: Sequence[InterventionCase]
    ) -> dict[str, Any]:
        readout, readout_label = self._readout()
        probes = self._raw_probe_bank()
        null_logits: list[torch.Tensor] = []
        patched_logits: list[torch.Tensor] = []
        batch_size = self.evaluator.config.intervention_batch_size
        for start in range(0, len(test_cases), batch_size):
            batch = test_cases[start : start + batch_size]
            null_logits.append(
                _forward_unedited_logits(
                    self.evaluator.encoder,
                    readout,
                    batch,
                    board_size=self.prepared.case.board_size,
                    device=self.device,
                ).cpu()
            )
            patched_logits.append(
                _forward_nanda_logits(
                    self.evaluator.encoder,
                    readout,
                    probes,
                    batch,
                    board_size=self.prepared.case.board_size,
                    device=self.device,
                    scale=float(NANDA_PROTOCOL["scale"]),
                ).cpu()
            )
        return {
            "protocol": "nanda_target_class_attention_branch_v1",
            "published_constants": NANDA_PROTOCOL,
            "local_probe_layer": nanda_probe_layer(len(self.evaluator.encoder.blocks)),
            "local_branch_indices": list(
                nanda_branch_indices(len(self.evaluator.encoder.blocks))
            ),
            "branch_surface": (
                "attention output"
                if self.prepared.case.architecture == "transformer"
                else "Mamba mixer residual-branch output (architecture extension)"
            ),
            "readout": readout_label,
            "test": {
                "null": _score_logits(torch.cat(null_logits), test_cases),
                "target_class_direction": _score_logits(
                    torch.cat(patched_logits), test_cases
                ),
            },
        }

    def _run_adapted(
        self,
        selection_cases: Sequence[InterventionCase],
        test_cases: Sequence[InterventionCase],
    ) -> dict[str, Any]:
        readout, readout_label = self._readout()
        result = evaluate_causal_intervention(
            self.evaluator.encoder,
            readout,
            self._raw_probe_bank(),
            selection_cases=selection_cases,
            test_cases=test_cases,
            board_size=self.prepared.case.board_size,
            device=self.device,
            alpha_values=self.evaluator.config.intervention_alpha_values,
            batch_size=self.evaluator.config.intervention_batch_size,
            seed=self.evaluator.config.seed + 70_000,
        )
        result["protocol"] = "adapted_linear_sequential_residual_v1"
        result["lineage"] = (
            "thesis-specific hybrid: target-minus-source relative linear direction, "
            "residual-std scaling, every post-block residual, held-out alpha selection, "
            "and magnitude-matched random control"
        )
        result["readout"] = readout_label
        return result

    def _run_li(self, test_cases: Sequence[InterventionCase]) -> dict[str, Any]:
        readout, readout_label = self._readout()
        probes = self._li_probe_bank()
        null_logits: list[torch.Tensor] = []
        edited_logits: list[torch.Tensor] = []
        diagnostics: list[dict[str, Any]] = []
        batch_size = self.evaluator.config.intervention_batch_size
        progress_dir = self.output_dir / "li_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        total_batches = math.ceil(len(test_cases) / batch_size)
        for batch_number, start in enumerate(
            range(0, len(test_cases), batch_size), start=1
        ):
            batch = test_cases[start : start + batch_size]
            case_ids = [case.case_id for case in batch]
            path = progress_dir / f"batch_{start:06d}.pt"
            if path.is_file() and not self.force:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("case_ids") != case_ids or payload.get("steps") != self.profile.li_steps:
                    raise RuntimeError(f"Incompatible Li progress batch: {path}")
                null_logits.append(payload["null_logits"])
                edited_logits.append(payload["edited_logits"])
                diagnostics.extend(payload["diagnostics"])
                continue
            started = time.perf_counter()
            null = _forward_unedited_logits(
                self.evaluator.encoder,
                readout,
                batch,
                board_size=self.prepared.case.board_size,
                device=self.device,
            ).cpu()
            edited, batch_diagnostics = _forward_li_logits(
                self.evaluator.encoder,
                readout,
                probes,
                batch,
                board_size=self.prepared.case.board_size,
                device=self.device,
                steps=self.profile.li_steps,
            )
            elapsed = time.perf_counter() - started
            payload = {
                "case_ids": case_ids,
                "steps": self.profile.li_steps,
                "null_logits": null,
                "edited_logits": edited.cpu(),
                "diagnostics": batch_diagnostics,
                "seconds": elapsed,
            }
            torch.save(payload, path)
            null_logits.append(null)
            edited_logits.append(edited.cpu())
            diagnostics.extend(batch_diagnostics)
            print(
                f"Li batch {batch_number}/{total_batches}: {elapsed:.1f}s; "
                f"projected method time {elapsed * total_batches / 60:.1f} min",
                flush=True,
            )
        return {
            "protocol": "li_nonlinear_activation_optimization_v1",
            "published_constants": LI_PROTOCOL,
            "executed_steps": self.profile.li_steps,
            "layers": list(probes.layers),
            "readout": readout_label,
            "test": {
                "null": _score_logits(torch.cat(null_logits), test_cases),
                "optimized_activation": _score_logits(
                    torch.cat(edited_logits), test_cases
                ),
            },
            "optimization_diagnostics": diagnostics,
            "resume_granularity": "case batch",
        }

    def write_summary(self) -> Path:
        identity = self.state["identity"]
        lines = [
            "# Causal-intervention comparison",
            "",
            f"- Condition: `{identity['architecture']}` / `{identity['objective']}` / "
            f"`{identity['board_size']}x{identity['board_size']}`",
            f"- Profile: `{identity['profile']}`",
            f"- Claim scope: `{identity['claim_scope']}`",
            "- Shared counterfactuals: occupied-square colour flips that change the "
            "legal-move set.",
            "",
            "| Method | Published operation | Local intervention | Top-N FP+FN | Target legal mass |",
            "|---|---|---|---:|---:|",
        ]
        result_keys = {
            LI_METHOD: ("optimized_activation", "Adam optimization through nonlinear absolute probe"),
            NANDA_METHOD: ("target_class_direction", "fixed normalized linear direction in residual branch"),
            ADAPTED_METHOD: ("probe_direction", "target-minus-source edits after every block"),
        }
        for method in METHODS:
            result = self.state["methods"].get(method)
            if not result:
                continue
            key, description = result_keys[method]
            metrics = result["test"][key]
            lines.append(
                f"| {method} | {description} | `{result['protocol']}` | "
                f"{metrics['mean_topn_false_positive_plus_false_negative']:.3f} | "
                f"{metrics['target_legal_probability_mass']:.4f} |"
            )
        lines += [
            "",
            "The 8x8 Transformer-AR condition follows the released intervention "
            "algorithms and constants, but uses this thesis checkpoint, corpus, and "
            "probe split. JEPA uses one fixed post-hoc readout for all methods; Mamba "
            "patches the mixer branch corresponding to Nanda's attention branch. "
            "Those conditions, and larger boards, are extensions.",
            "",
            "`smoke` is only an execution gate. Thesis conclusions require `full`.",
        ]
        path = self.output_dir / f"summary__{CAUSAL_SUITE_ID}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def run(self) -> dict[str, str]:
        selection_cases, test_cases = self._cases()
        # Keep the expensive Li optimizer last; the two cheap diagnostics are
        # durably available even if a Colab allocation ends during Li.
        execution_order = (NANDA_METHOD, ADAPTED_METHOD, LI_METHOD)
        for method in execution_order:
            if method not in self.methods:
                continue
            if method in self.state["methods"] and not self.force:
                print(f"{method} stage already complete; reusing result", flush=True)
                continue
            started = time.perf_counter()
            if method == NANDA_METHOD:
                result = self._run_nanda(test_cases)
            elif method == ADAPTED_METHOD:
                result = self._run_adapted(selection_cases, test_cases)
            else:
                result = self._run_li(test_cases)
            result["claim_scope"] = self.state["identity"]["claim_scope"]
            self.state["methods"][method] = result
            self.state["timing_seconds"][method] = time.perf_counter() - started
            self._save()
        summary = self.write_summary()
        return {"results": str(self.state_path), "summary": str(summary)}


__all__ = [
    "ADAPTED_METHOD",
    "CAUSAL_SUITE_ID",
    "CausalInterventionSuite",
    "LI_METHOD",
    "LI_PROTOCOL",
    "LiAbsoluteBoardProbe",
    "LiProbeBank",
    "METHODS",
    "NANDA_METHOD",
    "NANDA_PROTOCOL",
    "PROFILES",
    "causal_claim_scope",
    "build_flip_intervention_cases",
    "li_layer_indices",
    "li_optimize_activation",
    "nanda_branch_indices",
    "nanda_probe_layer",
    "nanda_target_directions",
    "train_li_probe_bank",
]
