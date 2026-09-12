"""Standalone Li, Nanda, and adapted causal-intervention protocols.

The two literature methods intentionally remain separate.  Li et al. optimize
an activation through a nonlinear absolute-board probe.  Nanda et al. add one
normalized linear relative-board direction to selected residual branches.  The
historical thesis diagnostic is retained as a third, explicitly adapted method;
it is not presented as either paper's intervention.

The published constants below come from the authors' released 8x8 code.  Runs
on JEPA, Mamba, or larger boards are labelled extensions in the output rather
than literal replications of the original model experiment.

Why three methods instead of one
--------------------------------
A probe shows that board information is *decodable* from the residual stream.
It does not show that the model uses it. The standard test for use is
intervention: edit the activation so the decoded board says a square has
flipped colour, then ask whether the model's move distribution changes in the
way the flipped board implies. If it does, the representation is load-bearing;
if legality predictions ignore the edit, the probe was reading a correlate.

The trouble is that "edit the activation" is underdetermined, and the two
published protocols make opposite choices:

- **Li**: run gradient descent *on the activation itself* through a frozen
  nonlinear probe until the probe reports the counterfactual board, with a
  penalty holding the activation near its original value. Strong -- it can
  reach board states no single direction points to -- but the edit is whatever
  the optimiser finds, which need not resemble anything the model produces.
- **Nanda**: add one fixed, unit-normalised linear direction, scaled by a
  constant. Weak by construction, and precisely for that reason a much cleaner
  causal claim: if a single linear direction suffices, the feature is linearly
  represented and the model reads it linearly.

They can disagree, and the disagreement is informative rather than a defect:
Li succeeding where Nanda fails is the signature of board information that is
present but not linearly readable. Reporting only one method would silently
pick a side. The adapted method is the thesis's own variant and is labelled as
such everywhere it appears.

Transfer beyond 8x8
-------------------
Every published constant -- which layers to edit, which scale to use, which
layer to probe -- was tuned on one 8-block Transformer trained
autoregressively on 8x8 Othello. None of it transfers by construction to
Mamba's 15 blocks, to JEPA-trained encoders, or to 12x12 and 16x16 boards.
Layer indices are therefore mapped by *normalised depth* rather than reused
verbatim (see ``li_layer_indices`` and ``nanda_branch_indices``), and
``causal_claim_scope`` stamps every result with how far it has travelled from
the setting the constants were derived in. Only one cell of the grid --
Transformer / AR / 8x8 -- is entitled to be called a replication.
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
)
from othello_thesis.game_engine.board import OthelloBoardState


CAUSAL_SUITE_ID = "causal_intervention_suite_v2"
ADAPTED_PROTOCOL_ID = "adapted_linear_sequential_residual_v2"
ADAPTED_RANDOM_CONTROL_ID = "shared_signed_permutation_v1"
LI_METHOD = "li"
NANDA_METHOD = "nanda"
ADAPTED_METHOD = "adapted"
METHODS = (LI_METHOD, NANDA_METHOD, ADAPTED_METHOD)
JEPA_READOUT_TYPES = ("linear", "mlp")
JEPA_READOUT_POLICIES = ("best", *JEPA_READOUT_TYPES, "both")
READOUT_SELECTION_METRIC = "best_selection_legal_probability_mass"

# Constants transcribed from the authors' released 8x8 code, kept in one place
# so any deviation from the published setting is visible rather than buried in
# a call site. They are not tuned here and must not be tuned per condition --
# retuning them per model would turn a replication into a search over
# interventions and make the cross-model comparison meaningless.
LI_PROTOCOL: dict[str, Any] = {
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    # 1000 Adam steps per case is what makes the Li method expensive; it is the
    # published length and is only shortened in the smoke profile.
    "steps": 1000,
    # Weight on the term holding the edited activation near the original. Too
    # low and the optimiser is free to leave the manifold the model ever sees;
    # too high and the probe never reports the counterfactual board.
    "preservation_weight": 0.2,
    "layers_8x8": (4, 5, 6, 7, 8),
}
NANDA_PROTOCOL: dict[str, Any] = {
    # Edit magnitude in units of the unit-normalised probe direction. Fixed,
    # not swept: a swept scale would let the method pick its own best case per
    # model and inflate the apparent causal effect.
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
    """Compute budget for one suite run.

    The split between ``selection_cases`` and ``test_cases`` is the protocol,
    not an implementation detail. Anything chosen by looking at data -- which
    JEPA readout head to use, which alpha the adapted method runs at -- is
    chosen on the selection cases; the reported numbers come from the disjoint
    test cases. Collapsing the two would mean tuning the intervention on the
    same cases it is scored on.
    """

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


def resolve_causal_readout_runs(objective: str, policy: str) -> tuple[str, ...]:
    """Expand the notebook/CLI policy into isolated causal-suite runs.

    ``both`` becomes two complete, separate suites rather than one suite
    carrying two heads. Keeping them isolated means each run's artifacts,
    resume state and summary describe exactly one readout -- there is no way to
    end up quoting a Li result obtained with the linear head next to a Nanda
    result obtained with the MLP head.

    AR models ignore the policy entirely: they have their own head.
    """
    objective = str(objective).lower()
    policy = str(policy).lower()
    if objective not in {"ar", "jepa"}:
        raise ValueError("objective must be 'ar' or 'jepa'")
    if policy not in JEPA_READOUT_POLICIES:
        raise ValueError(
            f"JEPA readout policy must be one of {JEPA_READOUT_POLICIES}"
        )
    if objective == "ar":
        return ("native",)
    if policy == "both":
        return JEPA_READOUT_TYPES
    return (policy,)


def causal_output_subdir(profile: str, readout_run: str) -> Path:
    """Return the v2 output root, isolated by requested readout policy.

    Suite id, profile and readout each get their own path segment so runs that
    differ in any of them cannot overwrite one another's artifacts -- in
    particular, a smoke run and a full run of the same condition stay separate
    on disk, which is what keeps a smoke result from being mistaken for a real
    one later.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {tuple(PROFILES)}")
    if readout_run not in {"native", "best", *JEPA_READOUT_TYPES}:
        raise ValueError(f"Unknown causal readout run: {readout_run!r}")
    return (
        Path("causal_intervention")
        / CAUSAL_SUITE_ID
        / profile
        / f"readout_{readout_run}"
    )


def select_jepa_readout(
    common_results: Mapping[str, Any], policy: str
) -> dict[str, Any]:
    """Select a JEPA readout using selection data only, never final-test data.

    JEPA encoders have no next-move head of their own -- the objective never
    asks for one -- so measuring legality at all requires attaching a head to
    the frozen encoder. That creates a comparability problem the AR models do
    not have: the choice of head (linear or MLP) changes the measured legality,
    and picking the head that scores best on the test set would credit the
    encoder for the head's capacity.

    The rule here is that the head is chosen on selection data and then held
    fixed. ``both`` sidesteps the choice entirely by running the whole suite
    twice and reporting both, which is the honest option when the two heads
    disagree.
    """
    policy = str(policy).lower()
    if policy not in {"best", *JEPA_READOUT_TYPES}:
        raise ValueError("Suite-level JEPA readout must be 'best', 'linear', or 'mlp'")
    frozen = common_results.get("frozen_next_move") or {}
    scores: dict[str, float] = {}
    for head_type in JEPA_READOUT_TYPES:
        value = (
            (frozen.get(head_type) or {})
            .get("train", {})
            .get(READOUT_SELECTION_METRIC)
        )
        if value is None:
            if policy == "best" or policy == head_type:
                raise RuntimeError(
                    f"Missing {head_type} selection result {READOUT_SELECTION_METRIC!r}; "
                    "complete the common frozen-head evaluation first."
                )
            continue
        score = float(value)
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite {head_type} readout selection score")
        scores[head_type] = score

    if policy == "best":
        # The simpler Linear readout wins an exact tie.  Final-test metrics are
        # deliberately absent from this decision.
        selected = max(
            JEPA_READOUT_TYPES,
            key=lambda head_type: (scores[head_type], head_type == "linear"),
        )
    else:
        selected = policy
    return {
        "requested_policy": policy,
        "selected": selected,
        "selection_split": "selection",
        "selection_metric": READOUT_SELECTION_METRIC,
        "selection_scores": scores,
        "tie_break": "linear",
    }


def build_flip_intervention_cases(
    chunks: Sequence[str | Path],
    *,
    board_size: int,
    n_cases: int,
    seed: int,
) -> tuple[InterventionCase, ...]:
    """Build deterministic colour-flip cases shared by all three methods.

    All three methods must be scored on exactly the same cases, otherwise the
    comparison between them measures case difficulty as much as intervention
    strength. The cases are therefore built once, here, and passed to each
    method.

    Determinism is per case, not per run: the RNG for each (chunk, game) is
    seeded from a hash of the seed and its position in the corpus, so case
    ``k`` is identical whether the suite asks for 16 cases or 1000, and adding
    a chunk does not shift every case that follows it. A single sequential RNG
    would break both properties and make two runs at different budgets
    incomparable.

    A prefix must be long enough that the board has structure to flip and
    short enough that at least one move follows it, hence the length floor and
    the ``len(game) - 2`` bound. Cases where no legal flip exists are dropped
    by ``_counterfactual_case``, which is why the loop cannot simply take the
    first ``n_cases`` games.
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
    """Return the strongest honest description of the selected condition.

    Stamped into every result file so a number can never be quoted as a
    replication of Li or Nanda when it was produced in a setting their
    constants were not derived for. Exactly one cell of the experimental grid
    -- Transformer, AR, 8x8 -- matches the published setting; everything else
    names the axes along which it departs (``objective_board_extension`` and so
    on). The label is deliberately verbose rather than a boolean, because
    "extension along which axis" is what a reader needs to weigh the result.
    """
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
    """Map Li's post-block layers 4..8 by normalized depth.

    Li edits layers 4-8 of an 8-block model, i.e. the second half of the
    network. Reusing the literal indices on Mamba's 15 blocks would edit its
    middle rather than its second half, so what transfers is the *fractional*
    position: layer k of 8 becomes round(k * n_layers / 8).

    ``dict.fromkeys`` deduplicates while preserving order -- on a shallow model
    two source layers can round to the same target, and editing the same layer
    twice in one pass would double the intervention there.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    return tuple(
        dict.fromkeys(
            max(1, min(n_layers, round(layer * n_layers / 8)))
            for layer in LI_PROTOCOL["layers_8x8"]
        )
    )


def nanda_branch_indices(n_layers: int) -> tuple[int, ...]:
    """Map Nanda's zero-based attention blocks 2..7 by normalized depth.

    Same normalised-depth argument as ``li_layer_indices``, with a different
    index base: Nanda counts attention blocks from zero, so the denominator is
    ``n_layers - 1`` rather than ``n_layers``. Mixing the two conventions would
    shift every edit by one block.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    return tuple(
        dict.fromkeys(
            max(0, min(n_layers - 1, round(block * (n_layers - 1) / 7)))
            for block in NANDA_PROTOCOL["attention_blocks_8x8"]
        )
    )


def nanda_probe_layer(n_layers: int) -> int:
    """Translate official resid_post index 5 to local post-block layer L6.

    An off-by-one that is easy to get wrong: Nanda's ``resid_post`` index 5 is
    the stream *after* block 5 counting from zero, which is layer 6 in the
    one-based post-block convention used here. The direction is read from that
    layer and then applied to the branches ``nanda_branch_indices`` selects.
    """
    return max(1, min(n_layers, round(6 * n_layers / 8)))


class LiAbsoluteBoardProbe(nn.Module):
    """The released Li et al. two-layer battery-probe architecture.

    Reimplemented to match the released code rather than reusing the thesis's
    own ``PositionProbe``, because the Li intervention optimises *through* this
    probe -- its exact shape, nonlinearity and initialisation are part of the
    method, and substituting a different probe would change what the optimiser
    can reach. Note the differences from the thesis probe: ReLU rather than
    GELU, hidden width 128, no input LayerNorm, and normal(0, 0.02) init.
    """

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
    """Independent Li-style nonlinear absolute-board probes by layer.

    One probe per edited layer, trained separately. Li edits several layers in
    sequence, and each edit must be driven by a probe fitted to that layer's
    own activation statistics -- a probe trained at layer 6 reads layer 4 very
    differently, and reusing it would optimise the activation towards a board
    the layer-4 stream does not encode that way.
    """

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
    """Apply Li's Adam-on-activation intervention to a batch of states.

    The probe is frozen and the *activation* is the optimisation variable --
    the inversion that defines the method. Adam is asked to find a residual
    vector that the frozen probe decodes as the counterfactual board, starting
    from the model's own activation.

    The weighting is what keeps the edit targeted. Cross-entropy is computed
    per square and then weighted: the flipped square gets weight 1.0, every
    other square gets ``preservation_weight``. So the objective is "make the
    probe say this square changed colour, while continuing to say everything
    else is what it already was". Without the preservation term the optimiser
    is free to rewrite the whole decoded board, and any downstream change in
    the move distribution could no longer be attributed to the one square.

    Cost note: this runs ``steps`` Adam iterations per layer per case, which is
    why the Li arm dominates the suite's runtime and why the smoke profile cuts
    steps to 10 -- a setting that produces a pipeline check, not a result.

    Returns the edited activation plus the initial and final weighted losses,
    which are the diagnostic for whether the optimisation actually reached the
    counterfactual board: an edit whose loss barely moved is not evidence about
    the model, only about the optimiser failing.
    """
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if not 0 <= preservation_weight <= 1:
        raise ValueError("preservation_weight must lie in [0, 1]")
    probe.requires_grad_(False)
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
    # Full weight on the square being flipped, preservation_weight elsewhere.
    weights[torch.arange(edited.shape[0], device=edited.device), changed_squares] = 1.0
    # Adam over the activation tensor, not over any model parameter.
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
    """Return Nanda's normalized target-class vector (not a class difference).

    The parenthetical is the whole point and the easiest thing to get wrong.
    The thesis's own adapted method edits along ``w_target - w_source``, the
    difference between two class directions, which is the natural "move the
    representation from this class to that class" vector. Nanda's published
    intervention does not: it adds the target class's weight row on its own.

    Both are defensible, they are not the same edit, and silently substituting
    the difference vector here would mean reporting the adapted method twice
    while calling one of them Nanda. The two are kept distinct so the
    comparison between them is real.

    Requirements enforced above: the probe must be linear and must have no
    input LayerNorm, because a normalised input would make the weight row a
    direction in normalised space rather than in the residual stream, and
    adding it to the raw stream would be a different operation than intended.
    Vectors are unit-normalised so that ``scale`` means the same magnitude
    across layers, models and board sizes -- otherwise the fixed 2.3 would be a
    different-sized edit in every condition.
    """
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
    """Add scale * direction at the final prefix position only.

    Only the last position is edited because only the last position produces
    the next-move prediction being measured. Editing earlier positions would
    also change the board the model sees at those steps, mixing the
    intervention with a different counterfactual.

    ``last_indices`` is per row rather than a single index: prefixes in a batch
    have different lengths, so the final real position differs per case. The
    edit is computed in fp32 and cast back to the branch dtype, so a bf16
    forward pass does not quantise the direction before it is added.
    """
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
    """Re-run the forward pass with the direction added to selected branches.

    The block loop is unrolled by hand rather than hooked, because Nanda's
    intervention targets the *attention branch output* -- the vector attention
    contributes to the residual stream -- not the stream itself. That tensor
    only exists inside the block, between ``attn(ln_1(x))`` and the residual
    addition, so the block's internals have to be reproduced here to reach it.

    The two arms handle the two architectures: a Transformer block exposes
    ``attn``, a Mamba block exposes ``mixer`` and its pre-residual output is
    ``resid_dropout(_mix(ln_1(x)))``. The mixer branch is the closest
    structural analogue of the attention branch -- it is the sequence-mixing
    contribution to the residual stream in each case -- which is what makes the
    protocol transferable at all. It is still an analogy, not the published
    intervention, and results on Mamba are labelled as extensions accordingly.

    Keeping this in lockstep with the model definitions is the maintenance cost
    of the approach: a change to block internals in ``models/`` must be
    mirrored here, or the intervention silently edits the wrong tensor.
    """
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
    """Run the shared frozen model without requiring either probe family.

    The unedited baseline every method is scored against. It is written out
    here rather than reusing a model's own ``forward`` so that the baseline
    goes through *exactly* the same embedding, block loop, final LayerNorm,
    readout and dtype path as the edited passes. If the baseline took a
    different route -- a different autocast boundary, say -- part of the
    measured intervention effect would be that difference rather than the edit.

    It also deliberately takes no probe: the baseline must not depend on
    anything the Li or Nanda arms fit, so it stays valid even when one arm is
    skipped.
    """
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
    """Build the counterfactual board each Li edit is optimised towards.

    The ground truth comes from replaying the prefix on the reference engine
    and then applying the case's operation, rather than from anything the model
    or a probe predicts. That keeps the target independent of the thing being
    measured: the optimiser is pushed towards the board that would actually
    result from the flip, not towards the model's belief about it.

    Li's probes are absolute (black / white / empty), so the engine's
    +1/-1/0 state is re-encoded into the position-probe class contract below.
    """
    rows: list[torch.Tensor] = []
    for case in cases:
        board = OthelloBoardState(board_size)
        board.update(case.prefix_raw)
        state = np.asarray(board.get_state(), dtype=np.int8).reshape(-1).copy()
        if case.operation == "flip":
            state[case.square] *= -1
        elif case.operation == "erase":
            state[case.square] = 0
        else:
            raise ValueError(f"Unknown intervention operation: {case.operation!r}")
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
    """Run the forward pass, re-optimising the activation at each Li layer.

    The edits are *sequential*, not independent: the block loop runs normally,
    and whenever it reaches a configured layer the residual at the final prefix
    position is replaced by the optimised version before the next block sees
    it. So layer 6 is edited on top of the already-edited layer 5 output, and
    the intervention compounds through the second half of the network. Editing
    each layer from the clean stream instead would be a different and much
    weaker manipulation.

    The desired label vector is built from the probe's *own current reading*
    with only the flipped square overwritten by ground truth. This is the
    preservation principle again, now at the target level: the optimiser is not
    asked to correct the probe's mistakes elsewhere on the board, only to
    change the one square. Asking for a fully correct board would confound the
    intervention with probe error.

    Only the final prefix position is edited, for the reason given in
    ``_apply_last_position_edit``.

    ``completed_layers`` is checked against the configured layer set at the end
    because a silent mismatch -- a normalised index falling outside the block
    range -- would produce a weaker intervention that still returns plausible
    logits, and nothing downstream could detect it.

    Per-layer optimisation diagnostics are returned alongside the logits so a
    null result can be told apart from a failed optimisation.
    """
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
            # Start from what this probe already reads off the stream, then
            # overwrite only the flipped square with the true counterfactual
            # label. Everything else is asked to stay as the probe sees it.
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
        # Write the optimised activation back into the stream so subsequent
        # blocks run on the edited state -- this is what makes the edits
        # sequential rather than independent.
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
    """Mean per-square accuracy pooled over every Li probe layer.

    Deliberately a single pooled number rather than the per-layer breakdown the
    position-probe module produces: this is a sanity gate on the intervention,
    not a result. A Li run is only interpretable if its probes actually decode
    the board -- optimising an activation through a probe that is near chance
    tells you nothing about the model. The per-layer depth curve is reported
    from ``position_probe`` instead.
    """
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
    """Train the released Li probe shape on the thesis fixed split.

    Two conventions are being held at once: the probe *architecture* is Li's
    (see ``LiAbsoluteBoardProbe``), because the intervention optimises through
    it, while the *data protocol* is the thesis's -- the same fixed
    train/validation/test split used by every other probe, with selection on
    validation and a single scoring on test.

    Mixing them this way is intentional. Using Li's shape keeps the
    intervention faithful; using the thesis split keeps the probe accuracy
    comparable to every other probe number reported, and stops the Li arm from
    being validated on data the rest of the pipeline treats as held out.
    """
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
    """Write JSON via a temporary file and rename into place.

    The suite is resumable and its state file is rewritten after every method
    completes. A run interrupted mid-write -- a preempted GPU job, a
    disconnected notebook -- would leave a truncated JSON file that the next
    run cannot parse, losing hours of completed Li optimisation. ``os.replace``
    is atomic on the same filesystem, so the state file is always either the
    old version or the new one.

    Keys are sorted so two runs producing the same results produce
    byte-identical files, which makes diffing artefacts across runs useful.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class CausalInterventionSuite:
    """Resumable external-artifact runner for the three causal methods.

    Resumability is not a convenience here. A full run is the most expensive
    part of the thesis pipeline -- Li alone is 1000 Adam steps per layer per
    case, for 1000 cases -- and it is run once per (architecture, objective,
    board size) cell. Losing a completed Li arm because the Nanda arm crashed
    would cost hours, so each method writes its results into the state file as
    it finishes and a rerun skips what is already there.

    The safety mechanism that makes resumption trustworthy is the ``identity``
    block: checkpoint hash, split manifest hash, condition and profile. Results
    are only reused when the identity matches exactly. Without it, resuming
    after swapping a checkpoint would quietly mix results from two different
    models into one file -- and nothing in the output would reveal it.

    The one deliberate exception is the migration path below, which invalidates
    the adapted arm alone while keeping Li and Nanda. It exists because the
    adapted method's random control changed: it originally drew independent
    random directions per layer, which does not preserve the geometric
    relationship between layers that the real edit has, making the control
    easier than the intervention rather than matched to it. The fix was to
    share one signed permutation across layers. Old adapted results are
    archived rather than deleted, with the reason recorded, since they are not
    wrong so much as answering a weaker question.
    """

    def __init__(
        self,
        prepared: Any,
        *,
        profile: str,
        methods: Sequence[str] = METHODS,
        jepa_readout: str = "best",
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
        self._readout_artifacts: dict[str, Path] = {}
        self.readout_selection = self._resolve_readout_selection(jepa_readout)
        self.output_dir = Path(prepared.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / f"results__{CAUSAL_SUITE_ID}.json"
        condition_scope = causal_claim_scope(
            prepared.case.architecture,
            prepared.case.objective,
            prepared.case.board_size,
        )
        # Everything that would invalidate a cached result. A resumed run must
        # match this exactly or it refuses to continue; the checkpoint hash and
        # split manifest hash are what stop results from two different models
        # or two different splits being merged into one artifact.
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
            "readout_selection": self.readout_selection,
            "adapted_protocol": ADAPTED_PROTOCOL_ID,
            "adapted_random_control": ADAPTED_RANDOM_CONTROL_ID,
            "adapted_alpha_values": [
                float(value)
                for value in self.evaluator.config.intervention_alpha_values
            ],
            # A smoke run must never be mistaken for a result, so its scope
            # label overrides the condition label entirely.
            "claim_scope": (
                "pipeline_smoke_no_thesis_claim"
                if profile == "smoke"
                else condition_scope
            ),
        }
        if self.state_path.is_file() and not force:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            # A state file written before the adapted control was fixed matches
            # the current identity in every field except the three adapted
            # keys. That is a recognisable signature, so rather than discarding
            # the expensive Li and Nanda arms, only the adapted arm is retired.
            legacy_identity = dict(identity)
            legacy_identity.pop("adapted_protocol")
            legacy_identity.pop("adapted_random_control")
            legacy_identity.pop("adapted_alpha_values")
            if state.get("identity") == legacy_identity:
                previous = state.get("methods", {}).pop(ADAPTED_METHOD, None)
                if previous is not None:
                    archive = self.output_dir / (
                        "superseded_adapted__"
                        f"{previous.get('protocol', 'unknown')}.json"
                    )
                    _atomic_json(archive, previous)
                    state.setdefault("superseded_methods", {})[ADAPTED_METHOD] = {
                        "protocol": previous.get("protocol", "unknown"),
                        "artifact": archive.name,
                        "reason": (
                            "independent layer-wise random directions did not "
                            "preserve cross-layer intervention geometry"
                        ),
                    }
                    state.get("timing_seconds", {}).pop(ADAPTED_METHOD, None)
                state["identity"] = identity
                state.setdefault("migrations", []).append(
                    {
                        "adapted_protocol": ADAPTED_PROTOCOL_ID,
                        "action": "preserved Li/Nanda and invalidated only adapted",
                    }
                )
                _atomic_json(self.state_path, state)
            elif state.get("identity") != identity:
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

    def _resolve_readout_selection(self, policy: str) -> dict[str, Any]:
        """Resolve and validate the frozen readout before any causal work starts.

        Done in ``__init__``, deliberately: the readout must be settled before
        a single Adam step is spent, because discovering a missing or
        mismatched head after the Li arm has run would waste the most expensive
        part of the pipeline.

        The head is *reused* from the common evaluation rather than retrained
        here. Training a fresh head for the causal suite would give the
        intervention a readout fitted under different conditions from the one
        the legality numbers were measured with, and the two sets of results
        would no longer describe the same model. The checkpoint and split
        identity checks below enforce that the reused head belongs to this run.
        """
        if self.prepared.case.objective == "ar":
            return {
                "requested_policy": "not_applicable",
                "selected": "native_ar",
                "selection_split": "not_applicable",
                "selection_metric": "not_applicable",
                "selection_scores": {},
                "tie_break": "not_applicable",
            }

        policy = str(policy).lower()
        if policy not in {"best", *JEPA_READOUT_TYPES}:
            raise ValueError(
                "CausalInterventionSuite expects 'best', 'linear', or 'mlp'; "
                "the runner expands 'both' into two isolated suites."
            )
        common_dir = self.prepared.run_dir / "thesis_eval" / "final"
        current_split = self.evaluator.split.manifest_hash
        current_checkpoint = self.evaluator.results["metadata"]["checkpoint_sha256"]
        compact_results: dict[str, Any] = {"frozen_next_move": {}}
        required = JEPA_READOUT_TYPES if policy == "best" else (policy,)
        for head_type in required:
            artifact = common_dir / f"{head_type}_next_move_head__{PROTOCOL_ID}.pt"
            if not artifact.is_file():
                raise RuntimeError(
                    f"Missing common {head_type} frozen-head artifact: {artifact}. "
                    "Complete the common evaluation before causal intervention."
                )
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            if payload.get("head_type") not in (None, head_type):
                raise RuntimeError(f"Frozen JEPA artifact is not a {head_type} readout")
            saved_split = payload.get("split_manifest_hash")
            if saved_split not in (None, current_split):
                raise RuntimeError("Frozen JEPA readout has a different split identity")
            saved_checkpoint = (
                payload.get("checkpoint_sha256")
                or (payload.get("run_metadata") or {}).get("checkpoint_sha256")
            )
            if saved_checkpoint not in (None, current_checkpoint):
                raise RuntimeError(
                    "Frozen JEPA readout has a different checkpoint identity"
                )
            compact_results["frozen_next_move"][head_type] = {
                "train": payload.get("train") or {}
            }
            self._readout_artifacts[head_type] = artifact
        return select_jepa_readout(compact_results, policy)

    def _save(self) -> None:
        _atomic_json(self.state_path, self.state)

    def _legacy_output_dir(self) -> Path:
        """Return the read-only v1 artifact root for reusable probe weights."""
        return (
            self.prepared.run_dir
            / "causal_intervention"
            / "causal_intervention_suite_v1"
            / self.profile.name
        )

    def _cases(self) -> tuple[tuple[InterventionCase, ...], tuple[InterventionCase, ...]]:
        """Build the shared selection and test case sets, and record them.

        Cases are drawn from the evaluator's own selection and test corpus
        splits, so a case can never be built from a game the model trained on.
        The two seed offsets keep the case streams independent: without them
        the selection and test sets would start from the same RNG state and
        share their early cases, quietly turning held-out cases into tuning
        cases.

        The manifest is written out because every reported causal number is
        conditional on which cases were drawn. Storing them makes a result
        auditable and lets a later run re-examine an individual case without
        rebuilding the suite.
        """
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
        """Return the frozen next-move head and a label describing its origin.

        The asymmetry between the two objectives is the reason this exists. An
        AR model already has ``lm_head`` -- its own trained output layer, the
        one used for every legality number reported for it. A JEPA model has
        nothing: the objective never trains a move head, so one is attached to
        the frozen encoder.

        The label travels into the results so a reader can see which case a
        number came from. This is a real asymmetry in the comparison and is
        reported rather than smoothed over: the AR model is measured through
        its own head, the JEPA model through a head someone else fitted on top
        of it.
        """
        if self._readout_cache is not None:
            return self._readout_cache
        if self.prepared.case.objective == "ar":
            readout = self.prepared.primary_model.lm_head
            readout.eval()
            self._readout_cache = (readout, "native_ar")
            return self._readout_cache

        head_type = self.readout_selection["selected"]
        artifact = self._readout_artifacts[head_type]
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        selected = (payload.get("tuning") or {}).get("selected") or {}
        wrapper_kwargs: dict[str, Any] = {
            "encoder_precision": self.evaluator.config.encoder_precision
        }
        if head_type == "mlp":
            wrapper_kwargs.update(
                hidden_dim=int(
                    selected.get("hidden_dim", self.evaluator.config.mlp_hidden_dim)
                ),
                dropout=float(
                    selected.get("dropout", self.evaluator.config.mlp_dropout)
                ),
            )
        wrapper = FrozenEncoderNextMoveHead(
            self.evaluator.encoder,
            head_type,
            **wrapper_kwargs,
        ).to(self.device)
        wrapper.head.load_state_dict(payload["head"])
        label = f"reused_common_frozen_{head_type}"
        wrapper.eval()
        self._readout_cache = (wrapper.head, label)
        return self._readout_cache

    def _raw_probe_bank(self) -> PositionProbeBank:
        """Load (or fit) the unnormalised linear probe the edits are built from.

        "Raw" means no input LayerNorm: the Nanda and adapted edits add a
        direction to the residual stream itself, so the probe's weights must
        live in that stream's coordinates. A probe with a normalising input
        layer has weights in normalised space, and adding one of its rows to
        the raw stream would be a geometrically different edit.

        Search order is common artifacts, then the v1 directory, then this
        suite's own output. Reusing the probe the rest of the evaluation
        already fitted keeps the intervention direction identical to the one
        the probe results describe, and avoids refitting per method. Every
        reuse path re-checks checkpoint and split identity, because a probe
        from a different model is the kind of error that produces a plausible
        number rather than a crash.
        """
        if self._raw_probe_cache is not None:
            return self._raw_probe_cache
        common_dir = self.prepared.run_dir / "thesis_eval" / "final"
        probe_dirs = (
            common_dir,
            self._legacy_output_dir(),
            *(() if self.force else (self.output_dir,)),
        )
        probe_dir = next(
            (
                directory
                for directory in probe_dirs
                if (
                    directory
                    / f"raw_linear_intervention_probe__{PROTOCOL_ID}.pt"
                ).is_file()
            ),
            None,
        )
        if probe_dir is not None:
            probe_path = (
                probe_dir / f"raw_linear_intervention_probe__{PROTOCOL_ID}.pt"
            )
            results_path = probe_dir / f"results__{PROTOCOL_ID}.json"
            if not results_path.is_file():
                raise RuntimeError(
                    f"Reusable intervention probe lacks results identity: {probe_dir}"
                )
            probe_results = json.loads(
                results_path.read_text(encoding="utf-8")
            )
            if probe_results.get("metadata", {}).get("checkpoint_sha256") != (
                self.evaluator.results["metadata"]["checkpoint_sha256"]
            ):
                raise RuntimeError(
                    "Reusable intervention probe has a different checkpoint"
                )
            if probe_results.get("metadata", {}).get("split_manifest_hash") != (
                self.evaluator.split.manifest_hash
            ):
                raise RuntimeError("Reusable intervention probe has a different split")
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
            payload = torch.load(probe_path, map_location="cpu", weights_only=False)
            bank.load_state_dict(payload["probe_bank"])
            bank.eval()
        else:
            bank, _ = self.evaluator._intervention_probe_bank()
        self._raw_probe_cache = bank
        return bank

    def _li_probe_bank(self) -> LiProbeBank:
        """Load or fit the Li probes, then freeze them.

        Separate from ``_raw_probe_bank`` because the two methods need
        different probes: Li optimises through a nonlinear absolute-board probe
        of his own shape, the Nanda and adapted methods read a direction off a
        linear relative-board probe. They are not interchangeable and are
        cached separately.

        ``requires_grad_(False)`` at the end is load-bearing rather than
        housekeeping: ``li_optimize_activation`` runs Adam inside the forward
        pass, and if the probe's parameters were still trainable the probe
        would drift towards agreeing with the edit instead of the edit having
        to satisfy a fixed probe. That would make every intervention look
        successful.
        """
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
        legacy_path = (
            self._legacy_output_dir()
            / "li_absolute_probe__causal_intervention_suite_v1.pt"
        )
        reusable_path = next(
            (
                candidate
                for candidate in (path, legacy_path)
                if candidate.is_file()
            ),
            None,
        )
        if reusable_path is not None and not self.force:
            payload = torch.load(
                reusable_path, map_location="cpu", weights_only=False
            )
            if payload.get("checkpoint_sha256") not in (
                None,
                self.evaluator.results["metadata"]["checkpoint_sha256"],
            ):
                raise RuntimeError("Reusable Li probe has a different checkpoint")
            if payload.get("split_manifest_hash") not in (
                None,
                self.evaluator.split.manifest_hash,
            ):
                raise RuntimeError("Reusable Li probe has a different split")
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
        bank.requires_grad_(False)
        bank.eval()
        self._li_probe_cache = bank
        return bank

    def _run_nanda(
        self, test_cases: Sequence[InterventionCase]
    ) -> dict[str, Any]:
        """Score the Nanda arm: fixed-scale single-direction edit vs. unedited.

        No selection pass and no alpha sweep -- the scale is the published
        constant, used as given. That is the point of including this method:
        it is the one arm with nothing tuned on the data, so its effect cannot
        be an artefact of choosing a favourable hyper-parameter.

        Null and patched logits are computed per batch on the same cases in the
        same order, so the comparison is paired: every case contributes both
        numbers and the difference is within-case rather than between two
        independently sampled sets.
        """
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
            # Recorded per result, because on Mamba the edited tensor is the
            # mixer branch rather than an attention output -- a structural
            # analogue, not the published surface.
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
        """Score the thesis's own adapted method.

        The only arm that takes selection cases, because it is the only one
        with something to select: the edit magnitude alpha is swept and the
        best value is chosen on the selection set, then applied once to the
        test set. That is what makes a tuned method comparable to two untuned
        ones -- the tuning happens on data that never contributes to a reported
        number.

        The differences from the two published methods are recorded in
        ``lineage`` and are substantive, not cosmetic: the direction is the
        target-minus-source difference rather than Nanda's target row; the
        scale is expressed in units of residual standard deviation, so one
        alpha means a comparable perturbation across layers, models and board
        sizes; every post-block residual is edited rather than a published
        subset; and the random control is a signed permutation shared across
        layers, which preserves both the per-layer norm and the geometric
        relationship between layers that the real edit has.
        """
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
        result["protocol"] = ADAPTED_PROTOCOL_ID
        result["lineage"] = (
            "thesis-specific hybrid: target-minus-source relative linear direction, "
            "residual-std scaling, every post-block residual, held-out alpha selection, "
            "and a random signed-permutation control that preserves layer-wise norms "
            "and cross-layer direction geometry"
        )
        result["readout"] = readout_label
        return result

    def _run_li(self, test_cases: Sequence[InterventionCase]) -> dict[str, Any]:
        """Score the Li arm, checkpointing after every batch.

        The per-batch checkpointing exists because this arm is the expensive
        one: 1000 Adam steps per edited layer per case means a full run is
        hours, and losing it to an interrupted session is a real cost. Each
        batch's logits and diagnostics are written to ``li_progress`` as soon
        as they are produced, so a rerun resumes at the first unfinished batch.

        The resume path re-checks both the case ids and the step count in the
        saved batch. Case ids catch a progress directory left over from a
        different case manifest; the step count catches the more insidious
        error of resuming a smoke run's 10-step batches into a full run and
        silently reporting a mixture of two optimisation budgets as one result.
        """
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
                "projected Li optimization time "
                f"{elapsed * total_batches / 60:.1f} min",
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
        """Render the human-readable comparison table for this condition.

        The summary is written to be read on its own, away from the JSON, so it
        repeats the caveats that would otherwise be lost: which condition this
        is, how far it departs from the published setting, which readout a JEPA
        number came from, and that a smoke profile supports no claim. A table
        of numbers without those qualifications is the thing most likely to be
        quoted out of context.

        Each method is reported through its own result key because the three
        arms produce structurally different outputs -- there is no single
        "intervention effect" field they share.
        """
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
        ]
        readout_selection = identity["readout_selection"]
        if identity["objective"] == "jepa":
            scores = readout_selection["selection_scores"]
            score_text = ", ".join(
                f"{head_type}={score:.6f}"
                for head_type, score in scores.items()
            )
            lines.append(
                f"- JEPA readout: `{readout_selection['selected']}`; policy "
                f"`{readout_selection['requested_policy']}`, chosen only from the "
                f"selection split by `{readout_selection['selection_metric']}` "
                f"({score_text})."
            )
        else:
            lines.append("- Readout: native AR language-model head.")
        lines += [
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
        # The adapted arm is reported against its controls rather than against
        # the unedited baseline alone. An edit of any kind perturbs the model,
        # so the question is not "did the output change" but "did editing along
        # the probe direction change it more than an equally large edit in a
        # direction carrying no board information". The delta against the
        # geometry-matched control is the number that answers that; the
        # independent-layer control is kept only as a sensitivity check, since
        # it is the weaker comparison that motivated the migration above.
        adapted = self.state["methods"].get(ADAPTED_METHOD)
        if adapted:
            targeted = adapted["test"]["probe_direction"]
            matched = adapted["test"]["geometry_matched_random"]
            independent = adapted["test"].get("independent_layer_random")
            lines += [
                "",
                "The primary adapted-method random control applies one random signed "
                "permutation per case at every layer. It therefore preserves the "
                "norms and all cross-layer angles of the probe directions while "
                "breaking their alignment with the learned feature coordinates.",
                "",
                "| Adapted comparison | Delta top-1 (pp) | Delta target mass (pp) | Delta Top-N FP+FN |",
                "|---|---:|---:|---:|",
                f"| Targeted - geometry-matched random | "
                f"{100 * (targeted['target_top1_legal'] - matched['target_top1_legal']):+.2f} | "
                f"{100 * (targeted['target_legal_probability_mass'] - matched['target_legal_probability_mass']):+.2f} | "
                f"{targeted['mean_topn_false_positive_plus_false_negative'] - matched['mean_topn_false_positive_plus_false_negative']:+.3f} |",
            ]
            if independent is not None:
                lines.append(
                    f"| Targeted - independent-layer random (sensitivity only) | "
                    f"{100 * (targeted['target_top1_legal'] - independent['target_top1_legal']):+.2f} | "
                    f"{100 * (targeted['target_legal_probability_mass'] - independent['target_legal_probability_mass']):+.2f} | "
                    f"{targeted['mean_topn_false_positive_plus_false_negative'] - independent['mean_topn_false_positive_plus_false_negative']:+.3f} |"
                )
        lines += [
            "",
            "The 8x8 Transformer-AR condition follows the released intervention "
            "algorithms and constants, but uses this thesis checkpoint, corpus, and "
            "probe split. Each JEPA run uses the one recorded post-hoc readout for all "
            "three methods; automatic selection uses validation evidence and never "
            "the final causal test. Mamba patches the mixer branch corresponding to "
            "Nanda's attention branch. Those conditions, and larger boards, are "
            "extensions.",
            "",
            "`smoke` is only an execution gate. Thesis conclusions require `full`.",
        ]
        path = self.output_dir / f"summary__{CAUSAL_SUITE_ID}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def run(self) -> dict[str, str]:
        """Run the requested methods in cost order, saving after each.

        Returns the paths of the results JSON and the markdown summary.
        """
        selection_cases, test_cases = self._cases()
        # Keep the expensive Li optimizer last; the two cheap diagnostics are
        # durably available even if a Colab allocation ends during Li.
        execution_order = (NANDA_METHOD, ADAPTED_METHOD, LI_METHOD)
        for method in execution_order:
            if method not in self.methods:
                continue
            if method in self.state["methods"] and not self.force:
                print(
                    f"\n[{method.upper()}] COMPLETE — reusing saved result",
                    flush=True,
                )
                continue
            print(f"\n[{method.upper()}] START", flush=True)
            started = time.perf_counter()
            if method == NANDA_METHOD:
                result = self._run_nanda(test_cases)
            elif method == ADAPTED_METHOD:
                result = self._run_adapted(selection_cases, test_cases)
            else:
                result = self._run_li(test_cases)
            # Stamp the scope onto the result itself, so a method block stays
            # self-describing if it is ever read apart from the identity block.
            result["claim_scope"] = self.state["identity"]["claim_scope"]
            self.state["methods"][method] = result
            elapsed = time.perf_counter() - started
            self.state["timing_seconds"][method] = elapsed
            # Persist after every method: a crash in Li must not cost Nanda.
            self._save()
            print(f"[{method.upper()}] COMPLETE in {elapsed:.2f}s", flush=True)
        summary = self.write_summary()
        return {"results": str(self.state_path), "summary": str(summary)}


__all__ = [
    "ADAPTED_PROTOCOL_ID",
    "ADAPTED_RANDOM_CONTROL_ID",
    "ADAPTED_METHOD",
    "CAUSAL_SUITE_ID",
    "CausalInterventionSuite",
    "JEPA_READOUT_POLICIES",
    "JEPA_READOUT_TYPES",
    "LI_METHOD",
    "LI_PROTOCOL",
    "LiAbsoluteBoardProbe",
    "LiProbeBank",
    "METHODS",
    "NANDA_METHOD",
    "NANDA_PROTOCOL",
    "PROFILES",
    "READOUT_SELECTION_METRIC",
    "causal_output_subdir",
    "causal_claim_scope",
    "build_flip_intervention_cases",
    "li_layer_indices",
    "li_optimize_activation",
    "nanda_branch_indices",
    "nanda_probe_layer",
    "nanda_target_directions",
    "resolve_causal_readout_runs",
    "select_jepa_readout",
    "train_li_probe_bank",
]
