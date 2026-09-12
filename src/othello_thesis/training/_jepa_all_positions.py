"""Internal implementation module extracted from the canonical JEPA trainer.

The all-position machinery. Autoregressive training gets a gradient signal at
every ply of every game for free -- one forward pass, T predictions. A naive
JEPA step gets one (context, target) pair per game, which is a factor of ~T
less supervision for the same compute, and would make any comparison between
the two objectives a comparison of sample budgets rather than of objectives.

``forward_all_position_v1`` closes that gap: one causal pass over the batch
produces the context state at *every* prefix boundary, and each of those is
paired with its own target. A batch of 256 games at 8x8 therefore yields on the
order of 10^4 training pairs from a single encoder pass.

Two consequences shape the rest of the file. The target side now has to encode
tens of thousands of independent length-one sequences, so it is chunked (and on
Mamba it uses the algebraic length-one path in models/mamba.py). And statistics
are computed *per absolute position* rather than over the flattened pool -- see
the note in ``forward_all_position_v1``.

This module also holds the view construction and the collapse monitor shared
with the single-window path.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from othello_thesis.objectives.jepa import normalize_jepa_view_mode
from othello_thesis.training._jepa_config import (
    TrainConfig,
    jepa_view_mode,
    normalize_multiaction_loss_mode,
    objective_class,
)
from othello_thesis.training._jepa_runtime import unwrap_model

def output_metric(out: dict[str, Any], key: str) -> torch.Tensor | float:
    """Fetch a metric from flat model output or nested diagnostics."""
    if key in out:
        return out[key]
    diagnostics = out.get("diagnostics")
    if isinstance(diagnostics, dict) and key in diagnostics:
        return diagnostics[key]
    return float("nan")

def valid_window_mask(
    x: torch.Tensor,
    *,
    context_length: int,
    horizon: int,
    pad_token: int,
) -> torch.Tensor:
    """Return rows whose context plus K future target tokens contain no PAD."""
    end = context_length + horizon
    return (x[:, :end] != pad_token).all(dim=1)

def sample_valid_window(
    x: torch.Tensor,
    *,
    horizon: int,
    pad_token: int,
    attempts: int,
) -> tuple[int, torch.Tensor]:
    """Sample a batch context length and keep rows with a valid K-token target.

    Retries because a uniformly drawn ``t`` is often longer than most games in
    the batch, leaving no valid row at all. The first draw that keeps at least
    one row wins; if every attempt fails, the best seen is returned and the
    caller decides whether to skip.

    Note the asymmetry: the early return fires on ``count > 0``, so the sampled
    ``t`` is not the one that maximises valid rows. That is deliberate -- always
    taking the best ``t`` would bias training toward short contexts, which are
    the ones most games survive.
    """
    _, seq_len = x.shape
    best_t = 1
    best_mask = valid_window_mask(
        x,
        context_length=best_t,
        horizon=horizon,
        pad_token=pad_token,
    )
    best_count = int(best_mask.sum().item())

    n_attempts = max(1, attempts)
    for _ in range(n_attempts):
        t = torch.randint(1, seq_len - horizon + 1, (1,)).item()
        mask = valid_window_mask(
            x,
            context_length=t,
            horizon=horizon,
            pad_token=pad_token,
        )
        count = int(mask.sum().item())
        if count > 0:
            return t, mask
        if count > best_count:
            best_t = t
            best_mask = mask
            best_count = count

    return best_t, best_mask

def build_target_positions(
    *,
    context_length: int,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``(B, K)`` absolute future positions for a batch-uniform t."""
    positions = torch.arange(
        context_length,
        context_length + horizon,
        device=device,
        dtype=torch.long,
    )
    return positions.unsqueeze(0).expand(batch_size, -1)

def build_local_target_positions(
    *,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``(B, K)`` target positions local to a target-only future view."""
    positions = torch.arange(0, horizon, device=device, dtype=torch.long)
    return positions.unsqueeze(0).expand(batch_size, -1)

def build_jepa_views(
    cfg: TrainConfig,
    x_valid: torch.Tensor,
    *,
    context_length: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Construct context/target tensors for the configured JEPA view.

    ``hard_disjoint_future`` is target-only future: the target encoder sees
    ``x[:, t:t+K]`` and never sees the prefix observed by the context encoder.
    Predictive objectives retain absolute target positions ``t..t+K-1`` so
    removing the prefix is the only target-side information change.
    ``disjoint_future`` retains the v4/v5 behavior where target sees full
    history and target embeddings are gathered from absolute future positions.

    The distinction is the whole point of "hard". Under ``disjoint_future`` the
    target encoder still reads the prefix, so a target embedding contains the
    context the predictor was given -- the predictor can partly succeed by
    reproducing what it already has. ``hard_disjoint_future`` cuts the prefix
    away entirely, leaving genuine prediction as the only route to a low loss.

    Absolute positions are preserved on the target even though it is now a
    length-one sequence: the ply index is legitimate information about *when*
    the move happens, and stripping it would change two things at once and make
    the ablation uninterpretable.
    """
    x_ctx = x_valid[:, :context_length].contiguous()
    view_mode = jepa_view_mode(cfg)
    batch_size = x_valid.size(0)
    device = x_valid.device
    if view_mode == "hard_disjoint_future":
        x_tgt = x_valid[:, context_length: context_length + horizon].contiguous()
        if objective_class(cfg) == "jepa":
            # Predictive hard-disjoint targets keep the same absolute position
            # embeddings as their non-hard-disjoint counterparts. Only the
            # prefix tokens are removed from the target encoder input.
            target_positions = build_target_positions(
                context_length=context_length,
                horizon=horizon,
                batch_size=batch_size,
                device=device,
            )
        else:
            target_positions = build_local_target_positions(
                horizon=horizon,
                batch_size=batch_size,
                device=device,
            )
        return x_ctx, x_tgt, target_positions
    x_tgt = x_valid[:, : context_length + horizon].contiguous()
    target_positions = None
    if view_mode == "disjoint_future":
        target_positions = build_target_positions(
            context_length=context_length,
            horizon=horizon,
            batch_size=batch_size,
            device=device,
        )
    return x_ctx, x_tgt, target_positions

def maybe_warn_collapse(cfg: TrainConfig, out: dict[str, torch.Tensor], step: int) -> None:
    """Print collapse warnings without aborting the run.

    Warns rather than raises on purpose. Collapse is a matter of degree and the
    thresholds are heuristic, so an automatic abort would kill runs that recover
    during warmup. The signals are logged periodically instead, and the decision
    to discard a run is made by the researcher looking at the trajectory.

    The three signals, in order of how early they move: ``z_std`` falling means
    the target embeddings are losing spread; ``cos_sim_offdiag`` rising toward 1
    means distinct positions are becoming indistinguishable; contrastive
    ``positive_accuracy`` falling means the objective itself has stopped being
    solvable. The accuracy check is suppressed until after warmup, when it is
    expected to be low for benign reasons.
    """
    every = int(cfg.log_collapse_stats_every_steps)
    if every <= 0 or step % every != 0:
        return
    z_std = float(output_metric(out, "z_std"))
    cos_sim = float(output_metric(out, "cos_sim_offdiag"))
    positive_accuracy = float(output_metric(out, "positive_accuracy"))
    diag_accuracy = float(output_metric(out, "diag_accuracy"))
    uses_contrastive_accuracy = objective_class(cfg) in {
        "jepa_contrastive",
        "jepa_hard_disjoint_infonce",
        "jepa_order_aware",
    } or (
        objective_class(cfg) == "jepa_multiaction"
        and normalize_multiaction_loss_mode(cfg.rollout_loss_mode) == "infonce"
    ) or (
        objective_class(cfg) == "jepa_action_conditioned"
        and (
            cfg.action_loss_mode == "grouped_infonce"
            or (
                cfg.action_loss_mode == "hybrid"
                and cfg.hybrid_base == "grouped_infonce"
            )
        )
    )
    print(
        f"  collapse stats step={step}: "
        f"z_std={z_std:.4f} c_std={float(output_metric(out, 'c_std')):.4f} "
        f"p_std={float(output_metric(out, 'p_std')):.4f} "
        f"cos_sim_offdiag={cos_sim:.4f}",
        flush=True,
    )
    if uses_contrastive_accuracy and not math.isnan(positive_accuracy):
        print(
            f"  contrastive stats step={step}: "
            f"positive_accuracy={positive_accuracy:.4f} "
            f"diag_accuracy={diag_accuracy:.4f} "
            f"n_positives={float(output_metric(out, 'n_positives_mean')):.2f}",
            flush=True,
        )
    if not math.isnan(z_std) and z_std < cfg.collapse_warning_z_std:
        print(
            f"  WARNING: target z_std {z_std:.4f} < "
            f"collapse_warning_z_std={cfg.collapse_warning_z_std:.4f}",
            flush=True,
        )
    if not math.isnan(cos_sim) and cos_sim > cfg.collapse_warning_cos_sim:
        print(
            f"  WARNING: target off-diagonal cosine {cos_sim:.4f} > "
            f"collapse_warning_cos_sim={cfg.collapse_warning_cos_sim:.4f}",
            flush=True,
        )
    if (
        uses_contrastive_accuracy
        and step > cfg.warmup_steps
        and not math.isnan(positive_accuracy)
        and positive_accuracy < cfg.collapse_warning_contrastive_accuracy
    ):
        print(
            f"  WARNING: positive_accuracy {positive_accuracy:.4f} < "
            f"collapse_warning_contrastive_accuracy="
            f"{cfg.collapse_warning_contrastive_accuracy:.4f}",
            flush=True,
        )

def forward_all_position_v1(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    pad_token: int,
    target_chunk_size: int,
    stats_chunk_size: int,
    profile: bool = False,
) -> tuple[dict[str, torch.Tensor], int]:
    """Evaluate a JEPA objective at every valid prefix boundary.

    The context encoder runs once over the causal sequence.  The target view
    follows ``jepa_config.view_mode``:

    - ``hard_disjoint_future``: each target move is encoded as an independent
      length-one sequence at its original absolute position, so no prefix can
      leak into the target branch.
    - ``nested``: the target at boundary ``t`` is the target-path hidden state
      of the full prefix ``x[:t+1]`` at position ``t``.  With a shared (v1)
      encoder this reuses the context pass; with an EMA target encoder it is
      one extra no-grad pass.

    The loss follows ``jepa_config.loss_type`` (``vicreg``, ``smooth_l1``, or
    ``infonce``) and every statistic is computed separately at each absolute
    position across games, then pair-count weighted.  Flattening positions
    first would let absolute positional embeddings manufacture variance (or
    trivial in-batch negatives) and would mix highly correlated positions from
    the same game in the estimates.

    That last paragraph is the subtle part and worth restating. Pool all
    boundaries from all games together and the contrastive task becomes easy for
    the wrong reason: ply 3 and ply 40 differ in their positional embedding
    alone, so telling them apart requires no board understanding at all. The
    variance statistics are inflated the same way. Grouping by absolute position
    means every negative in a comparison comes from the *same* ply of a
    *different* game, so the only thing distinguishing them is the position on
    the board -- which is exactly what the objective is supposed to be learning.

    Returns ``(outputs, effective_batch_size)``; the second value is the number
    of pairs that contributed, which the caller uses to weight the loss.
    """
    raw_model = unwrap_model(model)
    # Validate the v1 JEPA interface rather than the concrete Python class.
    # Colab notebooks may reload objective modules before launching training;
    # after a reload, two otherwise identical OthelloJEPA classes have
    # different identities and an isinstance check rejects a valid model.
    required_attributes = (
        "context_encoder",
        "jepa_config",
        "encode_hidden",
        "_predict_embeddings",
        "_target_embeddings",
    )
    missing = [name for name in required_attributes if not hasattr(raw_model, name)]
    if missing:
        raise TypeError(
            "all-position v1 requires the OthelloJEPA interface; missing: "
            + ", ".join(missing)
        )
    if x.ndim != 2 or x.size(1) < 2:
        raise ValueError(f"Expected x with shape (B, T>=2), got {tuple(x.shape)}")

    # Multi-step horizons (v3/v4) supervise a K-token future window at every
    # boundary. That path is structurally different (K targets per boundary,
    # multi-position predictor), so it lives in its own function. Horizon-1
    # objectives (v1/v2/v5) stay on the battle-tested single-target path below.
    horizon = int(getattr(raw_model.jepa_config, "prediction_horizon", 1) or 1)
    if horizon > 1:
        return _forward_all_position_kstep(
            raw_model,
            x,
            pad_token=pad_token,
            horizon=horizon,
            target_chunk_size=target_chunk_size,
            stats_chunk_size=stats_chunk_size,
        )

    profile_times: dict[str, float] = {}

    def finish_profile_phase(name: str, started_at: float) -> float:
        if not profile:
            return started_at
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        now = time.perf_counter()
        profile_times[name] = now - started_at
        return now

    if profile and x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    phase_started_at = time.perf_counter()

    # Boundary t uses context x[:t] and target x[t].  Hence context hidden
    # index t-1 pairs with token index t; this exactly expands the historical
    # random-boundary domain to all available boundaries in x.
    context_hidden = raw_model.encode_hidden(raw_model.context_encoder, x)
    phase_started_at = finish_profile_phase("context", phase_started_at)
    # A boundary is usable when the token being predicted is a real move, so
    # validity is tested on x[:, 1:] (every token except the first, which can
    # only ever be a context token). nonzero flattens the (game, boundary)
    # grid into one list of pairs, which is what lets every boundary in the
    # batch be processed as a single flat batch.
    valid = x[:, 1:] != pad_token
    pair_indices = valid.nonzero(as_tuple=False)
    if pair_indices.numel() == 0:
        return {}, 0

    game_idx = pair_indices[:, 0]
    boundary_idx = pair_indices[:, 1]  # zero-based within x[:, 1:]
    absolute_positions = boundary_idx + 1
    context_summary = context_hidden[game_idx, boundary_idx, :]
    target_tokens = x[game_idx, absolute_positions].unsqueeze(1)
    target_positions = absolute_positions.unsqueeze(1)

    jcfg = raw_model.jepa_config
    view_mode = normalize_jepa_view_mode(jcfg.view_mode)
    loss_type = jcfg.loss_type
    if loss_type not in {"vicreg", "smooth_l1", "infonce"}:
        raise ValueError(
            f"all-position supervision supports vicreg/smooth_l1/infonce, got {loss_type!r}"
        )

    z_pred = raw_model._predict_embeddings(
        context_summary.unsqueeze(1), target_positions
    )
    phase_started_at = finish_profile_phase("predictor", phase_started_at)
    chunk_size = max(1, int(target_chunk_size))
    if view_mode == "nested":
        # Nested target at boundary t is the target-path encoding of x[:t+1]
        # at position t.  A shared encoder reuses the context pass verbatim;
        # an EMA target encoder runs one detached pass over the sequence.
        if hasattr(raw_model, "target_encoder"):
            with torch.no_grad():
                target_hidden = raw_model.encode_hidden(raw_model.target_encoder, x)
        else:
            target_hidden = context_hidden
        z_tgt = target_hidden[game_idx, absolute_positions, :].unsqueeze(1)
    else:
        target_parts: list[torch.Tensor] = []
        for start in range(0, target_tokens.size(0), chunk_size):
            end = min(start + chunk_size, target_tokens.size(0))
            target_parts.append(
                raw_model._target_embeddings(
                    target_tokens[start:end],
                    1,
                    target_positions[start:end],
                )
            )
        z_tgt = torch.cat(target_parts, dim=0)
    phase_started_at = finish_profile_phase("target", phase_started_at)

    # Scatter the flat list of pairs back onto a (game, position) grid. The
    # statistics below are all per-position across games, and the grid makes
    # that a masked reduction over dim 0 instead of a gather per position.
    # Invalid cells stay zero and are excluded by the mask, never by slicing.
    batch_size, seq_len = x.shape
    n_positions = seq_len - 1
    d_model = context_summary.size(-1)
    pred_grid = torch.zeros(
        batch_size, n_positions, d_model, device=x.device, dtype=z_pred.dtype
    ).index_put((game_idx, boundary_idx), z_pred[:, 0, :])
    target_grid = torch.zeros_like(pred_grid).index_put(
        (game_idx, boundary_idx), z_tgt[:, 0, :]
    )
    context_grid = context_hidden[:, :-1, :]
    phase_started_at = finish_profile_phase("grid", phase_started_at)

    weighted_loss: torch.Tensor | None = None
    weighted_metrics: dict[str, torch.Tensor] = {}
    used_pairs = 0
    eps = float(jcfg.variance_eps)
    gamma = float(jcfg.variance_threshold)
    position_chunk = max(1, int(stats_chunk_size))

    for start in range(0, n_positions, position_chunk):
        end = min(start + position_chunk, n_positions)
        mask = valid[:, start:end]
        # How many games in the batch actually reach this ply. Falls off sharply
        # toward the end of the sequence, which is why late positions often have
        # too few games to contribute.
        counts = mask.sum(dim=0)
        # Two is the minimum for anything here to be defined: a variance needs
        # two samples, and InfoNCE needs at least one negative.
        eligible = counts >= 2
        if not bool(eligible.any()):
            continue
        mask_f = mask.to(torch.float32).unsqueeze(-1)
        denom = counts.clamp_min(1).to(torch.float32)
        denom_e = denom.unsqueeze(-1)

        pred = pred_grid[:, start:end, :].float()
        target = target_grid[:, start:end, :].float()
        context = context_grid[:, start:end, :].float()

        # Masked mean/variance: multiplying by the 0/1 mask and dividing by the
        # true count gives the statistic over participating games only, without
        # materialising a ragged tensor.
        pred_mean = (pred * mask_f).sum(dim=0) / denom_e
        target_mean = (target * mask_f).sum(dim=0) / denom_e
        context_mean = (context * mask_f).sum(dim=0) / denom_e
        pred_centered = (pred - pred_mean.unsqueeze(0)) * mask_f
        target_centered = (target - target_mean.unsqueeze(0)) * mask_f
        context_centered = (context - context_mean.unsqueeze(0)) * mask_f

        pred_var = pred_centered.square().sum(dim=0) / denom_e
        target_var = target_centered.square().sum(dim=0) / denom_e
        context_var = context_centered.square().sum(dim=0) / denom_e
        pred_std = torch.sqrt(pred_var + eps)
        target_std = torch.sqrt(target_var + eps)
        context_std = torch.sqrt(context_var + eps)

        # Mean off-diagonal cosine without forming the n x n similarity matrix.
        # For unit vectors, ||sum_i u_i||^2 = sum_ij <u_i,u_j> = n + sum_(i!=j),
        # so subtracting n and dividing by n(n-1) gives the off-diagonal mean
        # directly. With tens of thousands of pairs the explicit matrix would
        # not fit.
        normalized_target = torch.nn.functional.normalize(target, dim=-1) * mask_f
        target_sum = normalized_target.sum(dim=0)
        cos_offdiag = (
            target_sum.square().sum(dim=-1) - denom
        ) / (denom * (denom - 1).clamp_min(1))
        base_metrics = {
            "z_pred_std": pred_std.mean(dim=-1),
            "z_tgt_std": target_std.mean(dim=-1),
            "z_std": torch.sqrt(target_var.clamp_min(0.0)).mean(dim=-1),
            "c_std": torch.sqrt(context_var.clamp_min(0.0)).mean(dim=-1),
            "p_std": torch.sqrt(pred_var.clamp_min(0.0)).mean(dim=-1),
            "cos_sim_offdiag": cos_offdiag,
        }

        if loss_type == "vicreg":
            inv = (
                ((pred - target).square() * mask_f).sum(dim=(0, 2))
                / (denom * d_model)
            )
            var = 0.5 * (
                torch.relu(gamma - pred_std).mean(dim=-1)
                + torch.relu(gamma - target_std).mean(dim=-1)
            )
            cov_denom = (counts - 1).clamp_min(1).to(torch.float32).view(-1, 1, 1)
            pred_cov = torch.einsum("bpd,bpe->pde", pred_centered, pred_centered) / cov_denom
            target_cov = torch.einsum("bpd,bpe->pde", target_centered, target_centered) / cov_denom
            pred_cov_penalty = (
                pred_cov.square().sum(dim=(1, 2))
                - pred_cov.diagonal(dim1=1, dim2=2).square().sum(dim=1)
            ) / d_model
            target_cov_penalty = (
                target_cov.square().sum(dim=(1, 2))
                - target_cov.diagonal(dim1=1, dim2=2).square().sum(dim=1)
            ) / d_model
            cov = pred_cov_penalty + target_cov_penalty
            loss_at_position = (
                float(jcfg.vicreg_lambda) * inv
                + float(jcfg.vicreg_mu) * var
                + float(jcfg.vicreg_nu) * cov
            )
            chunk_metrics = {
                "inv_loss": inv,
                "var_loss": var,
                "cov_loss": cov,
                **base_metrics,
            }
        elif loss_type == "smooth_l1":
            elementwise = torch.nn.functional.smooth_l1_loss(
                pred, target, beta=float(jcfg.smooth_l1_beta), reduction="none"
            )
            loss_at_position = (
                (elementwise * mask_f).sum(dim=(0, 2)) / (denom * d_model)
            )
            chunk_metrics = {
                "smooth_l1_loss": loss_at_position,
                **base_metrics,
            }
        else:  # infonce -- the thesis loss
            # One independent contrastive problem per absolute position: the
            # negatives for a game at ply j are the other games' targets at the
            # *same* ply j. Hence the Python loop rather than one batched
            # matmul -- each position has a different number of participants and
            # they must not be mixed.
            temperature = float(jcfg.contrastive_temperature)
            n_chunk = end - start
            loss_at_position = torch.zeros(n_chunk, device=x.device)
            acc_at_position = torch.zeros(n_chunk, device=x.device)
            pos_sim_at_position = torch.zeros(n_chunk, device=x.device)
            neg_sim_at_position = torch.zeros(n_chunk, device=x.device)
            for j in range(n_chunk):
                if not bool(eligible[j]):
                    continue
                sel = mask[:, j]
                pred_n = torch.nn.functional.normalize(pred[sel, j, :], dim=-1)
                tgt_n = torch.nn.functional.normalize(target[sel, j, :], dim=-1)
                logits = pred_n @ tgt_n.T / temperature
                labels = torch.arange(logits.size(0), device=logits.device)
                loss_at_position[j] = torch.nn.functional.cross_entropy(logits, labels)
                with torch.no_grad():
                    sims = pred_n @ tgt_n.T
                    diag = sims.diagonal()
                    off = ~torch.eye(sims.size(0), dtype=torch.bool, device=sims.device)
                    acc_at_position[j] = (logits.argmax(dim=1) == labels).float().mean()
                    pos_sim_at_position[j] = diag.mean()
                    neg_sim_at_position[j] = sims[off].mean()
            chunk_metrics = {
                "infonce_loss": loss_at_position,
                "positive_accuracy": acc_at_position,
                "positive_sim": pos_sim_at_position,
                "negative_sim": neg_sim_at_position,
                **base_metrics,
            }
        # Positions are weighted by how many games contributed to them, so a
        # ply reached by 250 games counts far more than one reached by 3. A
        # plain mean over positions would give the sparse, noisy late plies the
        # same influence as the dense early ones.
        eligible_weights = counts[eligible].to(torch.float32)
        chunk_weight = int(counts[eligible].sum().item())
        loss_contribution = (loss_at_position[eligible] * eligible_weights).sum()
        weighted_loss = (
            loss_contribution
            if weighted_loss is None
            else weighted_loss + loss_contribution
        )
        for key, value in chunk_metrics.items():
            contribution = (value[eligible] * eligible_weights).sum()
            weighted_metrics[key] = weighted_metrics.get(key, 0.0) + contribution
        used_pairs += chunk_weight

    if weighted_loss is None or used_pairs == 0:
        return {}, 0
    finish_profile_phase("vicreg_stats", phase_started_at)
    if profile:
        target_chunks = math.ceil(target_tokens.size(0) / chunk_size)
        timings = " ".join(
            f"{name}={seconds:.3f}s" for name, seconds in profile_times.items()
        )
        print(
            f"  all-position phase profile: pairs={used_pairs:,} "
            f"target_chunks={target_chunks} {timings}",
            flush=True,
        )
    # Divide out the accumulated weights once, at the end.
    out = {"loss": weighted_loss / used_pairs}
    out.update({key: value / used_pairs for key, value in weighted_metrics.items()})
    # Logged on the progress bar: how much supervision each game actually
    # yielded. A sudden drop means shards of unusually short games.
    out["positions_per_game"] = torch.tensor(
        used_pairs / max(1, x.size(0)), device=x.device
    )
    out["position_pairs"] = torch.tensor(float(used_pairs), device=x.device)
    return out, used_pairs

def _forward_all_position_kstep(
    raw_model: torch.nn.Module,
    x: torch.Tensor,
    *,
    pad_token: int,
    horizon: int,
    target_chunk_size: int,
    stats_chunk_size: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """All-position supervision for K-step hard-disjoint futures (v3/v4).

    Generalizes ``forward_all_position_v1`` from a single next token to a
    ``horizon``-token future window at every boundary. For boundary ``t``
    (context ``x[:t]``) the target is ``x[t:t+K]`` encoded by the target path as
    an independent length-K sequence at its original absolute positions
    (``hard_disjoint_future``): the target branch never sees the prefix. The
    predictor emits K predictions (``mlp_multi_pos``) or one prediction
    broadcast over the window (plain ``mlp``). The context encoder runs once.

    Only ``view_mode='hard_disjoint_future'`` with ``loss_type='smooth_l1'`` is
    supported here; those are the v3/v4 objectives. The return contract matches
    ``forward_all_position_v1``: ``(out, used_pairs)`` with ``out['loss']``, the
    smooth-L1 metric keys, collapse diagnostics, and ``positions_per_game``.
    A (game, boundary) pair is one supervised position, exactly as in the
    horizon-1 path; each such pair carries K future targets.

    Not on the thesis path. The reported cells all use horizon 1; this
    generalisation is kept because the v3/v4 checkpoints were produced with it
    and it documents what a multi-step latent rollout looked like in this
    codebase.
    """
    jcfg = raw_model.jepa_config
    view_mode = normalize_jepa_view_mode(jcfg.view_mode)
    loss_type = jcfg.loss_type
    if view_mode != "hard_disjoint_future":
        raise ValueError(
            "K-step all-position supervision supports view_mode='hard_disjoint_future' "
            f"only, got {view_mode!r}."
        )
    if loss_type != "smooth_l1":
        raise ValueError(
            "K-step all-position supervision supports loss_type='smooth_l1' only, "
            f"got {loss_type!r}."
        )

    K = int(horizon)
    B, T = x.shape
    device = x.device
    n_boundaries = T - K  # boundary b (0-based) uses context x[:b+1], targets b+1..b+K
    if n_boundaries <= 0:
        return {}, 0

    # One causal context pass; hidden index b is the summary for boundary b+1.
    context_hidden = raw_model.encode_hidden(raw_model.context_encoder, x)  # (B, T, d)
    d_model = context_hidden.size(-1)

    # A boundary is valid for a game iff its whole K-token future window is real.
    # unfold builds every sliding window as a view, so the all-real test is one
    # reduction rather than a loop over boundaries.
    non_pad_future = (x[:, 1:] != pad_token)                          # (B, T-1)
    windows = non_pad_future.unfold(dimension=1, size=K, step=1)      # (B, n_boundaries, K)
    valid_bt = windows.all(dim=2)                                     # (B, n_boundaries)
    valid_idx = valid_bt.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        return {}, 0
    g_idx = valid_idx[:, 0]
    b_idx = valid_idx[:, 1]
    n_pairs = g_idx.size(0)

    # Flat gather of context summaries, target tokens, and absolute positions.
    c_flat = context_hidden[g_idx, b_idx, :]                          # (N, d)
    offsets = torch.arange(1, K + 1, device=device)
    abs_positions = b_idx.unsqueeze(1) + offsets.unsqueeze(0)         # (N, K) absolute
    target_tokens = x[g_idx.unsqueeze(1), abs_positions]             # (N, K)

    # Predictor: (N, K, d) for mlp_multi_pos, (N, 1, d) for plain mlp.
    z_pred = raw_model._predict_embeddings(c_flat.unsqueeze(1), abs_positions)
    pred_k = z_pred.size(1)

    # Target: encode each K-window as a length-K hard-disjoint sequence. Chunked
    # over pairs to bound the target-encoder batch size.
    chunk = max(1, int(target_chunk_size))
    tgt_parts: list[torch.Tensor] = []
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        tgt_parts.append(
            raw_model._target_embeddings(
                target_tokens[start:end], K, abs_positions[start:end]
            )
        )
    z_tgt = torch.cat(tgt_parts, dim=0)                               # (N, K, d)

    # Smooth-L1 over every valid (game, boundary, k) element. A single-prediction
    # (plain mlp) broadcasts across the K future positions.
    pred_for_loss = z_pred.expand(-1, K, -1) if pred_k == 1 else z_pred
    beta = float(jcfg.smooth_l1_beta)
    loss = torch.nn.functional.smooth_l1_loss(pred_for_loss, z_tgt, beta=beta)

    # Per-boundary collapse diagnostics (over the game dimension so absolute
    # positions never manufacture variance across boundaries). Detached grids.
    eps = float(jcfg.variance_eps)
    pred_grid = torch.zeros(B, n_boundaries, pred_k, d_model, device=device, dtype=z_pred.dtype)
    pred_grid[g_idx, b_idx] = z_pred.detach()
    target_grid = torch.zeros(B, n_boundaries, K, d_model, device=device, dtype=z_tgt.dtype)
    target_grid[g_idx, b_idx] = z_tgt
    counts = valid_bt.sum(dim=0)                                      # (n_boundaries,)

    sums = {
        key: torch.zeros((), device=device)
        for key in ("z_pred_std", "z_tgt_std", "z_std", "c_std", "p_std", "cos_sim_offdiag")
    }
    stat_pairs = 0
    stats_chunk = max(1, int(stats_chunk_size))
    with torch.no_grad():
        for b in range(n_boundaries):
            n_b = int(counts[b].item())
            if n_b < 2:
                continue
            sel = valid_bt[:, b]
            p = pred_grid[sel, b].float()                            # (n_b, pred_k, d)
            t = target_grid[sel, b].float()                          # (n_b, K, d)
            c = context_hidden[sel, b].float()                       # (n_b, d)
            pred_var = p.var(dim=0, unbiased=False)                  # (pred_k, d)
            tgt_var = t.var(dim=0, unbiased=False)                   # (K, d)
            ctx_var = c.var(dim=0, unbiased=False)                   # (d,)
            w = float(n_b)
            sums["z_pred_std"] += torch.sqrt(pred_var + eps).mean() * w
            sums["z_tgt_std"] += torch.sqrt(tgt_var + eps).mean() * w
            sums["z_std"] += torch.sqrt(tgt_var.clamp_min(0.0)).mean() * w
            sums["p_std"] += torch.sqrt(pred_var.clamp_min(0.0)).mean() * w
            sums["c_std"] += torch.sqrt(ctx_var.clamp_min(0.0)).mean() * w
            normed = torch.nn.functional.normalize(t, dim=-1)        # (n_b, K, d)
            col_sum = normed.sum(dim=0)                              # (K, d)
            cos_off = (col_sum.pow(2).sum(dim=-1) - n_b) / (n_b * (n_b - 1))
            sums["cos_sim_offdiag"] += cos_off.mean() * w
            stat_pairs += n_b

    out: dict[str, torch.Tensor] = {"loss": loss, "smooth_l1_loss": loss.detach()}
    stat_denom = max(1, stat_pairs)
    for key, value in sums.items():
        out[key] = (value / stat_denom).detach()
    out["positions_per_game"] = torch.tensor(n_pairs / max(1, B), device=device)
    out["position_pairs"] = torch.tensor(float(n_pairs), device=device)
    if hasattr(raw_model, "target_encoder"):
        out["ema_momentum"] = torch.tensor(jcfg.ema_momentum, device=device)
    return out, n_pairs
