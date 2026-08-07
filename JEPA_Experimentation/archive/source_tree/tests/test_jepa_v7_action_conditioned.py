from __future__ import annotations

import math
import pickle
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from othello_research.datasets.action_grouped_dataset import (
    ActionGroupedBatchConfig,
    ActionGroupedChunkBatches,
)
from othello_research.datasets.move_mapping import build_mappings
from othello_research.objectives.jepa_action_conditioned import (
    JEPAActionConditionedConfig,
    OthelloJEPAActionConditioned,
    grouped_infonce_loss,
)
from othello_research.training import train_jepa as tjm


def _tiny_cfg(**overrides) -> JEPAActionConditionedConfig:
    values = {
        "board_size": 8,
        "n_layers": 1,
        "n_heads": 4,
        "d_model": 32,
        "dropout": 0.0,
        "predictor_hidden_mult": 2,
        "predictor_n_layers": 2,
        "predictor_dropout": 0.0,
        "action_dim": 16,
        "loss_mode": "smooth_l1",
    }
    values.update(overrides)
    return JEPAActionConditionedConfig(**values)


def test_v7_context_and_target_receive_exact_nested_views() -> None:
    model = OthelloJEPAActionConditioned(_tiny_cfg())
    context = torch.randint(0, model.config.vocab_size, (4, 5))
    target = torch.cat(
        [context, torch.randint(0, model.config.vocab_size - 1, (4, 1))],
        dim=1,
    )
    seen: dict[str, torch.Tensor] = {}
    context_handle = model.context_encoder.wte.register_forward_pre_hook(
        lambda _module, args: seen.__setitem__("context", args[0].detach().clone())
    )
    target_handle = model.target_encoder.wte.register_forward_pre_hook(
        lambda _module, args: seen.__setitem__("target", args[0].detach().clone())
    )
    try:
        model(context, target)
    finally:
        context_handle.remove()
        target_handle.remove()
    assert torch.equal(seen["context"], context)
    assert torch.equal(seen["target"], target)
    with pytest.raises(ValueError, match="exactly the context plus one"):
        model(context, target[:, :-1])


def test_v7_gradients_ema_and_action_conditioning() -> None:
    torch.manual_seed(0)
    model = OthelloJEPAActionConditioned(_tiny_cfg())
    context = torch.randint(0, model.config.vocab_size, (4, 5))
    actions = torch.tensor([1, 2, 3, 4])
    target = torch.cat([context, actions[:, None]], dim=1)
    out = model(context, target)
    assert torch.isnan(out["positive_accuracy"])
    assert torch.isnan(out["action_only_accuracy"])
    out["loss"].backward()

    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.action_embedding.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    repeated_context = out["context_latent"][0:1].detach().expand(2, -1)
    predictions = model.predict(repeated_context, torch.tensor([1, 2]))
    assert not torch.allclose(predictions[0], predictions[1])

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(0.1)
    model.update_target_encoder(momentum=0.5)
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.allclose(before, after)


def test_grouped_infonce_uniform_and_perfect_cases() -> None:
    torch.manual_seed(0)
    groups, members, dim = 3, 8, 32
    group_ids = torch.arange(groups).repeat_interleave(members)
    predictions = torch.randn(groups * members, dim)
    targets = torch.randn(groups * members, dim)
    random_loss, _ = grouped_infonce_loss(
        predictions,
        targets,
        group_ids,
        temperature=100.0,
    )
    assert abs(float(random_loss) - math.log(members)) < 0.02

    eye = torch.eye(members).repeat(groups, 1)
    perfect_loss, perfect_accuracy = grouped_infonce_loss(
        eye,
        eye,
        group_ids,
        temperature=0.01,
    )
    assert float(perfect_loss) < 1e-4
    assert float(perfect_accuracy) == 1.0


def test_grouped_infonce_vectorized_matches_reference_and_gradients() -> None:
    torch.manual_seed(1)
    groups, members, dim = 4, 5, 13
    group_ids = torch.arange(groups).repeat_interleave(members)
    predictions = torch.randn(groups * members, dim, requires_grad=True)
    targets = torch.randn(groups * members, dim, requires_grad=True)
    loss, accuracy = grouped_infonce_loss(predictions, targets, group_ids, temperature=0.2)
    loss.backward()
    prediction_grad = predictions.grad.detach().clone()
    target_grad = targets.grad.detach().clone()

    reference_predictions = predictions.detach().clone().requires_grad_(True)
    reference_targets = targets.detach().clone().requires_grad_(True)
    losses: list[torch.Tensor] = []
    accuracies: list[torch.Tensor] = []
    for group_id in range(groups):
        mask = group_ids == group_id
        logits = (
            F.normalize(reference_predictions[mask], dim=-1)
            @ F.normalize(reference_targets[mask], dim=-1).T
            / 0.2
        )
        labels = torch.arange(members)
        losses.append(F.cross_entropy(logits, labels))
        accuracies.append((logits.argmax(dim=-1) == labels).float().mean())
    reference_loss = torch.stack(losses).mean()
    reference_accuracy = torch.stack(accuracies).mean()
    reference_loss.backward()

    assert torch.allclose(loss, reference_loss, atol=1e-6)
    assert torch.allclose(accuracy, reference_accuracy)
    assert torch.allclose(prediction_grad, reference_predictions.grad, atol=1e-6)
    assert torch.allclose(target_grad, reference_targets.grad, atol=1e-6)


def test_hybrid_lambda_zero_reproduces_base_loss() -> None:
    model = OthelloJEPAActionConditioned(
        _tiny_cfg(loss_mode="hybrid", hybrid_base="grouped_infonce", lambda_ce=0.0)
    )
    context = torch.randint(0, model.config.vocab_size, (6, 5))
    target = torch.cat(
        [context, torch.randint(0, model.config.vocab_size - 1, (6, 1))],
        dim=1,
    )
    out = model(context, target, group_ids=torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(out["main_loss"], out["base_loss"])


def test_action_grouped_sampler_invariants(tmp_path: Path) -> None:
    raw_to_token, _ = build_mappings(8)
    valid_raw = [raw for raw, token in enumerate(raw_to_token) if token >= 0]
    t = 4
    games: list[list[int]] = []
    for action in valid_raw[:4]:
        for game_idx in range(3):
            prefix = [valid_raw[(game_idx + offset + 8) % len(valid_raw)] for offset in range(t)]
            games.append([*prefix, action, *valid_raw[20:24]])
    chunk = tmp_path / "games.pickle"
    with open(chunk, "wb") as handle:
        pickle.dump(games, handle)

    batches = ActionGroupedChunkBatches(
        chunk,
        ActionGroupedBatchConfig(
            board_size=8,
            t_min=t,
            t_max=t,
            groups_per_batch=4,
            samples_per_group=3,
            seed=0,
        ),
    )
    batch = next(batches)
    assert batch["x_context"].shape == (12, t)
    assert batch["x_target"].shape == (12, t + 1)
    for group_id in range(4):
        mask = batch["group_ids"] == group_id
        assert int(mask.sum()) == 3
        assert torch.unique(batch["next_actions"][mask]).numel() == 1
        assert torch.unique(batch["game_indices"][mask]).numel() == 3
        assert torch.equal(batch["x_target"][mask, -1], batch["next_actions"][mask])


def test_action_grouped_sampler_caps_chunk_to_one_game_pass(tmp_path: Path) -> None:
    raw_to_token, _ = build_mappings(8)
    valid_raw = [raw for raw, token in enumerate(raw_to_token) if token >= 0]
    games: list[list[int]] = []
    for action_idx, action in enumerate(valid_raw[:4]):
        for game_idx in range(3):
            prefix = [
                valid_raw[(action_idx * 7 + game_idx + offset + 8) % len(valid_raw)]
                for offset in range(4)
            ]
            games.append([*prefix, action, action, *valid_raw[20:24]])
    chunk = tmp_path / "multi_t_games.pickle"
    with open(chunk, "wb") as handle:
        pickle.dump(games, handle)

    batches = ActionGroupedChunkBatches(
        chunk,
        ActionGroupedBatchConfig(
            board_size=8,
            t_min=4,
            t_max=5,
            groups_per_batch=4,
            samples_per_group=3,
            chunk_sample_multiplier=1.0,
            seed=0,
        ),
    )
    emitted = list(batches)
    assert len(emitted) == 1
    assert emitted[0]["games_consumed"] == len(games)
    assert batches.remaining_batches() == 0


@pytest.mark.parametrize(
    ("config_name", "loss_mode"),
    [
        ("jepa_v7_action_smooth_l1.yml", "smooth_l1"),
        ("jepa_v7_action_infonce.yml", "grouped_infonce"),
        ("jepa_v7_action_hybrid.yml", "hybrid"),
    ],
)
def test_v7_yaml_configs_resolve_and_build(config_name: str, loss_mode: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data = tjm.load_yaml_config(repo_root / "configs" / config_name)
    parser = tjm.build_parser()
    args = parser.parse_args([
        *tjm.config_to_cli_args(data, parser),
        "--out_dir", "__unused__",
        "--n_layers", "1",
        "--n_heads", "4",
        "--d_model", "32",
        "--dropout", "0.0",
        "--action_dim", "16",
        "--predictor_hidden_mult", "2",
        "--batch_size", "8",
        "--action_groups_per_batch", "2",
        "--action_samples_per_group", "4",
        "--precision", "fp32",
        "--probe_interval_steps", "0",
        "--drive_sync_dir", "",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    assert tjm.objective_class(cfg) == "jepa_action_conditioned"
    assert cfg.variant == "v7"
    assert cfg.loss_mode == loss_mode
    assert cfg.action_loss_mode == loss_mode
    model, objective_cfg = tjm.build_objective_model(cfg)
    assert isinstance(model, OthelloJEPAActionConditioned)
    assert objective_cfg.loss_mode == loss_mode


def test_v7_grouped_batch_validation_runs_at_training_boundary() -> None:
    parser = tjm.build_parser()
    args = parser.parse_args([
        "--out_dir", "__unused__",
        "--objective_class", "jepa_action_conditioned",
        "--variant", "v7",
        "--loss_mode", "grouped_infonce",
        "--batch_size", "4",
        "--action_groups_per_batch", "16",
        "--action_samples_per_group", "16",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    assert cfg.batch_size == 4
    with pytest.raises(ValueError, match="must equal"):
        tjm.validate_train_config(cfg)


def test_grouped_training_dispatch_runs_optimizer_and_ema_step(tmp_path: Path) -> None:
    raw_to_token, _ = build_mappings(8)
    valid_raw = [raw for raw, token in enumerate(raw_to_token) if token >= 0]
    games: list[list[int]] = []
    for action in valid_raw[:4]:
        for game_idx in range(4):
            prefix = [valid_raw[(game_idx + offset + 10) % len(valid_raw)] for offset in range(4)]
            games.append([*prefix, action, *valid_raw[24:30]])
    chunk = tmp_path / "grouped_train.pickle"
    with open(chunk, "wb") as handle:
        pickle.dump(games, handle)

    cfg = tjm.TrainConfig(
        out_dir="__unused__",
        objective_class="jepa_action_conditioned",
        variant="v7",
        predictor_type="action_conditioned_mlp",
        n_layers=1,
        n_heads=4,
        d_model=32,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        predictor_dropout=0.0,
        action_dim=16,
        action_loss_mode="grouped_infonce",
        loss_type="grouped_infonce",
        prediction_horizon=1,
        view_mode="nested",
        action_t_min=4,
        action_t_max=4,
        action_groups_per_batch=2,
        action_samples_per_group=2,
        batch_size=4,
        warmup_steps=1,
        precision="fp32",
        probe_interval_steps=0,
    )
    model, _ = tjm.build_objective_model(cfg)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    target_before = next(model.target_encoder.parameters()).detach().clone()
    result = tjm.dispatch_train_one_chunk(
        model,
        optimizer,
        scaler,
        chunk,
        cfg,
        model.config,
        torch.device("cpu"),
        0,
        10,
    )
    train_loss, metrics, step = result[:3]
    assert step > 0
    assert math.isfinite(train_loss)
    assert math.isfinite(metrics["context_utility"])
    assert not torch.allclose(
        target_before,
        next(model.target_encoder.parameters()).detach(),
    )


EXISTING_V1_TO_V6_CONFIGS = [
    "jepa_linear_k1_ema_b8_hard_disjoint.yml",
    "jepa_no_predictor_k1_vicreg_b8_hard_disjoint.yml",
    "jepa_v1_vicreg_b8.yml",
    "jepa_v1_vicreg_b8_hard_disjoint.yml",
    "jepa_v2_ema_b8.yml",
    "jepa_v2_ema_b8_hard_disjoint.yml",
    "jepa_v3_mlp_k8_b8.yml",
    "jepa_v4_multi_pos_mlp_k4_b8.yml",
    "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint.yml",
    "jepa_v4_multi_pos_mlp_k4_b8_hard_disjoint_vicreg.yml",
    "jepa_v5_contrastive.yml",
    "jepa_v5_contrastive_hard_disjoint.yml",
    "jepa_v6_hard_disjoint_action.yml",
]


@pytest.mark.parametrize("config_name", EXISTING_V1_TO_V6_CONFIGS)
def test_every_existing_v1_to_v6_config_still_runs_one_step(
    config_name: str,
    synthetic_game_batch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data = tjm.load_yaml_config(repo_root / "configs" / config_name)
    parser = tjm.build_parser()
    args = parser.parse_args([
        *tjm.config_to_cli_args(data, parser),
        "--out_dir", "__unused__",
        "--n_layers", "1",
        "--n_heads", "4",
        "--d_model", "32",
        "--dropout", "0.0",
        "--predictor_hidden_dim", "32",
        "--predictor_hidden_mult", "2",
        "--predictor_n_layers", "2",
        "--predictor_n_heads", "4",
        "--predictor_dropout", "0.0",
        "--batch_size", "4",
        "--precision", "fp32",
        "--eval_chunks", "0",
        "--drive_sync_dir", "",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    model, _ = tjm.build_objective_model(cfg)
    batch = synthetic_game_batch(B=4, T=24, board_size=8, seed=1)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)

    if tjm.objective_class(cfg) in {
        "jepa_hard_disjoint_action",
        "jepa_hard_disjoint_infonce",
    }:
        prefix_ids = torch.tensor([0, 0, 1, 1])
        x_context = batch[:, :6].clone()
        x_context[1] = x_context[0]
        x_context[3] = x_context[2]
        out = model(x_context, batch[:, 6], prefix_ids)
    else:
        horizon = cfg.prediction_horizon
        x_context, x_target, target_positions = tjm.build_jepa_views(
            cfg,
            batch,
            context_length=6,
            horizon=horizon,
        )
        out = model(x_context, x_target, target_positions)
    out["loss"].backward()
    optimizer.step()
    assert torch.isfinite(out["loss"])
