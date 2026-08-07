"""Smoke tests for JEPA v6 hard-disjoint action objective.

These tests are intentionally small: they use tiny models and synthetic
chunks so they can be run before launching a real Colab training run.
"""

from __future__ import annotations

import pickle
import random
import tempfile
from pathlib import Path

import torch

from othello_research.datasets.branch_prefix_dataset import (
    BranchPrefixBatchConfig,
    BranchPrefixChunkBatches,
)
from othello_research.datasets.move_mapping import build_mappings
from othello_research.objectives.jepa_hard_disjoint_action import (
    JEPAHardDisjointActionConfig,
    MultiModeMLPPredictor,
    OthelloJEPAHardDisjointAction,
    hard_disjoint_action_loss,
    select_phantom_token,
)
from othello_research.training import train_jepa as tjm


def _tiny_cfg() -> JEPAHardDisjointActionConfig:
    return JEPAHardDisjointActionConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        predictor_dropout=0.0,
        num_modes=4,
    )


def test_multi_mode_predictor_shape() -> None:
    predictor = MultiModeMLPPredictor(
        d_model=32,
        num_modes=5,
        hidden_mult=2,
        n_layers=2,
        dropout=0.0,
    )
    out = predictor(torch.randn(7, 32))
    assert out.shape == (7, 5, 32)


def test_phantom_token_selection_uses_batch_unused_token() -> None:
    actions = torch.tensor([0, 1, 2, 3])
    chosen = select_phantom_token(actions, vocab_actions=10)
    assert chosen is not None
    assert chosen not in actions.tolist()
    assert select_phantom_token(torch.arange(5), vocab_actions=5) is None


def test_v6_forward_backward_target_frozen_and_ema_updates() -> None:
    torch.manual_seed(0)
    model = OthelloJEPAHardDisjointAction(_tiny_cfg())
    batch_size = 8
    context_len = 4
    vocab = model.config.vocab_size

    x_context = torch.randint(0, vocab, (batch_size, context_len))
    prefix_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    x_context[:4] = x_context[0]
    x_context[4:] = x_context[4]

    next_actions = torch.randint(0, model._vocab_actions, (batch_size,))
    next_actions[4] = next_actions[0]  # different-prefix action collision

    out = model(x_context, next_actions, prefix_ids)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    assert float(out["dedup_fraction"]) > 0.0
    assert "c_std" in out
    assert torch.isfinite(out["c_std"])

    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        for param in model.context_encoder.parameters():
            param.add_(0.01)
    model.update_target_encoder(momentum=0.5)
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.allclose(before, after)


def test_loss_phantom_replacement_changes_only_push_path() -> None:
    torch.manual_seed(1)
    batch_size, modes, d_model = 6, 3, 8
    p = torch.randn(batch_size, modes, d_model)
    z = torch.randn(batch_size, d_model)
    prefix_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    next_actions = torch.tensor([10, 11, 10, 14, 15, 16])
    z_phantom = torch.randn(d_model)

    dedup = hard_disjoint_action_loss(
        p,
        z,
        prefix_ids,
        next_actions,
        z_phantom,
        margin=1.0,
        lambda_push=1.0,
        normalize=True,
        dedup_negative_actions=True,
    )
    no_dedup = hard_disjoint_action_loss(
        p,
        z,
        prefix_ids,
        next_actions,
        z_phantom,
        margin=1.0,
        lambda_push=1.0,
        normalize=True,
        dedup_negative_actions=False,
    )

    assert torch.allclose(dedup["pull_loss"], no_dedup["pull_loss"])
    assert not torch.allclose(dedup["push_loss"], no_dedup["push_loss"])
    assert float(dedup["dedup_fraction"]) > 0.0


def _write_synthetic_chunk(path: Path, n_games: int = 240, board_size: int = 8) -> None:
    rng = random.Random(0)
    raw_to_token, _ = build_mappings(board_size)
    valid_raw = [raw for raw, token in enumerate(raw_to_token) if token != -1]
    seed_prefixes = [tuple(rng.sample(valid_raw, k=6)) for _ in range(20)]

    games: list[list[int]] = []
    for _ in range(n_games):
        prefix = list(rng.choice(seed_prefixes))
        suffix = rng.sample(valid_raw, k=20)
        games.append(prefix + suffix)

    with open(path, "wb") as handle:
        pickle.dump(games, handle)


def test_branch_prefix_batches_yield_grouped_contexts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        chunk_path = Path(tmp) / "synthetic.pickle"
        _write_synthetic_chunk(chunk_path)

        cfg = BranchPrefixBatchConfig(
            board_size=8,
            t_min=4,
            t_max=6,
            groups_per_batch=4,
            samples_per_group=3,
            seed=42,
        )
        batcher = BranchPrefixChunkBatches(chunk_path, cfg)
        batch = next(batcher)

        expected_batch = cfg.groups_per_batch * cfg.samples_per_group
        assert batch["x_context"].shape == (expected_batch, batch["t"])
        assert batch["next_actions"].shape == (expected_batch,)
        assert batch["prefix_ids"].shape == (expected_batch,)

        for group_id in range(cfg.groups_per_batch):
            members = batch["x_context"][batch["prefix_ids"] == group_id]
            assert (members == members[0:1]).all()


def test_train_jepa_v6_yaml_dispatch_builds_tiny_model() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "configs" / "jepa_v6_hard_disjoint_action.yml"
    data = tjm.load_yaml_config(yaml_path)
    parser = tjm.build_parser()
    args = parser.parse_args(
        [
            *tjm.config_to_cli_args(data, parser),
            "--out_dir",
            "__unused__",
            "--n_layers",
            "2",
            "--n_heads",
            "4",
            "--d_model",
            "64",
            "--dropout",
            "0.0",
            "--branch_groups_per_batch",
            "2",
            "--branch_samples_per_group",
            "2",
            "--hard_disjoint_num_modes",
            "3",
            "--predictor_hidden_mult",
            "2",
        ]
    )
    cfg, _ = tjm.namespace_to_train_config(args)
    assert tjm.objective_class(cfg) == "jepa_hard_disjoint_action"
    assert "pull_loss" in tjm.objective_metric_keys(cfg)
    assert "dedup_fraction" in tjm.objective_metric_keys(cfg)

    model, objective_cfg = tjm.build_objective_model(cfg)
    assert isinstance(model, OthelloJEPAHardDisjointAction)
    assert isinstance(objective_cfg, JEPAHardDisjointActionConfig)


def _tiny_v6_train_config(chunk_dir: Path, *, position_sampling: str) -> "tjm.TrainConfig":
    return tjm.TrainConfig(
        out_dir=str(chunk_dir / "out"),
        objective_class="jepa_hard_disjoint_action",
        view_mode="hard_disjoint_action",
        position_sampling=position_sampling,
        board_size=8,
        use_ema_target=True,
        ema_momentum=0.996,
        hard_disjoint_num_modes=3,
        branch_t_min=4,
        branch_t_max=6,
        branch_groups_per_batch=4,
        branch_samples_per_group=3,
        batch_size=12,
        learning_rate=3e-4,
        warmup_steps=0,
        grad_clip=1.0,
        precision="fp32",
        seed=42,
    )


def test_train_one_chunk_hard_disjoint_all_positions_dispatch() -> None:
    """The all-position branch of train_one_chunk_hard_disjoint must run.

    Drives the real dispatch (sampler -> forward_all_positions -> loss ->
    optimizer step -> EMA update) on a synthetic chunk, exercising the wiring
    added in train_jepa.py rather than only the objective method.
    """
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chunk_path = tmp_path / "synthetic.pickle"
        _write_synthetic_chunk(chunk_path)

        model = OthelloJEPAHardDisjointAction(_tiny_cfg())
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        cfg = _tiny_v6_train_config(tmp_path, position_sampling="all")

        before = next(model.target_encoder.parameters()).detach().clone()
        (
            avg_loss, metrics, step, n_games, total_samples,
            seen_batches, skipped_batches, seen_samples, skipped_samples,
        ) = tjm.train_one_chunk_hard_disjoint(
            model, optimizer, scaler, chunk_path, cfg, model.config, device,
            step=0, total_steps=10,
        )

        assert seen_batches > 0
        assert total_samples > 0
        assert step == seen_batches  # one optimizer step per batch
        import math
        assert math.isfinite(avg_loss)
        for key in ("pull_loss", "push_loss", "dedup_fraction"):
            assert key in metrics and math.isfinite(metrics[key])
        # EMA target must have moved after the optimizer steps.
        after = next(model.target_encoder.parameters()).detach()
        assert not torch.allclose(before, after)


def test_existing_v5_yaml_still_resolves_as_contrastive() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "configs" / "jepa_v5_contrastive.yml"
    data = tjm.load_yaml_config(yaml_path)
    parser = tjm.build_parser()
    args = parser.parse_args([*tjm.config_to_cli_args(data, parser), "--out_dir", "__unused__"])
    cfg, _ = tjm.namespace_to_train_config(args)
    assert tjm.objective_class(cfg) == "jepa_contrastive"
    assert "positive_accuracy" in tjm.objective_metric_keys(cfg)
    assert "pull_loss" not in tjm.objective_metric_keys(cfg)
