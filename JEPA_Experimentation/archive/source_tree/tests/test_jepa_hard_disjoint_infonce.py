from pathlib import Path

import torch

from othello_research.objectives.jepa_hard_disjoint_infonce import (
    JEPAHardDisjointInfoNCEConfig,
    OthelloJEPAHardDisjointInfoNCE,
    hard_disjoint_infonce_loss,
)
from othello_research.training import train_jepa as tjm


def _tiny_cfg() -> JEPAHardDisjointInfoNCEConfig:
    return JEPAHardDisjointInfoNCEConfig(
        board_size=8,
        n_layers=2,
        n_heads=4,
        d_model=64,
        dropout=0.0,
        predictor_hidden_mult=2,
        predictor_n_layers=2,
        predictor_dropout=0.0,
    )


def test_hard_disjoint_infonce_forward_backward_target_frozen() -> None:
    torch.manual_seed(0)
    model = OthelloJEPAHardDisjointInfoNCE(_tiny_cfg())
    x_context = torch.randint(0, model.config.vocab_size, (8, 5))
    prefix_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    x_context[:4] = x_context[0]
    x_context[4:] = x_context[4]
    next_actions = torch.randint(0, model._vocab_actions, (8,))
    next_actions[4] = next_actions[0]

    out = model(x_context, next_actions, prefix_ids)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert float(out["n_positives_mean"]) == 4.0
    assert float(out["dedup_fraction"]) > 0.0
    assert "c_std" in out and torch.isfinite(out["c_std"])

    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_phantom_replacement_changes_infonce_loss_when_action_collides() -> None:
    torch.manual_seed(1)
    predictions = torch.randn(6, 16)
    targets = torch.randn(6, 16)
    z_phantom = torch.randn(16)
    prefix_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    next_actions = torch.tensor([10, 11, 10, 14, 15, 16], dtype=torch.long)

    dedup = hard_disjoint_infonce_loss(
        predictions,
        targets,
        prefix_ids,
        next_actions,
        z_phantom,
        temperature=0.1,
        dedup_negative_actions=True,
    )
    no_dedup = hard_disjoint_infonce_loss(
        predictions,
        targets,
        prefix_ids,
        next_actions,
        z_phantom,
        temperature=0.1,
        dedup_negative_actions=False,
    )

    assert float(dedup["dedup_fraction"]) > 0.0
    assert not torch.allclose(dedup["loss"], no_dedup["loss"])


def test_v5_hard_disjoint_infonce_yaml_dispatch_builds_model() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data = tjm.load_yaml_config(
        repo_root / "configs" / "jepa_v5_contrastive_hard_disjoint.yml"
    )
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
            "--predictor_hidden_mult",
            "2",
        ]
    )
    cfg, _ = tjm.namespace_to_train_config(args)
    assert tjm.objective_class(cfg) == "jepa_hard_disjoint_infonce"
    assert cfg.variant == "v5"
    assert cfg.loss_type == "infonce"
    assert cfg.view_mode == "hard_disjoint_action"
    assert "infonce_loss" in tjm.objective_metric_keys(cfg)
    assert "dedup_fraction" in tjm.objective_metric_keys(cfg)

    model, objective_cfg = tjm.build_objective_model(cfg)
    assert isinstance(model, OthelloJEPAHardDisjointInfoNCE)
    assert isinstance(objective_cfg, JEPAHardDisjointInfoNCEConfig)
    assert objective_cfg.variant == "jepa_v5_hard_disjoint_infonce"
