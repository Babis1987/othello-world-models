from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from othello_research.evaluation.linear_head import JEPALinearHead
from othello_research.objectives.v9_multiaction_jepa import (
    JEPAMultiActionConfig,
    OthelloJEPAMultiActionRollout,
)
from othello_research.training import train_jepa as tjm


def _tiny_cfg(**overrides) -> JEPAMultiActionConfig:
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
        "action_horizon": 3,
        "prediction_horizon": 3,
        "rollout_loss_mode": "smooth_l1",
        "loss_type": "smooth_l1",
        "lambda_var": 0.5,
        "lambda_cov": 0.05,
    }
    values.update(overrides)
    return JEPAMultiActionConfig(**values)


def _notebook_cell_source(name: str, marker: str) -> str:
    path = Path(__file__).resolve().parents[1] / "notebooks" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if marker in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1, (name, marker, len(matches))
    return matches[0]


def test_v9_context_target_views_and_recursive_gradients() -> None:
    torch.manual_seed(0)
    model = OthelloJEPAMultiActionRollout(_tiny_cfg())
    context = torch.randint(0, model.config.vocab_size, (4, 5))
    future = torch.randint(0, model.config.vocab_size - 1, (4, 3))
    target = torch.cat([context, future], dim=1)

    seen: dict[str, torch.Tensor] = {}
    context_handle = model.context_encoder.wte.register_forward_pre_hook(
        lambda _module, args: seen.__setitem__("context", args[0].detach().clone())
    )
    target_handle = model.target_encoder.wte.register_forward_pre_hook(
        lambda _module, args: seen.__setitem__("target", args[0].detach().clone())
    )
    try:
        out = model(context, target)
    finally:
        context_handle.remove()
        target_handle.remove()

    assert torch.equal(seen["context"], context)
    assert torch.equal(seen["target"], target)
    assert out["predictions"].shape == (4, 3, model.config.d_model)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["variance_loss"])
    assert torch.isfinite(out["covariance_loss"])

    out["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.action_embedding.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    repeated_context = out["context_latent"][0:1].expand(2, -1)
    predictions = model.predict(repeated_context, torch.tensor([1, 2]))
    assert not torch.allclose(predictions[0], predictions[1])

    with pytest.raises(ValueError, match="context plus action_horizon"):
        model(context, target[:, :-1])


def test_v9_infonce_rollout_reports_retrieval_metrics() -> None:
    torch.manual_seed(1)
    model = OthelloJEPAMultiActionRollout(
        _tiny_cfg(
            rollout_loss_mode="infonce",
            loss_type="infonce",
            lambda_var=0.0,
            lambda_cov=0.0,
            normalize_latents_for_loss=True,
        )
    )
    context = torch.randint(0, model.config.vocab_size, (6, 5))
    future = torch.randint(0, model.config.vocab_size - 1, (6, 3))
    out = model(context, torch.cat([context, future], dim=1))

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["infonce_loss"])
    assert 0.0 <= float(out["positive_accuracy"]) <= 1.0
    assert torch.isfinite(out["positive_sim"])
    assert torch.isfinite(out["negative_sim"])
    for step_idx in range(1, 4):
        assert torch.isfinite(out[f"step_{step_idx}_loss"])
        assert torch.isfinite(out[f"step_{step_idx}_weighted_loss"])
        assert torch.isfinite(out[f"step_{step_idx}_positive_accuracy"])

    out["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


@pytest.mark.parametrize(
    ("config_name", "mode"),
    [
        ("jepa_v9a_multiaction_smooth_l1.yml", "smooth_l1"),
        ("jepa_v9b_multiaction_infonce.yml", "infonce"),
    ],
)
def test_v9_yaml_configs_resolve_build_and_forward(
    config_name: str,
    mode: str,
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
        "--action_dim", "16",
        "--predictor_hidden_mult", "2",
        "--predictor_n_layers", "2",
        "--batch_size", "6",
        "--action_horizon", "2",
        "--prediction_horizon", "2",
        "--precision", "fp32",
        "--drive_sync_dir", "",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    tjm.validate_train_config(cfg)
    assert tjm.objective_class(cfg) == "jepa_multiaction"
    assert cfg.variant == "v9"
    assert cfg.rollout_loss_mode == mode
    assert cfg.action_horizon == 2
    assert cfg.prediction_horizon == 2

    model, objective_cfg = tjm.build_objective_model(cfg)
    assert isinstance(model, OthelloJEPAMultiActionRollout)
    assert objective_cfg.rollout_loss_mode == mode

    batch = synthetic_game_batch(B=6, T=16, board_size=8, seed=3)
    x_context, x_target, target_positions = tjm.build_jepa_views(
        cfg,
        batch,
        context_length=5,
        horizon=cfg.action_horizon,
    )
    assert target_positions is None
    assert x_context.shape == (6, 5)
    assert x_target.shape == (6, 7)
    out = model(x_context, x_target, target_positions)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_v9_checkpoint_rebuild_supports_evaluation_paths() -> None:
    cfg = _tiny_cfg(action_horizon=2, prediction_horizon=2)
    model = OthelloJEPAMultiActionRollout(cfg)
    checkpoint = {
        "model": model.state_dict(),
        "model_config": {"objective_class": "jepa_multiaction", **asdict(cfg)},
        "jepa_config": asdict(cfg),
        "train_config": {
            "objective_class": "jepa_multiaction",
            "variant": "v9",
        },
        "step": 1,
        "games_seen": 8,
    }

    loaded, loaded_cfg = tjm.build_model_from_checkpoint(checkpoint)
    loaded.load_state_dict(checkpoint["model"])
    loaded.eval()

    assert loaded_cfg.variant == "v9"
    assert hasattr(loaded, "context_encoder")
    tokens = torch.randint(0, loaded.config.vocab_size, (2, 6))
    hidden = loaded.encode_hidden(loaded.context_encoder, tokens)
    assert hidden.shape == (2, 6, loaded.config.d_model)

    head = JEPALinearHead(loaded)
    logits, loss = head(tokens, tokens)
    assert logits.shape == (2, 6, loaded.config.vocab_size)
    assert torch.isfinite(loss)


def test_evaluation_v2_skips_action_free_predictor_space_for_v9() -> None:
    model = OthelloJEPAMultiActionRollout(_tiny_cfg(action_horizon=2, prediction_horizon=2))
    helpers_source = _notebook_cell_source(
        "othello_gpt_jepa_evaluation_v2.ipynb",
        "# Predictor-space helpers. These do not modify or fine-tune JEPA.",
    )
    namespace = {
        "jepa_model": model,
        "device": torch.device("cpu"),
    }
    exec(compile(helpers_source, "v9-predictor-space-helpers", "exec"), namespace)
    assert namespace["PREDICTOR_SPACE_SUPPORTED"] is False
    assert "action embedding" in namespace["PREDICTOR_SPACE_SKIP_REASON"]
