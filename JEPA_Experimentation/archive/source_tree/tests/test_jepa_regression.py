from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from othello_research.training import train_jepa as tjm


REFERENCE_FILE = Path(__file__).parent / "regression_references.json"

CONFIGS_TO_TEST = [
    "jepa_v1_vicreg_b8.yml",
    "jepa_v2_ema_b8.yml",
    "jepa_v3_mlp_k8_b8.yml",
    "jepa_v4_multi_pos_mlp_k4_b8.yml",
    "jepa_v5_contrastive.yml",
]


def _load_references() -> dict[str, float]:
    if REFERENCE_FILE.exists():
        return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_reference(key: str, value: float) -> None:
    refs = _load_references()
    refs[key] = value
    REFERENCE_FILE.write_text(json.dumps(refs, indent=2, sort_keys=True), encoding="utf-8")


def _train_config_from_yaml(config_name: str) -> tjm.TrainConfig:
    yaml_path = Path(__file__).parents[1] / "configs" / config_name
    data = tjm.load_yaml_config(yaml_path)
    parser = tjm.build_parser()
    args = parser.parse_args([
        *tjm.config_to_cli_args(data, parser),
        "--out_dir", "__unused_regression_out__",
        "--n_layers", "2",
        "--n_heads", "4",
        "--d_model", "64",
        "--dropout", "0.0",
        "--predictor_hidden_dim", "64",
        "--predictor_hidden_mult", "2",
        "--predictor_n_layers", "2",
        "--predictor_n_heads", "4",
        "--predictor_dropout", "0.0",
        "--batch_size", "8",
        "--precision", "fp32",
        "--eval_chunks", "0",
        "--drive_sync_dir", "",
    ])
    cfg, _ = tjm.namespace_to_train_config(args)
    return cfg


@pytest.mark.parametrize("config_name", CONFIGS_TO_TEST)
def test_loss_regression(config_name, synthetic_game_batch):
    torch.manual_seed(42)
    cfg = _train_config_from_yaml(config_name)
    model, _ = tjm.build_objective_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = synthetic_game_batch(B=8, T=30, board_size=8, seed=42)
    t = 10
    K = cfg.prediction_horizon
    target_positions = torch.arange(t, t + K).unsqueeze(0).expand(8, K)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = model(batch[:, :t], batch[:, : t + K], target_positions)
        out["loss"].backward()
        optimizer.step()
        tjm.update_target_encoder(model)
        losses.append(float(out["loss"].detach()))

    value = losses[-1]
    key = f"{config_name}::loss_at_step_5"
    if os.environ.get("PYTEST_RECORD_REFERENCE"):
        _save_reference(key, value)
        pytest.skip(f"Recorded reference for {key}: {value}")

    refs = _load_references()
    assert key in refs, (
        f"No reference recorded for {key}. "
        "Run with PYTEST_RECORD_REFERENCE=1 pytest tests/test_jepa_regression.py -v"
    )
    expected = refs[key]
    assert abs(value - expected) / max(abs(expected), 1e-8) < 1e-4
