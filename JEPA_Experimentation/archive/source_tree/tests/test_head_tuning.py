from __future__ import annotations

import json
from pathlib import Path

import torch

from othello_research.evaluation.head_tuning import (
    CachedHeadDataset,
    HeadTuningConfig,
    default_parameter_grid,
    fit_cached_head,
    grouped_grid_search,
    make_group_kfold_splits,
    phase_stratified_positions,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_cache() -> CachedHeadDataset:
    generator = torch.Generator().manual_seed(7)
    n_samples = 48
    hidden_dim = 6
    vocab_size = 3
    targets = torch.arange(n_samples) % vocab_size
    features = torch.randn(
        n_samples,
        hidden_dim,
        generator=generator,
    ) * 0.05
    features[torch.arange(n_samples), targets] += 2.0
    legal_mask = torch.zeros(n_samples, vocab_size, dtype=torch.bool)
    legal_mask[torch.arange(n_samples), targets] = True
    legal_mask[torch.arange(n_samples), (targets + 1) % vocab_size] = True
    result = CachedHeadDataset(
        features=features.to(torch.float16),
        targets=targets.long(),
        legal_mask=legal_mask,
        shard_ids=torch.arange(n_samples) // 8,
        game_ids=torch.arange(n_samples) // 2,
        phases=torch.linspace(0.05, 1.0, n_samples),
        metadata={"identity": {"synthetic": True}},
    )
    result.validate()
    return result


def test_phase_sampler_is_unique_and_spans_the_game() -> None:
    assert phase_stratified_positions(1, 6) == (0,)
    assert phase_stratified_positions(6, 6) == (0, 1, 2, 3, 4, 5)
    positions = phase_stratified_positions(59, 6)
    assert len(positions) == len(set(positions)) == 6
    assert positions[0] < 10
    assert positions[-1] > 49


def test_full_grid_is_bounded_and_contains_only_requested_heads() -> None:
    grid = default_parameter_grid("full")
    linear = [row for row in grid if row["head_type"] == "linear"]
    mlp = [row for row in grid if row["head_type"] == "mlp"]
    assert len(linear) == 12
    assert len(mlp) == 12
    assert len({row["id"] for row in grid}) == len(grid)
    assert {row["head_type"] for row in grid} == {"linear", "mlp"}
    assert {row["regularization"] for row in linear} == {
        "none",
        "l1",
        "l2",
    }
    assert all(
        row["l1_strength"] > 0
        for row in linear
        if row["regularization"] == "l1"
    )
    assert all(
        row["weight_decay"] > 0
        for row in linear
        if row["regularization"] == "l2"
    )


def test_group_kfold_never_splits_a_shard() -> None:
    cache = _synthetic_cache()
    folds = make_group_kfold_splits(cache.shard_ids, 3, seed=42)
    validation_samples = []
    for train_indices, validation_indices in folds:
        train_groups = set(cache.shard_ids[train_indices].tolist())
        validation_groups = set(cache.shard_ids[validation_indices].tolist())
        assert train_groups.isdisjoint(validation_groups)
        validation_samples.extend(validation_indices.tolist())
    assert sorted(validation_samples) == list(range(cache.n_samples))


def test_linear_l1_regularization_is_applied_and_logged() -> None:
    cache = _synthetic_cache()
    config = HeadTuningConfig(
        cv_folds=2,
        train_games=12,
        selection_games=4,
        test_games=4,
        positions_per_game=2,
        head_batch_size=16,
        max_epochs=1,
        patience=1,
        bootstrap_resamples=10,
    )
    head, fit = fit_cached_head(
        cache,
        torch.arange(32),
        cache,
        torch.arange(32, 48),
        {
            "id": "linear_l1_smoke",
            "head_type": "linear",
            "learning_rate": 0.05,
            "regularization": "l1",
            "weight_decay": 0.0,
            "l1_strength": 1e-3,
        },
        config=config,
        device="cpu",
        seed=11,
    )
    assert fit["history"][0]["train_regularization_penalty"] > 0
    assert (
        fit["history"][0]["train_objective"]
        > fit["history"][0]["train_cross_entropy"]
    )
    del head


def test_grouped_grid_search_runs_linear_and_mlp_on_cached_features() -> None:
    cache = _synthetic_cache()
    config = HeadTuningConfig(
        cv_folds=2,
        train_games=12,
        selection_games=4,
        test_games=4,
        positions_per_game=2,
        head_batch_size=16,
        max_epochs=3,
        patience=2,
        bootstrap_resamples=10,
    )
    grid = [
        {
            "id": "linear_smoke",
            "head_type": "linear",
            "learning_rate": 0.05,
            "weight_decay": 0.0,
        },
        {
            "id": "mlp_smoke",
            "head_type": "mlp",
            "hidden_dim": 8,
            "dropout": 0.0,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
        },
    ]
    result = grouped_grid_search(
        cache,
        grid,
        config=config,
        device="cpu",
        progress=lambda _message: None,
    )
    assert result["complete"] is True
    assert set(result["best_by_head"]) == {"linear", "mlp"}
    assert result["best_by_head"]["linear"]["params"]["id"] == "linear_smoke"
    assert result["best_by_head"]["mlp"]["params"]["id"] == "mlp_smoke"
    for head_type in ("linear", "mlp"):
        assert len(result["results_by_head"][head_type][0]["folds"]) == 2


def test_generated_notebook_has_gpu_cv_and_report_workflow() -> None:
    path = ROOT / "notebooks" / "thesis_8x8_jepa_head_tuning.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert 'MODEL_SET = "Both"' in source
    assert 'PROFILE = "Full"' in source
    assert "Transformer-JEPA" in source
    assert "Mamba-JEPA" in source
    assert "run_head_tuning_case(" in source
    assert "write_head_tuning_report(" in source
    assert "mamba-ssm" in source
    assert "cv_search.json" in source
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile(
                "".join(cell.get("source", [])),
                f"{path}:cell-{index}",
                "exec",
            )
