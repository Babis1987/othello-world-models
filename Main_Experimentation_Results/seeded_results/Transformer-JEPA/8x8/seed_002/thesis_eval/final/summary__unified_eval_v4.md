# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `a7948e0bc69fdd42ab9944424a8037928d3c89a55de739f6d4a87428d3b310d1`
- **Split manifest SHA-256:** `11b5caa28bf97a4427edae226c56379272dbb54d28e97ec5afa6acc08e39f808`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `3de85ba0c54c02ccde18de35609415525e0553c52a3a009de764d5034018cf1d`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.42% | 96.56% | 92.26% | 97.76% |
| MLP | 99.43% | 96.54% | 92.24% | 97.55% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L3 | 0.375 | 69.12% | 68.97% | 75.64% | 53.50% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.56% | 96.74% | 97.46% | 95.17% | 99.99% |
| MLP | absolute | L5 | 0.625 | 94.35% | 94.44% | 95.63% | 91.66% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.25% | 96.44% | 97.24% | 94.75% | 99.97% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.68% | 55.79% | 64.79% | 58.17% |  |
| L1 | 75.10% | 68.39% | 87.44% | 83.98% |  |
| L2 | 75.64% | 68.97% | 92.05% | 89.78% |  |
| L3 | 75.64% | 68.97% | 94.17% | 92.52% | absolute |
| L4 | 75.71% | 69.06% | 95.95% | 94.80% |  |
| L5 | 75.68% | 69.02% | 97.11% | 96.29% |  |
| L6 | 75.56% | 68.87% | 97.46% | 96.74% | relative |
| L7 | 75.09% | 68.34% | 97.03% | 96.24% |  |
| L8 | 74.80% | 68.04% | 96.64% | 95.78% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.11% | 58.75% | 65.24% | 58.61% |  |
| L1 | 84.25% | 80.13% | 86.70% | 83.11% |  |
| L2 | 89.12% | 86.15% | 91.77% | 89.42% |  |
| L3 | 91.36% | 89.00% | 93.98% | 92.28% |  |
| L4 | 94.06% | 92.44% | 95.77% | 94.58% |  |
| L5 | 95.63% | 94.44% | 96.95% | 96.08% | absolute |
| L6 | 95.15% | 93.84% | 97.24% | 96.44% | relative |
| L7 | 93.00% | 91.22% | 96.59% | 95.69% |  |
| L8 | 90.58% | 88.24% | 96.04% | 95.06% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 90.00% | 88.77% | 2.406 |
| Random direction | 90.90% | 88.50% | 2.436 |
| Relative-board direction | 94.00% | 92.72% | 0.610 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b8_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed002/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed002/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b8_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 158.9 s |
| board_probe_linear | 55.0 s |
| board_probe_mlp | 64.6 s |
| causal_intervention | 63.0 s |
| frozen_head_linear | 1571.7 s |
| frozen_head_mlp | 1386.7 s |
| head_tuning_linear | 92.3 s |
| head_tuning_mlp | 2.1 s |
