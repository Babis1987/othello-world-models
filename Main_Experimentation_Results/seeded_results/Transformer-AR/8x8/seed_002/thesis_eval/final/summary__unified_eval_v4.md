# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `25965b0c96cdfb3b2784d51773c992df7cba26d9f0fc0cf2c878ae1f9ab9fcad`
- **Split manifest SHA-256:** `0d867d4a1947a87a16ab37e22e3f09aa99486de676109ed5776bf3ecae795513`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `3de85ba0c54c02ccde18de35609415525e0553c52a3a009de764d5034018cf1d`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.70% | 96.76% | 92.43% | 98.47% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L3 | 0.375 | 69.03% | 68.91% | 75.60% | 53.41% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.83% | 97.05% | 97.71% | 95.63% | 99.99% |
| MLP | absolute | L5 | 0.625 | 94.22% | 94.36% | 95.57% | 91.55% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.53% | 96.77% | 97.50% | 95.24% | 99.99% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.70% | 55.78% | 64.81% | 58.20% |  |
| L1 | 75.19% | 68.48% | 87.33% | 83.81% |  |
| L2 | 75.59% | 68.90% | 92.13% | 89.89% |  |
| L3 | 75.60% | 68.91% | 94.55% | 93.00% | absolute |
| L4 | 75.56% | 68.87% | 96.06% | 94.95% |  |
| L5 | 75.56% | 68.86% | 97.23% | 96.46% |  |
| L6 | 75.45% | 68.73% | 97.71% | 97.05% | relative |
| L7 | 75.22% | 68.45% | 97.63% | 96.97% |  |
| L8 | 75.17% | 68.39% | 97.54% | 96.86% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.11% | 58.74% | 65.22% | 58.55% |  |
| L1 | 84.02% | 79.80% | 86.64% | 82.95% |  |
| L2 | 89.50% | 86.64% | 91.84% | 89.47% |  |
| L3 | 92.39% | 90.31% | 94.36% | 92.75% |  |
| L4 | 93.99% | 92.35% | 95.92% | 94.76% |  |
| L5 | 95.57% | 94.36% | 97.11% | 96.27% | absolute |
| L6 | 94.41% | 92.88% | 97.50% | 96.77% | relative |
| L7 | 90.26% | 87.62% | 97.37% | 96.63% |  |
| L8 | 85.68% | 81.80% | 97.21% | 96.44% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `4`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.90% | 89.70% | 2.276 |
| Random direction | 90.00% | 89.58% | 2.296 |
| Relative-board direction | 94.80% | 95.61% | 0.470 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b8_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b8_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 160.3 s |
| board_probe_linear | 55.3 s |
| board_probe_mlp | 64.1 s |
| causal_intervention | 62.8 s |
| native_ar | 1097.4 s |
