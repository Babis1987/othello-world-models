# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `1d7e725418192720c9ecc3118869efa46170ed5f79ccc2139f9fffb17380cb2a`
- **Split manifest SHA-256:** `f2e3e25a0870aa50c586098aad5a4736d4a9d539f6eadaa5d4868ba790bb36a3`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `3de85ba0c54c02ccde18de35609415525e0553c52a3a009de764d5034018cf1d`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.82% | 96.88% | 92.55% | 99.30% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 68.79% | 68.68% | 75.42% | 53.08% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 97.72% | 97.96% | 98.41% | 96.97% | 100.00% |
| MLP | absolute | L11 | 0.733 | 86.68% | 86.71% | 89.56% | 80.07% | 100.00% |
| MLP | relative | L12 | 0.800 | 97.48% | 97.77% | 98.28% | 96.73% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.63% | 55.60% | 64.80% | 58.19% |  |
| L1 | 73.91% | 67.07% | 78.81% | 73.09% |  |
| L2 | 72.81% | 65.98% | 88.49% | 85.80% |  |
| L3 | 73.51% | 66.65% | 89.74% | 87.21% |  |
| L4 | 73.87% | 66.96% | 91.12% | 88.81% |  |
| L5 | 74.18% | 67.29% | 92.21% | 90.14% |  |
| L6 | 74.25% | 67.36% | 93.25% | 91.47% |  |
| L7 | 74.93% | 68.13% | 94.31% | 92.76% |  |
| L8 | 74.70% | 67.86% | 96.30% | 95.32% |  |
| L9 | 74.97% | 68.15% | 96.86% | 96.01% |  |
| L10 | 75.40% | 68.66% | 97.53% | 96.83% |  |
| L11 | 75.47% | 68.74% | 97.88% | 97.29% |  |
| L12 | 75.42% | 68.68% | 98.41% | 97.96% | absolute + relative |
| L13 | 75.32% | 68.56% | 98.40% | 97.94% |  |
| L14 | 74.39% | 67.52% | 97.44% | 96.81% |  |
| L15 | 74.40% | 67.52% | 97.44% | 96.81% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.06% | 58.72% | 65.22% | 58.59% |  |
| L1 | 75.90% | 69.68% | 78.15% | 72.34% |  |
| L2 | 83.35% | 79.65% | 87.49% | 84.75% |  |
| L3 | 81.23% | 76.66% | 88.73% | 86.11% |  |
| L4 | 80.66% | 75.71% | 90.19% | 87.73% |  |
| L5 | 85.57% | 81.94% | 91.33% | 89.11% |  |
| L6 | 86.19% | 82.71% | 92.35% | 90.40% |  |
| L7 | 86.15% | 82.46% | 93.66% | 91.94% |  |
| L8 | 89.03% | 86.18% | 95.70% | 94.58% |  |
| L9 | 85.62% | 81.76% | 96.43% | 95.47% |  |
| L10 | 88.53% | 85.41% | 97.30% | 96.53% |  |
| L11 | 89.56% | 86.71% | 97.71% | 97.05% | absolute |
| L12 | 80.89% | 75.66% | 98.28% | 97.77% | relative |
| L13 | 79.32% | 73.67% | 98.16% | 97.62% |  |
| L14 | 74.49% | 67.64% | 96.84% | 96.05% |  |
| L15 | 74.25% | 67.33% | 96.85% | 96.06% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `2`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 90.40% | 90.43% | 2.290 |
| Random direction | 91.00% | 90.40% | 2.322 |
| Relative-board direction | 96.90% | 93.24% | 0.966 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b8_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b8_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 196.2 s |
| board_probe_linear | 100.0 s |
| board_probe_mlp | 119.1 s |
| causal_intervention | 109.1 s |
| native_ar | 1150.0 s |
