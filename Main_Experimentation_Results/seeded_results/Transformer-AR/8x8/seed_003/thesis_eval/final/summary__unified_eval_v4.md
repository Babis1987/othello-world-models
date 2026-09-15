# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `8945cc94c8face57f7b2fe99ba886d0f5cfe3492db6dbddb4a01c7450c853c2a`
- **Split manifest SHA-256:** `f97acacc82699366dca6f035884fadd336a5a8df30a523c416b203681b685277`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.71% | 96.78% | 92.45% | 98.50% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L4 | 0.500 | 68.96% | 68.89% | 75.59% | 53.39% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.72% | 96.98% | 97.65% | 95.51% | 99.99% |
| MLP | absolute | L5 | 0.625 | 93.67% | 93.85% | 95.18% | 90.79% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.43% | 96.70% | 97.45% | 95.14% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.69% | 55.78% | 64.79% | 58.19% |  |
| L1 | 75.17% | 68.44% | 87.50% | 84.04% |  |
| L2 | 75.58% | 68.89% | 92.46% | 90.31% |  |
| L3 | 75.61% | 68.92% | 94.58% | 93.04% |  |
| L4 | 75.59% | 68.89% | 96.04% | 94.92% | absolute |
| L5 | 75.53% | 68.82% | 97.22% | 96.44% |  |
| L6 | 75.40% | 68.67% | 97.65% | 96.98% | relative |
| L7 | 75.18% | 68.41% | 97.61% | 96.95% |  |
| L8 | 75.11% | 68.32% | 97.54% | 96.86% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.12% | 58.78% | 65.23% | 58.57% |  |
| L1 | 84.48% | 80.39% | 86.85% | 83.25% |  |
| L2 | 89.87% | 87.11% | 92.18% | 89.94% |  |
| L3 | 92.11% | 89.95% | 94.36% | 92.77% |  |
| L4 | 93.79% | 92.10% | 95.93% | 94.76% |  |
| L5 | 95.18% | 93.85% | 97.11% | 96.27% | absolute |
| L6 | 93.90% | 92.24% | 97.45% | 96.70% | relative |
| L7 | 89.15% | 86.20% | 97.33% | 96.57% |  |
| L8 | 86.03% | 82.23% | 97.19% | 96.41% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `2`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.20% | 89.67% | 2.302 |
| Random direction | 91.10% | 89.66% | 2.276 |
| Relative-board direction | 97.60% | 95.69% | 0.474 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b8_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b8_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 159.0 s |
| board_probe_linear | 56.4 s |
| board_probe_mlp | 65.3 s |
| causal_intervention | 64.3 s |
| native_ar | 1106.7 s |
