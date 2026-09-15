# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `01c0e04cee8c0175e13d8bad97ae414b05eab5b6cdb642db10921a0f627c57ea`
- **Split manifest SHA-256:** `86fdcb66e63d50c6754ec12c7e31eb57749ac6d6e2768150e152fb906b48f5f7`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.83% | 96.90% | 92.57% | 99.42% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L10 | 0.667 | 68.86% | 68.71% | 75.44% | 53.11% | 99.99% |
| LINEAR | relative | L13 | 0.867 | 97.82% | 98.05% | 98.48% | 97.11% | 99.99% |
| MLP | absolute | L11 | 0.733 | 91.31% | 91.54% | 93.36% | 87.33% | 99.99% |
| MLP | relative | L12 | 0.800 | 97.60% | 97.88% | 98.36% | 96.89% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.61% | 55.56% | 64.81% | 58.17% |  |
| L1 | 72.60% | 65.78% | 78.24% | 72.74% |  |
| L2 | 74.21% | 67.39% | 83.34% | 78.88% |  |
| L3 | 73.46% | 66.61% | 87.11% | 83.89% |  |
| L4 | 73.06% | 66.17% | 88.82% | 86.16% |  |
| L5 | 73.60% | 66.67% | 89.87% | 87.32% |  |
| L6 | 73.49% | 66.56% | 91.61% | 89.55% |  |
| L7 | 74.33% | 67.46% | 92.58% | 90.63% |  |
| L8 | 74.12% | 67.23% | 93.36% | 91.69% |  |
| L9 | 73.59% | 66.65% | 95.88% | 94.88% |  |
| L10 | 75.44% | 68.71% | 97.33% | 96.58% | absolute |
| L11 | 75.44% | 68.71% | 98.01% | 97.45% |  |
| L12 | 75.47% | 68.74% | 98.49% | 98.07% |  |
| L13 | 75.38% | 68.63% | 98.48% | 98.05% | relative |
| L14 | 74.32% | 67.43% | 97.51% | 96.89% |  |
| L15 | 74.30% | 67.41% | 97.51% | 96.90% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.08% | 58.70% | 65.24% | 58.62% |  |
| L1 | 75.23% | 69.22% | 77.60% | 71.93% |  |
| L2 | 78.31% | 72.75% | 82.43% | 77.80% |  |
| L3 | 82.80% | 78.73% | 86.03% | 82.67% |  |
| L4 | 83.99% | 80.36% | 87.60% | 84.84% |  |
| L5 | 84.54% | 80.81% | 88.75% | 86.05% |  |
| L6 | 86.23% | 83.06% | 90.42% | 88.21% |  |
| L7 | 85.98% | 82.41% | 91.62% | 89.49% |  |
| L8 | 87.27% | 84.11% | 92.30% | 90.43% |  |
| L9 | 91.43% | 89.56% | 94.91% | 93.76% |  |
| L10 | 91.41% | 89.07% | 97.08% | 96.25% |  |
| L11 | 93.36% | 91.54% | 97.85% | 97.24% | absolute |
| L12 | 84.49% | 80.26% | 98.36% | 97.88% | relative |
| L13 | 79.27% | 73.59% | 98.28% | 97.78% |  |
| L14 | 74.61% | 67.79% | 96.95% | 96.18% |  |
| L15 | 74.67% | 67.88% | 96.92% | 96.15% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `4`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.40% | 90.46% | 2.318 |
| Random direction | 89.60% | 90.38% | 2.298 |
| Relative-board direction | 95.50% | 94.45% | 0.680 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b8_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b8_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 193.6 s |
| board_probe_linear | 100.1 s |
| board_probe_mlp | 119.6 s |
| causal_intervention | 109.3 s |
| native_ar | 1114.6 s |
