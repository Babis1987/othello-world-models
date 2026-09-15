# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `5992d241b42782f4af086ed71dce537ba8ebc362bf31aecadfb04183e7b031b3`
- **Split manifest SHA-256:** `87a69e28ed5d98e370ecb47d758f65abb4c784a7b784c116f49613d49232def2`
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
| Native AR | 99.85% | 96.91% | 92.58% | 99.43% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L9 | 0.600 | 68.78% | 68.74% | 75.47% | 53.17% | 99.99% |
| LINEAR | relative | L12 | 0.800 | 97.79% | 98.02% | 98.45% | 97.05% | 100.00% |
| MLP | absolute | L10 | 0.667 | 89.45% | 89.64% | 91.86% | 84.49% | 99.97% |
| MLP | relative | L12 | 0.800 | 97.56% | 97.81% | 98.31% | 96.79% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.57% | 55.52% | 64.79% | 58.16% |  |
| L1 | 73.48% | 66.63% | 78.08% | 72.26% |  |
| L2 | 73.84% | 67.02% | 79.75% | 74.38% |  |
| L3 | 72.44% | 65.60% | 87.26% | 84.36% |  |
| L4 | 73.86% | 66.98% | 88.84% | 85.95% |  |
| L5 | 74.02% | 67.14% | 89.00% | 86.14% |  |
| L6 | 73.20% | 66.23% | 92.14% | 90.25% |  |
| L7 | 72.79% | 65.76% | 93.20% | 91.65% |  |
| L8 | 74.72% | 67.88% | 95.28% | 94.04% |  |
| L9 | 75.47% | 68.74% | 96.11% | 95.02% | absolute |
| L10 | 75.42% | 68.68% | 97.73% | 97.09% |  |
| L11 | 75.40% | 68.66% | 98.25% | 97.76% |  |
| L12 | 75.36% | 68.61% | 98.45% | 98.02% | relative |
| L13 | 75.22% | 68.44% | 98.40% | 97.94% |  |
| L14 | 74.10% | 67.17% | 97.36% | 96.73% |  |
| L15 | 74.13% | 67.20% | 97.37% | 96.74% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.05% | 58.69% | 65.24% | 58.61% |  |
| L1 | 75.03% | 68.69% | 77.32% | 71.38% |  |
| L2 | 75.73% | 69.55% | 78.96% | 73.46% |  |
| L3 | 82.00% | 78.09% | 85.92% | 82.89% |  |
| L4 | 82.63% | 78.29% | 87.80% | 84.69% |  |
| L5 | 82.86% | 78.53% | 87.94% | 84.89% |  |
| L6 | 86.79% | 83.78% | 90.78% | 88.69% |  |
| L7 | 86.65% | 83.73% | 91.76% | 90.06% |  |
| L8 | 88.16% | 85.09% | 94.42% | 93.00% |  |
| L9 | 86.69% | 83.06% | 95.56% | 94.32% |  |
| L10 | 91.86% | 89.64% | 97.51% | 96.79% | absolute |
| L11 | 82.76% | 78.05% | 98.13% | 97.58% |  |
| L12 | 77.06% | 70.79% | 98.31% | 97.81% | relative |
| L13 | 76.31% | 69.82% | 98.16% | 97.62% |  |
| L14 | 73.88% | 66.87% | 96.71% | 95.91% |  |
| L15 | 74.02% | 67.05% | 96.72% | 95.92% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `2`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.90% | 90.50% | 2.302 |
| Random direction | 89.70% | 90.41% | 2.306 |
| Relative-board direction | 95.10% | 92.20% | 1.060 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b8_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b8_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 192.9 s |
| board_probe_linear | 95.1 s |
| board_probe_mlp | 113.8 s |
| causal_intervention | 103.6 s |
| native_ar | 1104.5 s |
