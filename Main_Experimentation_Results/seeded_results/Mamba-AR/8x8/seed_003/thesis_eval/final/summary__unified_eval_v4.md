# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `fefed6e1bbe8a99dc13bcbde77acb4bb4cc631d4359f08dda325fbd25db3dc2c`
- **Split manifest SHA-256:** `97c9660b64fbe8f445a02068c91d5b9578b2a8604a7dffad2de0d562b542ecae`
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
| Native AR | 99.84% | 96.90% | 92.57% | 99.37% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 68.81% | 68.81% | 75.51% | 53.28% | 99.96% |
| LINEAR | relative | L12 | 0.800 | 97.43% | 97.71% | 98.20% | 96.61% | 99.95% |
| MLP | absolute | L11 | 0.733 | 90.64% | 90.79% | 92.75% | 86.21% | 99.94% |
| MLP | relative | L12 | 0.800 | 97.23% | 97.56% | 98.11% | 96.45% | 99.93% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.63% | 55.58% | 64.76% | 58.11% |  |
| L1 | 74.02% | 67.14% | 80.36% | 74.99% |  |
| L2 | 72.98% | 66.07% | 86.13% | 82.71% |  |
| L3 | 71.66% | 64.73% | 90.46% | 88.42% |  |
| L4 | 72.64% | 65.68% | 91.83% | 89.93% |  |
| L5 | 73.32% | 66.34% | 93.11% | 91.38% |  |
| L6 | 73.61% | 66.65% | 93.96% | 92.43% |  |
| L7 | 74.30% | 67.42% | 95.12% | 93.83% |  |
| L8 | 74.71% | 67.88% | 95.94% | 94.86% |  |
| L9 | 75.24% | 68.48% | 96.74% | 95.84% |  |
| L10 | 75.45% | 68.73% | 97.10% | 96.31% |  |
| L11 | 75.47% | 68.76% | 97.87% | 97.29% |  |
| L12 | 75.51% | 68.81% | 98.20% | 97.71% | absolute + relative |
| L13 | 75.35% | 68.60% | 98.11% | 97.58% |  |
| L14 | 74.52% | 67.69% | 97.22% | 96.53% |  |
| L15 | 74.26% | 67.35% | 97.12% | 96.40% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.09% | 58.74% | 65.28% | 58.67% |  |
| L1 | 75.99% | 69.73% | 79.47% | 74.01% |  |
| L2 | 81.02% | 76.52% | 85.05% | 81.51% |  |
| L3 | 84.65% | 81.62% | 89.29% | 87.23% |  |
| L4 | 82.85% | 78.91% | 90.75% | 88.73% |  |
| L5 | 81.31% | 76.63% | 92.18% | 90.28% |  |
| L6 | 79.81% | 74.64% | 93.04% | 91.33% |  |
| L7 | 84.16% | 80.05% | 94.42% | 93.00% |  |
| L8 | 85.17% | 81.26% | 95.39% | 94.18% |  |
| L9 | 88.83% | 85.80% | 96.54% | 95.55% |  |
| L10 | 89.88% | 87.13% | 96.96% | 96.09% |  |
| L11 | 92.75% | 90.79% | 97.82% | 97.19% | absolute |
| L12 | 86.86% | 83.29% | 98.11% | 97.56% | relative |
| L13 | 78.49% | 72.62% | 97.87% | 97.26% |  |
| L14 | 74.70% | 67.92% | 96.70% | 95.88% |  |
| L15 | 74.58% | 67.75% | 96.66% | 95.83% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `4`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.50% | 90.40% | 2.304 |
| Random direction | 91.00% | 90.36% | 2.318 |
| Relative-board direction | 96.20% | 94.94% | 0.616 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b8_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b8_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b8_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 193.6 s |
| board_probe_linear | 94.9 s |
| board_probe_mlp | 114.0 s |
| causal_intervention | 103.6 s |
| native_ar | 1107.2 s |
