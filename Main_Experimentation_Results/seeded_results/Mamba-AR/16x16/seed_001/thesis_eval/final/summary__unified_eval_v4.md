# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `3af7b6265cc72414d0691fb732b6b487e0d5051d71c12bec7eb2e25242b209b2`
- **Split manifest SHA-256:** `23f3e62983aea3630779fad6ec3580ae4b9eb30d32e82876c076674487fd12d2`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.02% | 98.19% | 97.12% | 90.35% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L11 | 0.733 | 66.67% | 66.61% | 74.64% | 49.93% | 99.97% |
| LINEAR | relative | L12 | 0.800 | 83.04% | 82.98% | 87.11% | 74.55% | 99.98% |
| MLP | absolute | L12 | 0.800 | 66.30% | 66.25% | 74.34% | 49.42% | 99.89% |
| MLP | relative | L11 | 0.733 | 81.15% | 81.10% | 85.57% | 72.02% | 99.46% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.75% | 56.59% | 65.65% | 57.77% |  |
| L1 | 70.86% | 62.80% | 72.17% | 64.47% |  |
| L2 | 69.05% | 61.00% | 76.33% | 70.71% |  |
| L3 | 68.43% | 60.37% | 79.63% | 75.20% |  |
| L4 | 70.22% | 62.15% | 81.31% | 76.85% |  |
| L5 | 70.66% | 62.59% | 81.66% | 77.17% |  |
| L6 | 71.04% | 62.99% | 82.21% | 77.78% |  |
| L7 | 71.43% | 63.38% | 82.87% | 78.52% |  |
| L8 | 71.98% | 63.92% | 83.82% | 79.57% |  |
| L9 | 73.39% | 65.36% | 85.47% | 81.30% |  |
| L10 | 74.04% | 66.02% | 86.32% | 82.19% |  |
| L11 | 74.64% | 66.61% | 87.10% | 82.99% | absolute |
| L12 | 74.64% | 66.60% | 87.11% | 82.98% | relative |
| L13 | 74.54% | 66.47% | 86.95% | 82.78% |  |
| L14 | 74.45% | 66.36% | 86.83% | 82.62% |  |
| L15 | 74.46% | 66.37% | 86.78% | 82.57% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.06% | 56.85% | 65.83% | 57.88% |  |
| L1 | 69.31% | 61.26% | 70.36% | 62.53% |  |
| L2 | 68.35% | 60.49% | 73.77% | 67.85% |  |
| L3 | 67.41% | 59.38% | 77.19% | 72.50% |  |
| L4 | 69.02% | 60.95% | 78.69% | 74.00% |  |
| L5 | 69.49% | 61.40% | 79.05% | 74.30% |  |
| L6 | 69.90% | 61.86% | 79.61% | 74.91% |  |
| L7 | 70.27% | 62.26% | 80.25% | 75.67% |  |
| L8 | 70.93% | 62.91% | 81.29% | 76.76% |  |
| L9 | 72.62% | 64.59% | 83.31% | 78.82% |  |
| L10 | 73.47% | 65.43% | 84.41% | 79.94% |  |
| L11 | 74.32% | 66.25% | 85.57% | 81.10% | relative |
| L12 | 74.34% | 66.25% | 85.58% | 81.04% | absolute |
| L13 | 74.24% | 66.10% | 85.30% | 80.67% |  |
| L14 | 74.11% | 65.93% | 85.07% | 80.36% |  |
| L15 | 74.09% | 65.91% | 85.00% | 80.26% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b16_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b16_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1651.4 s |
| board_probe_linear | 178.2 s |
| board_probe_mlp | 198.9 s |
| native_ar | 16874.5 s |
