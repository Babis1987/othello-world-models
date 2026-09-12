# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `1e350a96ded6586032173bb406c6d6d9eec316a7a0105bf16a643fc0f55ee2bd`
- **Split manifest SHA-256:** `207e81d6c5dd56917273b56f10bdfc1c9eb3b28951fd497de1e4f03e456003bf`
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
| Native AR | 99.10% | 98.31% | 97.27% | 91.31% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L10 | 0.667 | 66.79% | 66.71% | 74.72% | 50.09% | 99.96% |
| LINEAR | relative | L11 | 0.733 | 84.14% | 84.13% | 87.97% | 76.26% | 99.96% |
| MLP | absolute | L10 | 0.667 | 66.39% | 66.38% | 74.41% | 49.69% | 99.74% |
| MLP | relative | L11 | 0.733 | 82.10% | 82.06% | 86.32% | 73.40% | 99.56% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.75% | 56.61% | 65.66% | 57.81% |  |
| L1 | 69.96% | 61.85% | 71.03% | 63.21% |  |
| L2 | 69.40% | 61.35% | 72.63% | 65.64% |  |
| L3 | 68.46% | 60.42% | 78.49% | 73.72% |  |
| L4 | 69.54% | 61.49% | 79.25% | 74.41% |  |
| L5 | 70.15% | 62.10% | 79.80% | 74.94% |  |
| L6 | 70.64% | 62.59% | 80.36% | 75.52% |  |
| L7 | 71.77% | 63.72% | 81.58% | 76.74% |  |
| L8 | 72.55% | 64.50% | 82.48% | 77.66% |  |
| L9 | 73.66% | 65.63% | 83.83% | 79.06% |  |
| L10 | 74.72% | 66.71% | 85.23% | 80.53% | absolute |
| L11 | 74.70% | 66.69% | 87.97% | 84.13% | relative |
| L12 | 74.68% | 66.66% | 87.93% | 84.07% |  |
| L13 | 74.59% | 66.53% | 87.77% | 83.86% |  |
| L14 | 74.51% | 66.43% | 87.64% | 83.69% |  |
| L15 | 74.50% | 66.43% | 87.61% | 83.65% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.17% | 57.00% | 65.82% | 57.87% |  |
| L1 | 68.68% | 60.64% | 69.52% | 61.63% |  |
| L2 | 68.35% | 60.42% | 70.71% | 63.49% |  |
| L3 | 68.64% | 61.02% | 76.04% | 70.98% |  |
| L4 | 68.67% | 60.72% | 76.70% | 71.60% |  |
| L5 | 69.04% | 61.03% | 77.13% | 72.03% |  |
| L6 | 69.47% | 61.45% | 77.52% | 72.36% |  |
| L7 | 70.62% | 62.61% | 78.80% | 73.61% |  |
| L8 | 71.44% | 63.44% | 79.67% | 74.51% |  |
| L9 | 72.73% | 64.69% | 81.27% | 76.10% |  |
| L10 | 74.41% | 66.38% | 83.47% | 78.31% | absolute |
| L11 | 74.44% | 66.40% | 86.32% | 82.06% | relative |
| L12 | 74.37% | 66.27% | 86.30% | 81.99% |  |
| L13 | 74.27% | 66.13% | 86.00% | 81.58% |  |
| L14 | 74.16% | 65.99% | 85.78% | 81.30% |  |
| L15 | 74.12% | 65.95% | 85.77% | 81.29% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b16_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b16_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1694.3 s |
| board_probe_linear | 178.7 s |
| board_probe_mlp | 199.1 s |
| native_ar | 17202.4 s |
