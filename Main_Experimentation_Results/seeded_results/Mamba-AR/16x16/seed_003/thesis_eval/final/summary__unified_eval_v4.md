# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `ce2a4acc067f71be608b3a6c4920c221adbaf94c98bf0c31dd93d36becfbf1ee`
- **Split manifest SHA-256:** `6e4bf67f59bb39cb18ad352c74782553479e50aa24fec0e4c579432e48e01f6a`
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
| Native AR | 99.00% | 98.19% | 97.13% | 90.68% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 66.65% | 66.60% | 74.64% | 49.90% | 99.99% |
| LINEAR | relative | L12 | 0.800 | 83.46% | 83.44% | 87.45% | 75.24% | 99.97% |
| MLP | absolute | L12 | 0.800 | 66.30% | 66.26% | 74.35% | 49.45% | 99.87% |
| MLP | relative | L12 | 0.800 | 81.46% | 81.48% | 85.89% | 72.53% | 99.58% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.75% | 56.60% | 65.66% | 57.79% |  |
| L1 | 69.99% | 61.87% | 71.18% | 63.39% |  |
| L2 | 69.11% | 61.06% | 73.22% | 66.56% |  |
| L3 | 67.84% | 59.76% | 77.57% | 72.66% |  |
| L4 | 68.70% | 60.62% | 80.39% | 76.07% |  |
| L5 | 69.34% | 61.28% | 80.96% | 76.65% |  |
| L6 | 69.76% | 61.71% | 81.50% | 77.24% |  |
| L7 | 70.50% | 62.45% | 82.49% | 78.32% |  |
| L8 | 71.39% | 63.35% | 83.67% | 79.59% |  |
| L9 | 72.75% | 64.74% | 85.17% | 81.14% |  |
| L10 | 73.52% | 65.50% | 86.13% | 82.13% |  |
| L11 | 73.93% | 65.92% | 86.61% | 82.61% |  |
| L12 | 74.64% | 66.60% | 87.45% | 83.44% | absolute + relative |
| L13 | 74.56% | 66.50% | 87.26% | 83.19% |  |
| L14 | 74.45% | 66.36% | 87.09% | 82.98% |  |
| L15 | 74.43% | 66.33% | 87.05% | 82.92% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.17% | 57.00% | 65.83% | 57.89% |  |
| L1 | 68.59% | 60.53% | 69.52% | 61.64% |  |
| L2 | 68.28% | 60.30% | 71.44% | 64.55% |  |
| L3 | 68.14% | 60.44% | 75.36% | 70.25% |  |
| L4 | 67.87% | 59.85% | 78.17% | 73.72% |  |
| L5 | 68.39% | 60.37% | 78.59% | 74.12% |  |
| L6 | 68.72% | 60.67% | 79.07% | 74.62% |  |
| L7 | 69.40% | 61.36% | 79.95% | 75.55% |  |
| L8 | 70.25% | 62.21% | 81.04% | 76.67% |  |
| L9 | 71.80% | 63.77% | 82.83% | 78.51% |  |
| L10 | 72.82% | 64.79% | 84.08% | 79.76% |  |
| L11 | 73.35% | 65.29% | 84.75% | 80.40% |  |
| L12 | 74.35% | 66.26% | 85.89% | 81.48% | absolute + relative |
| L13 | 74.22% | 66.07% | 85.57% | 81.02% |  |
| L14 | 74.07% | 65.88% | 85.27% | 80.64% |  |
| L15 | 74.09% | 65.90% | 85.25% | 80.60% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b16_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b16_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1658.1 s |
| board_probe_linear | 176.3 s |
| board_probe_mlp | 196.4 s |
| native_ar | 16782.6 s |
