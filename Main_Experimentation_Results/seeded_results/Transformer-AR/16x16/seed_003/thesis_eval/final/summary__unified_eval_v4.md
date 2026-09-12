# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `8d632f09d64539ce21780222f8d84a7bb3dab0d94d406c31baae492765766d99`
- **Split manifest SHA-256:** `51b69c8386f8b4d48aa75c80165c51da2c5069ec66d19587a9f094bd83137b93`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 98.96% | 98.10% | 97.00% | 88.74% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 66.76% | 66.76% | 74.75% | 50.18% | 99.93% |
| LINEAR | relative | L8 | 1.000 | 82.87% | 82.88% | 87.02% | 74.39% | 99.97% |
| MLP | absolute | L8 | 1.000 | 66.86% | 66.86% | 74.81% | 50.33% | 99.91% |
| MLP | relative | L7 | 0.875 | 80.50% | 80.47% | 85.14% | 70.97% | 99.67% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.62% | 56.61% | 65.53% | 57.81% |  |
| L1 | 66.67% | 58.69% | 67.91% | 60.30% |  |
| L2 | 72.69% | 64.63% | 74.15% | 66.51% |  |
| L3 | 74.46% | 66.45% | 76.60% | 69.22% |  |
| L4 | 74.65% | 66.65% | 80.46% | 74.25% |  |
| L5 | 74.75% | 66.76% | 85.50% | 80.88% | absolute |
| L6 | 74.72% | 66.71% | 86.73% | 82.48% |  |
| L7 | 74.62% | 66.58% | 87.01% | 82.85% |  |
| L8 | 74.60% | 66.55% | 87.02% | 82.88% | relative |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.52% | 57.50% | 65.82% | 57.87% |  |
| L1 | 66.69% | 58.69% | 67.19% | 59.30% |  |
| L2 | 71.29% | 63.25% | 72.32% | 64.50% |  |
| L3 | 73.48% | 65.43% | 74.91% | 67.20% |  |
| L4 | 74.09% | 66.04% | 77.94% | 70.98% |  |
| L5 | 74.51% | 66.53% | 83.33% | 78.08% |  |
| L6 | 74.71% | 66.74% | 84.93% | 80.19% |  |
| L7 | 74.57% | 66.55% | 85.14% | 80.47% | relative |
| L8 | 74.81% | 66.86% | 85.12% | 80.42% | absolute |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b16_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b16_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1646.1 s |
| board_probe_linear | 99.0 s |
| board_probe_mlp | 109.8 s |
| native_ar | 16708.2 s |
