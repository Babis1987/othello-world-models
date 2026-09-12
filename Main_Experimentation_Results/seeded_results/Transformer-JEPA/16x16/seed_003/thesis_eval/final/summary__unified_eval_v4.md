# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `d27263bbcc9659b83bf42c99d630f5c49f6de85ccb5d58be10381843b0d8a39c`
- **Split manifest SHA-256:** `f50704af953ce1ba2c1caa82a0d735be45e3248b383f789f027d99a3cfe8a4fb`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 97.57% | 96.52% | 95.32% | 87.79% |
| MLP | 97.21% | 96.15% | 94.94% | 87.49% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L6 | 0.750 | 66.33% | 66.32% | 74.40% | 49.52% | 99.90% |
| LINEAR | relative | L7 | 0.875 | 78.21% | 78.30% | 83.42% | 67.83% | 99.39% |
| MLP | absolute | L6 | 0.750 | 66.29% | 66.34% | 74.34% | 49.72% | 99.58% |
| MLP | relative | L7 | 0.875 | 75.88% | 75.91% | 81.48% | 64.54% | 98.83% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.64% | 56.57% | 65.51% | 57.71% |  |
| L1 | 65.64% | 57.59% | 67.05% | 59.45% |  |
| L2 | 68.14% | 60.01% | 69.61% | 61.95% |  |
| L3 | 70.47% | 62.39% | 72.21% | 64.66% |  |
| L4 | 72.58% | 64.48% | 74.86% | 67.44% |  |
| L5 | 74.28% | 66.19% | 79.22% | 72.67% |  |
| L6 | 74.40% | 66.32% | 83.43% | 78.20% | absolute |
| L7 | 74.02% | 65.90% | 83.42% | 78.30% | relative |
| L8 | 73.60% | 65.48% | 82.30% | 76.97% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.50% | 57.49% | 65.78% | 57.83% |  |
| L1 | 66.09% | 58.04% | 66.70% | 58.83% |  |
| L2 | 67.49% | 59.46% | 68.34% | 60.49% |  |
| L3 | 69.24% | 61.23% | 70.34% | 62.61% |  |
| L4 | 71.01% | 62.92% | 72.61% | 64.99% |  |
| L5 | 73.34% | 65.19% | 76.96% | 69.93% |  |
| L6 | 74.34% | 66.34% | 81.51% | 75.86% | absolute |
| L7 | 73.63% | 65.47% | 81.48% | 75.91% | relative |
| L8 | 73.06% | 64.87% | 80.01% | 74.11% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b16_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed003/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed003/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b16_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1713.4 s |
| board_probe_linear | 104.3 s |
| board_probe_mlp | 116.1 s |
| frozen_head_linear | 20764.6 s |
| frozen_head_mlp | 19534.3 s |
| head_tuning_linear | 1601.9 s |
| head_tuning_mlp | 10.6 s |
