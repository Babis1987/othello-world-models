# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `7046e3d2667316b7d5cdbadf2c1e586ed5a2242d528dbec09cf8802ad1010cb4`
- **Split manifest SHA-256:** `7838ea6971305536361fec6dc20bb8966b2fef1db83e3638e998ae9d857da81d`
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
| LINEAR | 97.59% | 96.54% | 95.33% | 87.77% |
| MLP | 97.17% | 96.10% | 94.89% | 87.50% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L6 | 0.750 | 66.20% | 66.19% | 74.29% | 49.36% | 99.84% |
| LINEAR | relative | L7 | 0.875 | 77.74% | 77.80% | 83.05% | 67.06% | 99.44% |
| MLP | absolute | L6 | 0.750 | 66.36% | 66.43% | 74.37% | 49.93% | 99.42% |
| MLP | relative | L7 | 0.875 | 75.39% | 75.44% | 81.14% | 63.77% | 98.95% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.59% | 56.48% | 65.49% | 57.67% |  |
| L1 | 65.74% | 57.70% | 67.06% | 59.45% |  |
| L2 | 68.25% | 60.15% | 69.64% | 61.98% |  |
| L3 | 70.59% | 62.52% | 72.24% | 64.65% |  |
| L4 | 72.45% | 64.34% | 74.25% | 66.67% |  |
| L5 | 73.94% | 65.81% | 80.58% | 74.58% |  |
| L6 | 74.29% | 66.19% | 83.06% | 77.73% | absolute |
| L7 | 74.07% | 65.96% | 83.05% | 77.80% | relative |
| L8 | 73.67% | 65.58% | 82.06% | 76.67% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.50% | 57.48% | 65.77% | 57.81% |  |
| L1 | 66.15% | 58.07% | 66.69% | 58.76% |  |
| L2 | 67.65% | 59.64% | 68.33% | 60.45% |  |
| L3 | 69.26% | 61.19% | 70.32% | 62.53% |  |
| L4 | 70.89% | 62.79% | 72.08% | 64.34% |  |
| L5 | 72.88% | 64.74% | 78.10% | 71.75% |  |
| L6 | 74.37% | 66.43% | 81.10% | 75.39% | absolute |
| L7 | 73.79% | 65.66% | 81.14% | 75.44% | relative |
| L8 | 73.08% | 64.89% | 79.86% | 73.93% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b16_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed001/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed001/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b16_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1646.2 s |
| board_probe_linear | 104.6 s |
| board_probe_mlp | 115.6 s |
| frozen_head_linear | 20535.2 s |
| frozen_head_mlp | 19558.3 s |
| head_tuning_linear | 1577.4 s |
| head_tuning_mlp | 10.3 s |
