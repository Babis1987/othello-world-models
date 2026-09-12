# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `318065f6633a198860466a57e4fec40959b7e6ece35bd595a7a1bb66bb67e6c8`
- **Split manifest SHA-256:** `11b74050df1e48660ede1eed3b14588ded25d266fc65b49c823cd68912dcd42e`
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
| LINEAR | 97.44% | 96.38% | 95.18% | 87.76% |
| MLP | 97.07% | 96.02% | 94.83% | 87.47% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 66.32% | 66.28% | 74.36% | 49.50% | 99.85% |
| LINEAR | relative | L6 | 0.750 | 78.59% | 78.67% | 83.79% | 68.17% | 99.79% |
| MLP | absolute | L6 | 0.750 | 65.93% | 65.90% | 74.00% | 49.06% | 99.56% |
| MLP | relative | L6 | 0.750 | 76.32% | 76.37% | 81.88% | 65.07% | 99.12% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.70% | 56.62% | 65.54% | 57.74% |  |
| L1 | 65.62% | 57.60% | 67.01% | 59.43% |  |
| L2 | 68.36% | 60.27% | 69.74% | 62.07% |  |
| L3 | 71.28% | 63.15% | 72.80% | 65.14% |  |
| L4 | 72.71% | 64.58% | 75.62% | 68.40% |  |
| L5 | 74.36% | 66.28% | 80.50% | 74.34% | absolute |
| L6 | 74.40% | 66.31% | 83.79% | 78.67% | relative |
| L7 | 74.11% | 66.00% | 83.46% | 78.33% |  |
| L8 | 73.69% | 65.58% | 82.52% | 77.26% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.53% | 57.51% | 65.79% | 57.83% |  |
| L1 | 66.08% | 57.98% | 66.65% | 58.72% |  |
| L2 | 67.70% | 59.65% | 68.45% | 60.55% |  |
| L3 | 69.80% | 61.74% | 70.77% | 62.98% |  |
| L4 | 71.24% | 63.18% | 73.21% | 65.75% |  |
| L5 | 73.43% | 65.29% | 78.17% | 71.58% |  |
| L6 | 74.00% | 65.90% | 81.88% | 76.37% | absolute + relative |
| L7 | 73.61% | 65.41% | 81.57% | 75.97% |  |
| L8 | 73.12% | 64.92% | 80.32% | 74.51% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b16_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed000/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b16_seed000/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b16_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b16_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1646.1 s |
| board_probe_linear | 99.4 s |
| board_probe_mlp | 110.2 s |
| frozen_head_linear | 20933.4 s |
| frozen_head_mlp | 19663.8 s |
| head_tuning_linear | 1591.6 s |
| head_tuning_mlp | 10.2 s |
