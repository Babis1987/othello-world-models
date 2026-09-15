# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `4a3d91e22f2ab45bbc85b58d27a15a36eaffefbeb858b43adb7ee113e73ecf8e`
- **Split manifest SHA-256:** `d91e118d3ff06e35cfd95d14d8b453bc4f62a47281d19588fcf6c0db56982ef2`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.42% | 96.56% | 92.26% | 97.69% |
| MLP | 99.43% | 96.53% | 92.23% | 97.87% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L4 | 0.500 | 69.07% | 69.08% | 75.73% | 53.67% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 95.94% | 96.34% | 97.16% | 94.58% | 99.99% |
| MLP | absolute | L5 | 0.625 | 94.33% | 94.40% | 95.60% | 91.59% | 100.00% |
| MLP | relative | L6 | 0.750 | 95.66% | 96.06% | 96.94% | 94.19% | 99.97% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.69% | 55.79% | 64.80% | 58.17% |  |
| L1 | 75.29% | 68.61% | 87.78% | 84.39% |  |
| L2 | 75.63% | 68.95% | 92.43% | 90.28% |  |
| L3 | 75.72% | 69.06% | 94.14% | 92.48% |  |
| L4 | 75.73% | 69.08% | 95.77% | 94.56% | absolute |
| L5 | 75.81% | 69.18% | 96.80% | 95.89% |  |
| L6 | 75.64% | 68.97% | 97.16% | 96.34% | relative |
| L7 | 75.12% | 68.37% | 96.66% | 95.75% |  |
| L8 | 74.93% | 68.20% | 96.21% | 95.22% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.08% | 58.76% | 65.23% | 58.61% |  |
| L1 | 84.35% | 80.22% | 87.19% | 83.70% |  |
| L2 | 89.70% | 86.90% | 92.14% | 89.88% |  |
| L3 | 91.19% | 88.79% | 93.92% | 92.20% |  |
| L4 | 93.95% | 92.30% | 95.59% | 94.32% |  |
| L5 | 95.60% | 94.40% | 96.66% | 95.70% | absolute |
| L6 | 94.83% | 93.43% | 96.94% | 96.06% | relative |
| L7 | 92.51% | 90.59% | 96.27% | 95.28% |  |
| L8 | 88.28% | 85.32% | 95.64% | 94.55% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.20% | 89.09% | 2.394 |
| Random direction | 91.30% | 88.68% | 2.414 |
| Relative-board direction | 95.30% | 94.00% | 0.560 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b8_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed003/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed003/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b8_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 159.8 s |
| board_probe_linear | 57.8 s |
| board_probe_mlp | 67.7 s |
| causal_intervention | 66.0 s |
| frozen_head_linear | 1564.7 s |
| frozen_head_mlp | 1694.4 s |
| head_tuning_linear | 92.4 s |
| head_tuning_mlp | 2.1 s |
