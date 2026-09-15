# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `37d4dd61621ee4b0c1e8eace9422e22c26776978fc8279e8872e2f06f1a71351`
- **Split manifest SHA-256:** `4fd0bfa141d51f8ff969526a9b5128a1afb4b9e0afc730f547102bb2ad8f59fd`
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
| LINEAR | 99.40% | 96.54% | 92.25% | 97.76% |
| MLP | 99.39% | 96.51% | 92.22% | 97.58% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 68.94% | 69.01% | 75.67% | 53.56% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.71% | 96.89% | 97.58% | 95.39% | 99.99% |
| MLP | absolute | L5 | 0.625 | 95.04% | 95.08% | 96.13% | 92.62% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.38% | 96.56% | 97.33% | 94.92% | 99.97% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.71% | 55.80% | 64.80% | 58.18% |  |
| L1 | 75.14% | 68.44% | 87.70% | 84.34% |  |
| L2 | 75.70% | 69.04% | 91.78% | 89.44% |  |
| L3 | 75.69% | 69.02% | 94.19% | 92.55% |  |
| L4 | 75.68% | 69.02% | 96.13% | 95.03% |  |
| L5 | 75.67% | 69.01% | 97.30% | 96.53% | absolute |
| L6 | 75.62% | 68.95% | 97.58% | 96.89% | relative |
| L7 | 75.25% | 68.55% | 97.12% | 96.34% |  |
| L8 | 74.99% | 68.28% | 96.68% | 95.84% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.11% | 58.78% | 65.24% | 58.63% |  |
| L1 | 84.78% | 80.83% | 87.02% | 83.51% |  |
| L2 | 88.84% | 85.79% | 91.53% | 89.10% |  |
| L3 | 91.79% | 89.54% | 93.94% | 92.24% |  |
| L4 | 94.33% | 92.79% | 95.96% | 94.82% |  |
| L5 | 96.13% | 95.08% | 97.13% | 96.31% | absolute |
| L6 | 95.24% | 93.95% | 97.33% | 96.56% | relative |
| L7 | 92.68% | 90.82% | 96.67% | 95.80% |  |
| L8 | 89.27% | 86.58% | 96.02% | 95.05% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 89.80% | 88.77% | 2.418 |
| Random direction | 90.00% | 88.44% | 2.404 |
| Relative-board direction | 95.90% | 93.31% | 0.572 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 161.4 s |
| board_probe_linear | 54.4 s |
| board_probe_mlp | 64.7 s |
| causal_intervention | 63.0 s |
| frozen_head_linear | 1583.4 s |
| frozen_head_mlp | 1449.3 s |
| head_tuning_linear | 97.0 s |
| head_tuning_mlp | 2.1 s |
