# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `c75a1fc74eb33d49b0593af85c6407343bf71de7c5183d9a613e6e07bcd69043`
- **Split manifest SHA-256:** `e000730655e82dc7c298768a1973687b8c96f332b1979b66bb03447b3925c8dc`
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
| LINEAR | 99.43% | 96.56% | 92.26% | 97.66% |
| MLP | 99.41% | 96.53% | 92.23% | 97.50% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L3 | 0.375 | 69.05% | 69.03% | 75.69% | 53.59% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.13% | 96.38% | 97.18% | 94.63% | 99.98% |
| MLP | absolute | L5 | 0.625 | 94.40% | 94.52% | 95.69% | 91.78% | 100.00% |
| MLP | relative | L6 | 0.750 | 95.81% | 96.07% | 96.94% | 94.21% | 99.95% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.68% | 55.77% | 64.77% | 58.14% |  |
| L1 | 75.18% | 68.47% | 87.58% | 84.15% |  |
| L2 | 75.71% | 69.06% | 92.08% | 89.83% |  |
| L3 | 75.69% | 69.03% | 94.26% | 92.63% | absolute |
| L4 | 75.67% | 69.00% | 96.03% | 94.91% |  |
| L5 | 75.68% | 69.01% | 96.98% | 96.12% |  |
| L6 | 75.54% | 68.85% | 97.18% | 96.38% | relative |
| L7 | 75.19% | 68.46% | 96.72% | 95.83% |  |
| L8 | 74.89% | 68.16% | 96.33% | 95.39% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.11% | 58.76% | 65.25% | 58.64% |  |
| L1 | 84.37% | 80.26% | 86.94% | 83.37% |  |
| L2 | 89.05% | 86.06% | 91.84% | 89.51% |  |
| L3 | 91.62% | 89.34% | 94.03% | 92.35% |  |
| L4 | 94.34% | 92.80% | 95.86% | 94.69% |  |
| L5 | 95.69% | 94.52% | 96.84% | 95.94% | absolute |
| L6 | 94.35% | 92.83% | 96.94% | 96.07% | relative |
| L7 | 92.39% | 90.46% | 96.28% | 95.29% |  |
| L8 | 89.25% | 86.56% | 95.72% | 94.65% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 90.20% | 88.63% | 2.420 |
| Random direction | 90.80% | 88.28% | 2.424 |
| Relative-board direction | 95.60% | 93.12% | 0.586 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b8_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed001/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed001/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b8_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 160.6 s |
| board_probe_linear | 55.1 s |
| board_probe_mlp | 65.2 s |
| causal_intervention | 63.2 s |
| frozen_head_linear | 1515.9 s |
| frozen_head_mlp | 1395.1 s |
| head_tuning_linear | 92.9 s |
| head_tuning_mlp | 2.1 s |
