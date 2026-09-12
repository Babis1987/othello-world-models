# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `c466a775c883a691b9d5f487cbe2a05013514d1631d9fb0bcaee895c28859729`
- **Split manifest SHA-256:** `fd4428a7030e5222cb9f11ad03f8ebec48d89803125937845e8601501923dadb`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `3de85ba0c54c02ccde18de35609415525e0553c52a3a009de764d5034018cf1d`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.75% | 96.82% | 92.50% | 98.56% |
| MLP | 99.72% | 96.80% | 92.49% | 99.01% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 68.96% | 68.93% | 75.61% | 53.43% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 97.98% | 98.21% | 98.61% | 97.35% | 100.00% |
| MLP | absolute | L12 | 0.800 | 94.08% | 94.12% | 95.38% | 91.19% | 99.99% |
| MLP | relative | L12 | 0.800 | 97.83% | 98.05% | 98.48% | 97.12% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.60% | 55.57% | 64.79% | 58.18% |  |
| L1 | 74.04% | 67.21% | 78.32% | 72.41% |  |
| L2 | 73.12% | 66.34% | 87.46% | 84.45% |  |
| L3 | 74.01% | 67.18% | 88.70% | 85.76% |  |
| L4 | 73.80% | 66.91% | 88.82% | 85.95% |  |
| L5 | 73.97% | 67.07% | 89.11% | 86.29% |  |
| L6 | 72.72% | 65.72% | 92.40% | 90.61% |  |
| L7 | 74.26% | 67.40% | 93.73% | 92.10% |  |
| L8 | 73.73% | 66.82% | 96.55% | 95.67% |  |
| L9 | 75.05% | 68.28% | 97.33% | 96.58% |  |
| L10 | 75.33% | 68.60% | 97.79% | 97.17% |  |
| L11 | 75.61% | 68.93% | 98.14% | 97.61% |  |
| L12 | 75.61% | 68.93% | 98.61% | 98.21% | absolute + relative |
| L13 | 75.54% | 68.84% | 98.47% | 98.03% |  |
| L14 | 74.99% | 68.23% | 97.24% | 96.54% |  |
| L15 | 75.29% | 68.71% | 96.49% | 95.66% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.08% | 58.75% | 65.27% | 58.64% |  |
| L1 | 75.79% | 69.51% | 77.70% | 71.73% |  |
| L2 | 83.97% | 80.40% | 86.50% | 83.48% |  |
| L3 | 83.62% | 79.60% | 87.69% | 84.70% |  |
| L4 | 83.46% | 79.45% | 87.65% | 84.65% |  |
| L5 | 83.87% | 79.85% | 87.97% | 85.00% |  |
| L6 | 87.82% | 85.29% | 90.86% | 88.92% |  |
| L7 | 88.10% | 85.16% | 92.72% | 90.92% |  |
| L8 | 93.19% | 91.83% | 95.78% | 94.83% |  |
| L9 | 92.25% | 90.24% | 96.99% | 96.19% |  |
| L10 | 93.46% | 91.72% | 97.61% | 96.95% |  |
| L11 | 95.18% | 93.87% | 98.00% | 97.44% |  |
| L12 | 95.38% | 94.12% | 98.48% | 98.05% | absolute + relative |
| L13 | 93.34% | 91.53% | 98.21% | 97.70% |  |
| L14 | 87.30% | 83.98% | 96.58% | 95.73% |  |
| L15 | 82.58% | 78.07% | 95.56% | 94.51% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 92.20% | 90.23% | 2.402 |
| Random direction | 92.00% | 90.09% | 2.398 |
| Relative-board direction | 97.60% | 92.75% | 0.840 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b8_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed000/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed000/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b8_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 194.8 s |
| board_probe_linear | 95.7 s |
| board_probe_mlp | 115.1 s |
| causal_intervention | 105.3 s |
| frozen_head_linear | 2139.2 s |
| frozen_head_mlp | 2966.0 s |
| head_tuning_linear | 97.6 s |
| head_tuning_mlp | 2.1 s |
