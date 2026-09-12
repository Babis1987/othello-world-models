# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `efc79dc529c0351cac7dbf9fde3f7716493a16f5649f6adfdca647577c2fcd23`
- **Split manifest SHA-256:** `7cd5eadbc9d4d528b7de1f989a0fa8abc596b3801f3cca52299d290b833669de`
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
| LINEAR | 99.74% | 96.82% | 92.50% | 98.58% |
| MLP | 99.73% | 96.80% | 92.49% | 98.97% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 69.11% | 69.06% | 75.71% | 53.63% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 98.00% | 98.21% | 98.61% | 97.35% | 100.00% |
| MLP | absolute | L12 | 0.800 | 94.85% | 94.94% | 96.02% | 92.41% | 100.00% |
| MLP | relative | L12 | 0.800 | 97.80% | 98.03% | 98.48% | 97.10% | 99.99% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.62% | 55.58% | 64.80% | 58.19% |  |
| L1 | 74.05% | 67.21% | 79.18% | 73.53% |  |
| L2 | 73.18% | 66.35% | 86.43% | 83.12% |  |
| L3 | 73.22% | 66.34% | 86.99% | 83.80% |  |
| L4 | 73.86% | 66.99% | 87.62% | 84.44% |  |
| L5 | 73.36% | 66.44% | 89.05% | 86.39% |  |
| L6 | 71.84% | 64.84% | 94.73% | 93.51% |  |
| L7 | 73.75% | 66.84% | 95.61% | 94.48% |  |
| L8 | 74.19% | 67.31% | 96.14% | 95.15% |  |
| L9 | 74.57% | 67.74% | 96.84% | 96.01% |  |
| L10 | 75.34% | 68.61% | 97.73% | 97.09% |  |
| L11 | 75.61% | 68.93% | 98.11% | 97.57% |  |
| L12 | 75.71% | 69.06% | 98.61% | 98.21% | absolute + relative |
| L13 | 75.47% | 68.76% | 98.37% | 97.90% |  |
| L14 | 74.92% | 68.14% | 97.18% | 96.47% |  |
| L15 | 75.29% | 68.70% | 96.33% | 95.47% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.06% | 58.68% | 65.28% | 58.69% |  |
| L1 | 75.96% | 69.72% | 78.45% | 72.67% |  |
| L2 | 82.32% | 78.21% | 85.38% | 81.94% |  |
| L3 | 82.39% | 78.27% | 85.76% | 82.41% |  |
| L4 | 81.95% | 77.46% | 86.47% | 83.08% |  |
| L5 | 83.57% | 79.69% | 87.71% | 84.84% |  |
| L6 | 88.94% | 86.97% | 93.48% | 92.23% |  |
| L7 | 86.62% | 83.39% | 94.80% | 93.57% |  |
| L8 | 84.41% | 80.45% | 95.40% | 94.30% |  |
| L9 | 85.36% | 81.54% | 96.25% | 95.34% |  |
| L10 | 91.47% | 89.20% | 97.50% | 96.79% |  |
| L11 | 95.31% | 94.03% | 97.98% | 97.41% |  |
| L12 | 96.02% | 94.94% | 98.48% | 98.03% | absolute + relative |
| L13 | 94.07% | 92.47% | 98.03% | 97.46% |  |
| L14 | 87.29% | 83.97% | 96.53% | 95.67% |  |
| L15 | 82.17% | 77.55% | 95.44% | 94.35% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.40% | 90.15% | 2.350 |
| Random direction | 90.90% | 89.93% | 2.368 |
| Relative-board direction | 96.50% | 92.97% | 0.800 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b8_seed003/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed003/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed003/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed003/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed003/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b8_seed003/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 2105.7 s |
| head_tuning_linear | 93.6 s |
| head_tuning_mlp | 108.9 s |
| frozen_head_mlp | 3271.8 s |
| board_feature_extraction | 191.2 s |
| board_probe_linear | 95.4 s |
| board_probe_mlp | 113.5 s |
| causal_intervention | 104.5 s |
