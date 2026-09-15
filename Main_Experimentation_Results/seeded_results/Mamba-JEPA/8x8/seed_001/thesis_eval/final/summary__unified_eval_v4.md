# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `21fc832a9d851ed2d2951344f9d80e89610511cb77f065fde40b43805d24b26d`
- **Split manifest SHA-256:** `e7f5684124d57258e2e789b3efc86d5af6b40fb6c64f29d7bd03e63229dcb961`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.74% | 96.82% | 92.50% | 98.66% |
| MLP | 99.72% | 96.81% | 92.49% | 99.03% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 69.04% | 68.96% | 75.64% | 53.48% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 98.06% | 98.22% | 98.62% | 97.37% | 100.00% |
| MLP | absolute | L11 | 0.733 | 93.71% | 93.75% | 95.09% | 90.62% | 100.00% |
| MLP | relative | L12 | 0.800 | 97.85% | 98.04% | 98.48% | 97.12% | 99.99% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.59% | 55.55% | 64.83% | 58.22% |  |
| L1 | 73.49% | 66.64% | 78.26% | 72.47% |  |
| L2 | 73.83% | 67.00% | 81.18% | 76.23% |  |
| L3 | 72.78% | 65.95% | 87.66% | 84.79% |  |
| L4 | 73.92% | 67.02% | 89.16% | 86.30% |  |
| L5 | 73.93% | 67.03% | 89.20% | 86.38% |  |
| L6 | 72.79% | 65.79% | 93.47% | 91.90% |  |
| L7 | 73.30% | 66.34% | 93.88% | 92.40% |  |
| L8 | 73.22% | 66.25% | 95.97% | 95.00% |  |
| L9 | 74.27% | 67.43% | 96.85% | 96.05% |  |
| L10 | 75.54% | 68.84% | 97.91% | 97.31% |  |
| L11 | 75.56% | 68.87% | 98.17% | 97.66% |  |
| L12 | 75.64% | 68.96% | 98.62% | 98.22% | absolute + relative |
| L13 | 75.38% | 68.64% | 98.35% | 97.88% |  |
| L14 | 74.78% | 67.97% | 97.04% | 96.31% |  |
| L15 | 74.35% | 67.53% | 96.16% | 95.28% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.08% | 58.74% | 65.23% | 58.64% |  |
| L1 | 75.33% | 69.10% | 77.51% | 71.64% |  |
| L2 | 76.81% | 70.92% | 80.15% | 75.03% |  |
| L3 | 83.75% | 80.27% | 86.31% | 83.35% |  |
| L4 | 84.71% | 80.96% | 88.08% | 85.08% |  |
| L5 | 83.35% | 79.21% | 88.02% | 85.00% |  |
| L6 | 89.78% | 87.74% | 92.15% | 90.44% |  |
| L7 | 88.90% | 86.47% | 92.67% | 91.02% |  |
| L8 | 92.31% | 90.83% | 94.99% | 93.93% |  |
| L9 | 92.20% | 90.39% | 96.20% | 95.31% |  |
| L10 | 93.38% | 91.58% | 97.75% | 97.10% |  |
| L11 | 95.09% | 93.75% | 98.04% | 97.49% | absolute |
| L12 | 93.59% | 91.84% | 98.48% | 98.04% | relative |
| L13 | 86.28% | 82.56% | 98.01% | 97.44% |  |
| L14 | 80.10% | 74.79% | 96.34% | 95.45% |  |
| L15 | 75.96% | 69.64% | 95.25% | 94.17% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.40% | 90.17% | 2.374 |
| Random direction | 90.70% | 89.98% | 2.340 |
| Relative-board direction | 95.60% | 91.76% | 1.134 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_jepa_b8_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed001/runs/selected/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/mamba_jepa_b8_seed001/runs/selected/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_jepa_b8_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_jepa_b8_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 196.2 s |
| board_probe_linear | 101.6 s |
| board_probe_mlp | 122.0 s |
| causal_intervention | 110.9 s |
| frozen_head_linear | 2130.8 s |
| frozen_head_mlp | 3194.7 s |
| head_tuning_linear | 95.9 s |
| head_tuning_mlp | 2.2 s |
