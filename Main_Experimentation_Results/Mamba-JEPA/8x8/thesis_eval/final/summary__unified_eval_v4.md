# Unified evaluation: mamba_jepa_v5_hd_infonce_allpos_b8

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `6fb692303ad514c5122cb1e500f30b5f7576125ce81317e3a6f0a428612c351f`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
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
| LINEAR | 99.72% | 96.81% | 92.50% | 98.61% |
| MLP | 99.72% | 96.80% | 92.49% | 98.90% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L12 | 0.800 | 68.90% | 68.92% | 75.60% | 53.42% | 100.00% |
| LINEAR | relative | L13 | 0.867 | 97.80% | 98.04% | 98.48% | 97.12% | 99.97% |
| MLP | absolute | L9 | 0.600 | 89.39% | 89.31% | 91.41% | 84.40% | 99.12% |
| MLP | relative | L12 | 0.800 | 97.58% | 97.82% | 98.30% | 96.77% | 99.99% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.61% | 55.55% | 64.82% | 58.16% |  |
| L1 | 71.50% | 64.77% | 76.13% | 70.45% |  |
| L2 | 70.22% | 63.55% | 83.90% | 80.87% |  |
| L3 | 73.05% | 66.24% | 87.22% | 84.19% |  |
| L4 | 73.91% | 67.06% | 88.63% | 85.72% |  |
| L5 | 74.35% | 67.49% | 89.46% | 86.66% |  |
| L6 | 73.92% | 66.99% | 93.36% | 91.64% |  |
| L7 | 74.46% | 67.60% | 93.94% | 92.33% |  |
| L8 | 74.75% | 67.93% | 94.19% | 92.65% |  |
| L9 | 74.49% | 67.66% | 97.09% | 96.32% |  |
| L10 | 74.68% | 67.88% | 97.41% | 96.74% |  |
| L11 | 75.45% | 68.73% | 98.20% | 97.69% |  |
| L12 | 75.60% | 68.92% | 98.44% | 98.00% | absolute |
| L13 | 75.49% | 68.77% | 98.48% | 98.04% | relative |
| L14 | 74.76% | 67.95% | 97.10% | 96.37% |  |
| L15 | 74.51% | 67.68% | 96.54% | 95.70% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.08% | 58.73% | 65.26% | 58.67% |  |
| L1 | 73.84% | 67.91% | 75.62% | 69.92% |  |
| L2 | 80.42% | 76.93% | 82.96% | 79.95% |  |
| L3 | 82.23% | 78.16% | 86.11% | 82.94% |  |
| L4 | 83.22% | 79.07% | 87.62% | 84.56% |  |
| L5 | 83.57% | 79.32% | 88.53% | 85.53% |  |
| L6 | 88.53% | 85.80% | 92.20% | 90.29% |  |
| L7 | 87.98% | 84.92% | 93.07% | 91.30% |  |
| L8 | 87.32% | 84.01% | 93.39% | 91.65% |  |
| L9 | 91.41% | 89.31% | 96.51% | 95.64% | absolute |
| L10 | 90.51% | 88.12% | 96.89% | 96.12% |  |
| L11 | 85.62% | 81.71% | 98.06% | 97.51% |  |
| L12 | 81.24% | 76.10% | 98.30% | 97.82% | relative |
| L13 | 87.39% | 83.96% | 98.23% | 97.72% |  |
| L14 | 81.00% | 75.95% | 96.39% | 95.50% |  |
| L15 | 78.21% | 72.44% | 95.66% | 94.64% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `frozen_mlp`
- **Selected alpha:** `4`
- **Interpretation scope:** JEPA encoder plus post-hoc frozen MLP readout

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.50% | 90.06% | 2.392 |
| Random direction | 92.20% | 89.89% | 2.358 |
| Relative-board direction | 97.00% | 93.26% | 0.832 |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 187.1 s |
| board_probe_linear | 96.6 s |
| board_probe_mlp | 116.8 s |
| causal_intervention | 106.2 s |
| frozen_head_linear | 2175.3 s |
| frozen_head_mlp | 2894.7 s |
| head_tuning_linear | 99.9 s |
| head_tuning_mlp | 2.1 s |
