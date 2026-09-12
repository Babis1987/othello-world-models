# Unified evaluation: mamba_jepa_v5_hd_infonce_allpos_b8

- **Protocol:** `unified_eval_v2`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `jepa_v1_infonce_hard_disjoint_future`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `6fb692303ad514c5122cb1e500f30b5f7576125ce81317e3a6f0a428612c351f`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `4fdfec5bb0841fd14c03b7c6e527d88a640c41407e6aa1f5c323d7157a9aa1ae`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm 2.3.2.post1` / `causal-conv1d 1.6.2.post1`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.72% | 96.82% | 92.51% | 98.66% |
| MLP | 99.55% | 96.68% | 92.40% | 99.00% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L11 | 0.733 | 68.34% | 68.32% | 75.14% | 52.57% | 99.97% |
| LINEAR | relative | L12 | 0.800 | 97.43% | 97.65% | 98.17% | 96.52% | 99.99% |
| MLP | absolute | L9 | 0.600 | 75.03% | 74.76% | 79.92% | 62.77% | 98.78% |
| MLP | relative | L12 | 0.800 | 96.97% | 97.16% | 97.78% | 95.80% | 99.96% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.82% | 55.84% | 64.96% | 58.39% |  |
| L1 | 70.59% | 63.76% | 74.82% | 69.03% |  |
| L2 | 69.05% | 62.25% | 82.38% | 79.15% |  |
| L3 | 72.07% | 65.17% | 85.76% | 82.55% |  |
| L4 | 73.10% | 66.16% | 87.17% | 84.04% |  |
| L5 | 73.55% | 66.54% | 88.10% | 85.06% |  |
| L6 | 73.01% | 65.94% | 92.11% | 90.16% |  |
| L7 | 73.79% | 66.79% | 92.90% | 91.04% |  |
| L8 | 74.23% | 67.30% | 93.18% | 91.38% |  |
| L9 | 73.79% | 66.82% | 96.32% | 95.38% |  |
| L10 | 73.91% | 66.93% | 96.67% | 95.85% |  |
| L11 | 75.14% | 68.32% | 97.91% | 97.32% | absolute |
| L12 | 75.27% | 68.49% | 98.17% | 97.65% | relative |
| L13 | 75.15% | 68.34% | 98.02% | 97.46% |  |
| L14 | 73.63% | 66.69% | 95.23% | 94.19% |  |
| L15 | 73.19% | 66.29% | 94.16% | 92.99% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.69% | 58.19% | 65.30% | 58.65% |  |
| L1 | 71.15% | 64.73% | 73.54% | 67.85% |  |
| L2 | 73.23% | 67.95% | 80.45% | 77.20% |  |
| L3 | 74.30% | 68.35% | 83.64% | 80.36% |  |
| L4 | 74.59% | 68.34% | 85.20% | 81.91% |  |
| L5 | 76.08% | 69.99% | 86.07% | 82.76% |  |
| L6 | 78.96% | 73.81% | 90.01% | 87.80% |  |
| L7 | 78.57% | 73.05% | 91.19% | 89.06% |  |
| L8 | 78.54% | 72.91% | 91.55% | 89.46% |  |
| L9 | 79.92% | 74.76% | 94.98% | 93.85% | absolute |
| L10 | 78.76% | 73.27% | 95.38% | 94.38% |  |
| L11 | 76.59% | 70.22% | 97.48% | 96.79% |  |
| L12 | 75.45% | 68.76% | 97.78% | 97.16% | relative |
| L13 | 75.67% | 69.03% | 97.36% | 96.64% |  |
| L14 | 73.24% | 66.46% | 92.84% | 91.36% |  |
| L15 | 72.30% | 65.47% | 91.10% | 89.36% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/unified_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/unified_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/unified_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/unified_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/unified_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/unified_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 312.9 s |
| frozen_head_mlp | 271.8 s |
| board_feature_extraction | 64.4 s |
| board_probe_linear | 10.1 s |
| board_probe_mlp | 12.1 s |
