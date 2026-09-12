# Unified evaluation: mamba_jepa_v5_hd_infonce_allpos_b16

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `93fdf9560afd7bc712ac2c2fdf32c682a59a89176fcdff697c62695ac3fa05cc`
- **Split manifest SHA-256:** `20dc4d4e6c57313038b202fb3f6f762a18beae2fe62617601bf77765f68a124e`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 98.41% | 97.58% | 96.55% | 90.42% |
| MLP | 98.24% | 97.41% | 96.38% | 90.12% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L13 | 0.867 | 66.20% | 66.18% | 74.28% | 49.36% | 99.81% |
| LINEAR | relative | L12 | 0.800 | 83.77% | 83.77% | 87.56% | 76.04% | 99.37% |
| MLP | absolute | L13 | 0.867 | 65.70% | 65.66% | 73.82% | 48.70% | 99.57% |
| MLP | relative | L13 | 0.867 | 81.26% | 81.26% | 85.58% | 72.46% | 99.03% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.66% | 56.51% | 65.55% | 57.67% |  |
| L1 | 68.38% | 60.26% | 69.66% | 61.93% |  |
| L2 | 68.83% | 60.73% | 71.79% | 64.68% |  |
| L3 | 68.78% | 60.66% | 72.51% | 65.67% |  |
| L4 | 71.36% | 63.24% | 75.84% | 69.19% |  |
| L5 | 71.11% | 63.00% | 79.05% | 73.61% |  |
| L6 | 72.13% | 64.04% | 80.08% | 74.60% |  |
| L7 | 72.60% | 64.52% | 80.44% | 74.91% |  |
| L8 | 72.94% | 64.86% | 80.59% | 74.99% |  |
| L9 | 72.00% | 63.87% | 85.94% | 82.22% |  |
| L10 | 72.82% | 64.72% | 86.62% | 82.91% |  |
| L11 | 73.64% | 65.55% | 87.27% | 83.53% |  |
| L12 | 74.06% | 65.98% | 87.56% | 83.77% | relative |
| L13 | 74.28% | 66.18% | 87.62% | 83.75% | absolute |
| L14 | 74.00% | 65.84% | 86.85% | 82.78% |  |
| L15 | 73.55% | 65.35% | 85.96% | 81.75% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.19% | 57.01% | 65.79% | 57.82% |  |
| L1 | 67.77% | 59.71% | 68.65% | 60.83% |  |
| L2 | 67.87% | 59.87% | 70.04% | 62.74% |  |
| L3 | 67.71% | 59.73% | 70.45% | 63.40% |  |
| L4 | 69.66% | 61.63% | 73.07% | 66.14% |  |
| L5 | 69.87% | 61.95% | 75.96% | 70.24% |  |
| L6 | 70.68% | 62.64% | 77.10% | 71.30% |  |
| L7 | 71.30% | 63.22% | 77.70% | 71.79% |  |
| L8 | 71.77% | 63.68% | 77.95% | 71.90% |  |
| L9 | 70.57% | 62.42% | 82.80% | 78.77% |  |
| L10 | 71.63% | 63.51% | 83.82% | 79.83% |  |
| L11 | 72.71% | 64.58% | 84.82% | 80.75% |  |
| L12 | 73.39% | 65.24% | 85.34% | 81.13% |  |
| L13 | 73.82% | 65.66% | 85.58% | 81.26% | absolute + relative |
| L14 | 73.50% | 65.27% | 84.53% | 79.91% |  |
| L15 | 72.98% | 64.70% | 83.51% | 78.70% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1643.6 s |
| board_probe_linear | 179.5 s |
| board_probe_mlp | 199.0 s |
| frozen_head_linear | 20518.5 s |
| frozen_head_mlp | 19288.3 s |
| head_tuning_linear | 1588.4 s |
| head_tuning_mlp | 10.3 s |
