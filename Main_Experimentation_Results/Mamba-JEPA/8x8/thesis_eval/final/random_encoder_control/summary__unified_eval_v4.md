# Unified evaluation: mamba_jepa_v5_hd_infonce_allpos_b8__random_encoder_control

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `architecture_matched_random_encoder_control`
- **Checkpoint:** `final.pt` (step `0`, games `0`)
- **Checkpoint SHA-256:** `606dfff0905ba23dd52280e972c371deec31ff4931169955c0c413924775ec1e`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `0` games / `0` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 40.56% | 36.38% | 32.55% | 22.42% |
| MLP | 50.17% | 44.73% | 39.75% | 28.62% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L1 | 0.067 | 56.00% | 55.86% | 62.52% | 40.34% | 86.90% |
| LINEAR | relative | L1 | 0.067 | 58.42% | 58.28% | 64.60% | 44.59% | 86.60% |
| MLP | absolute | L0 | 0.000 | 58.20% | 58.16% | 64.67% | 43.00% | 88.51% |
| MLP | relative | L1 | 0.067 | 58.65% | 58.67% | 65.25% | 44.54% | 88.03% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.57% | 55.56% | 64.82% | 58.22% |  |
| L1 | 62.52% | 55.86% | 64.60% | 58.28% | absolute + relative |
| L2 | 62.13% | 55.52% | 64.18% | 57.89% |  |
| L3 | 61.80% | 55.16% | 63.83% | 57.49% |  |
| L4 | 61.56% | 54.90% | 63.53% | 57.15% |  |
| L5 | 61.34% | 54.64% | 63.30% | 56.85% |  |
| L6 | 61.19% | 54.46% | 63.15% | 56.66% |  |
| L7 | 61.04% | 54.27% | 62.93% | 56.36% |  |
| L8 | 60.89% | 54.11% | 62.80% | 56.21% |  |
| L9 | 60.70% | 53.91% | 62.56% | 55.97% |  |
| L10 | 60.55% | 53.75% | 62.44% | 55.83% |  |
| L11 | 60.46% | 53.65% | 62.33% | 55.70% |  |
| L12 | 60.36% | 53.55% | 62.21% | 55.57% |  |
| L13 | 60.20% | 53.36% | 62.08% | 55.42% |  |
| L14 | 60.03% | 53.20% | 61.96% | 55.30% |  |
| L15 | 59.97% | 53.11% | 61.90% | 55.21% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.67% | 58.16% | 65.16% | 58.50% | absolute |
| L1 | 64.24% | 57.67% | 65.25% | 58.67% | relative |
| L2 | 63.99% | 57.39% | 65.13% | 58.54% |  |
| L3 | 63.71% | 57.05% | 65.07% | 58.42% |  |
| L4 | 63.44% | 56.71% | 65.00% | 58.35% |  |
| L5 | 63.34% | 56.56% | 64.89% | 58.22% |  |
| L6 | 63.30% | 56.56% | 64.83% | 58.17% |  |
| L7 | 63.13% | 56.33% | 64.74% | 58.04% |  |
| L8 | 62.97% | 56.13% | 64.68% | 57.94% |  |
| L9 | 62.88% | 56.03% | 64.66% | 57.92% |  |
| L10 | 62.82% | 55.99% | 64.57% | 57.84% |  |
| L11 | 62.76% | 55.90% | 64.49% | 57.72% |  |
| L12 | 62.65% | 55.79% | 64.46% | 57.70% |  |
| L13 | 62.62% | 55.74% | 64.37% | 57.60% |  |
| L14 | 62.60% | 55.73% | 64.31% | 57.53% |  |
| L15 | 62.53% | 55.67% | 64.23% | 57.43% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/random_encoder_control/results__unified_eval_v4.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/random_encoder_control/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/random_encoder_control/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/random_encoder_control/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/random_encoder_control/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-JEPA/mamba_jepa_v5_hd_infonce_allpos_b8/thesis_eval/final/random_encoder_control/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 189.1 s |
| board_probe_linear | 96.6 s |
| board_probe_mlp | 116.5 s |
| frozen_head_linear | 1758.1 s |
| frozen_head_mlp | 3354.0 s |
| head_tuning_linear | 93.1 s |
| head_tuning_mlp | 2.2 s |
