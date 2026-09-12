# Unified evaluation: jepa_v5_infonce_b8_run_001_hd_allpos_bf16__random_encoder_control

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `architecture_matched_random_encoder_control`
- **Checkpoint:** `final.pt` (step `0`, games `0`)
- **Checkpoint SHA-256:** `f23723a1e12724b0c75b1f93efde023ad8b2b33d89aa62405ccba9bb5e0d56e7`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `d6c865f5c3ef1ef7834aaadedb7a01e3bcdb20c134f4745164d6f2b8b41285c7`
- **Exact pretraining budget:** `0` games / `0` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 62.13% | 56.21% | 51.28% | 36.09% |
| MLP | 74.04% | 68.34% | 62.86% | 50.50% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L1 | 0.125 | 63.94% | 63.84% | 70.47% | 48.46% | 94.68% |
| LINEAR | relative | L1 | 0.125 | 65.92% | 65.87% | 72.20% | 51.90% | 94.53% |
| MLP | absolute | L1 | 0.125 | 64.74% | 64.70% | 71.07% | 49.93% | 94.32% |
| MLP | relative | L1 | 0.125 | 65.67% | 65.71% | 72.09% | 51.97% | 94.22% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.60% | 55.58% | 64.83% | 58.26% |  |
| L1 | 70.47% | 63.84% | 72.20% | 65.87% | absolute + relative |
| L2 | 70.08% | 63.42% | 71.73% | 65.31% |  |
| L3 | 69.65% | 62.99% | 71.31% | 64.84% |  |
| L4 | 69.37% | 62.69% | 71.03% | 64.51% |  |
| L5 | 69.10% | 62.38% | 70.78% | 64.20% |  |
| L6 | 68.97% | 62.25% | 70.62% | 64.01% |  |
| L7 | 68.74% | 62.00% | 70.47% | 63.79% |  |
| L8 | 68.63% | 61.92% | 70.34% | 63.66% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.65% | 58.13% | 65.16% | 58.51% |  |
| L1 | 71.07% | 64.70% | 72.09% | 65.71% | absolute + relative |
| L2 | 70.38% | 63.89% | 71.65% | 65.21% |  |
| L3 | 69.95% | 63.44% | 71.22% | 64.78% |  |
| L4 | 69.57% | 63.03% | 70.95% | 64.43% |  |
| L5 | 69.40% | 62.92% | 70.72% | 64.28% |  |
| L6 | 69.21% | 62.68% | 70.56% | 64.00% |  |
| L7 | 68.90% | 62.31% | 70.39% | 63.85% |  |
| L8 | 68.82% | 62.25% | 70.30% | 63.71% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/random_encoder_control/results__unified_eval_v4.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/random_encoder_control/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/random_encoder_control/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/random_encoder_control/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/random_encoder_control/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/random_encoder_control/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| head_tuning_linear | 91.7 s |
| frozen_head_linear | 1966.5 s |
| head_tuning_mlp | 2.1 s |
| frozen_head_mlp | 2793.5 s |
| board_feature_extraction | 158.5 s |
| board_probe_linear | 57.3 s |
| board_probe_mlp | 67.0 s |
