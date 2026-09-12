# Unified evaluation: jepa_v5_infonce_b8_run_001_hd_allpos_bf16

- **Protocol:** `unified_eval_v2`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `0c4eda770574ddd8577a7bf7ca0422a659e374232226e8b8ea7359c7d3447001`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `308b542c21b15ff83c9b778c1f894206d956c8b0ab874d8d0c649ea9e35f1e7e`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 99.42% | 96.56% | 92.27% | 97.79% |
| MLP | 99.27% | 96.39% | 92.11% | 97.86% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L4 | 0.500 | 68.89% | 68.68% | 75.41% | 53.06% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 95.99% | 96.20% | 97.04% | 94.38% | 99.97% |
| MLP | absolute | L5 | 0.625 | 85.92% | 85.90% | 88.93% | 78.87% | 100.00% |
| MLP | relative | L5 | 0.625 | 95.49% | 95.67% | 96.64% | 93.58% | 100.00% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.60% | 55.79% | 64.89% | 58.57% |  |
| L1 | 74.10% | 67.26% | 85.77% | 82.03% |  |
| L2 | 75.24% | 68.48% | 91.11% | 88.65% |  |
| L3 | 75.39% | 68.66% | 93.73% | 91.97% |  |
| L4 | 75.41% | 68.68% | 95.75% | 94.56% | absolute |
| L5 | 75.43% | 68.71% | 96.96% | 96.11% |  |
| L6 | 75.15% | 68.36% | 97.04% | 96.20% | relative |
| L7 | 74.33% | 67.47% | 95.94% | 94.92% |  |
| L8 | 73.90% | 67.07% | 95.21% | 94.13% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.02% | 58.61% | 65.29% | 58.74% |  |
| L1 | 78.07% | 72.55% | 83.86% | 79.72% |  |
| L2 | 83.28% | 78.77% | 90.12% | 87.34% |  |
| L3 | 85.50% | 81.55% | 93.14% | 91.18% |  |
| L4 | 87.52% | 84.10% | 95.34% | 94.00% |  |
| L5 | 88.93% | 85.90% | 96.64% | 95.67% | absolute + relative |
| L6 | 84.64% | 80.47% | 96.32% | 95.26% |  |
| L7 | 80.96% | 76.07% | 94.56% | 93.26% |  |
| L8 | 76.44% | 70.52% | 93.45% | 92.05% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/thesis_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 130.5 s |
| frozen_head_mlp | 122.6 s |
| board_feature_extraction | 67.1 s |
| board_probe_linear | 4.7 s |
| board_probe_mlp | 5.9 s |
