# Unified evaluation: jepa_v5_infonce_b8_run_001_hd_allpos_bf16

- **Protocol:** `unified_eval_v2`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v1_infonce_hard_disjoint`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `0c4eda770574ddd8577a7bf7ca0422a659e374232226e8b8ea7359c7d3447001`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `7171920ccde083a981b6b51ac620e6f60bc16bc8343fc298ed12d589a6ca5f9a`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts (primary comparison)

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

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/unified_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/unified_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/unified_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/unified_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/unified_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v5_infonce_b8_run_001_hd_allpos_bf16/unified_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 132.9 s |
| frozen_head_mlp | 126.2 s |
| board_feature_extraction | 70.5 s |
| board_probe_linear | 3.6 s |
| board_probe_mlp | 4.7 s |
