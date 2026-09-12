# Unified evaluation: jepa_v4_multi_pos_k4_hd_run_001_allpos_bf16

- **Protocol:** `unified_eval_v2`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v4_smooth_l1_hard_disjoint`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `a7ac57ac972ea358da7c23fbb4c9015f9c292842b25979700844359f82236ba0`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `4233de4099701ae0d31629cff5bc04787741d43e39ba117b3a9036dfd9b149a4`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts (primary comparison)

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 97.48% | 95.15% | 90.86% | 87.76% |
| MLP | 97.68% | 95.00% | 90.76% | 91.31% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L4 | 0.500 | 76.21% | 76.29% | 81.36% | 64.51% | 99.90% |
| LINEAR | relative | L3 | 0.375 | 89.42% | 89.43% | 91.76% | 84.30% | 99.96% |
| MLP | absolute | L2 | 0.250 | 76.35% | 76.04% | 81.17% | 64.10% | 99.96% |
| MLP | relative | L3 | 0.375 | 88.36% | 88.32% | 90.90% | 82.68% | 99.94% |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_k4_hd_run_001_allpos_bf16/unified_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_k4_hd_run_001_allpos_bf16/unified_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_k4_hd_run_001_allpos_bf16/unified_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_k4_hd_run_001_allpos_bf16/unified_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_k4_hd_run_001_allpos_bf16/unified_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_k4_hd_run_001_allpos_bf16/unified_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 557.8 s |
| frozen_head_mlp | 579.8 s |
| board_feature_extraction | 180.2 s |
| board_probe_linear | 11.1 s |
| board_probe_mlp | 15.2 s |
