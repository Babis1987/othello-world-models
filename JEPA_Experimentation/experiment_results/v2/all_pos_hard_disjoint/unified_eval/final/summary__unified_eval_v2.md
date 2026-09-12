# Unified evaluation: jepa_v2_ema_b8_run_001_hd_allpos_bf16

- **Protocol:** `unified_eval_v2`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v2_smooth_l1_hard_disjoint`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `8abcb8c00daa78f8dd199f34e223f967965403759057fdb457f5b087df9699df`
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
| LINEAR | 98.87% | 95.50% | 90.59% | 83.06% |
| MLP | 98.22% | 95.12% | 90.48% | 88.95% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L1 | 0.125 | 68.54% | 68.68% | 75.40% | 53.11% | 99.93% |
| LINEAR | relative | L2 | 0.250 | 88.26% | 88.24% | 90.81% | 82.59% | 99.85% |
| MLP | absolute | L1 | 0.125 | 74.54% | 74.47% | 79.91% | 61.84% | 99.79% |
| MLP | relative | L2 | 0.250 | 86.71% | 86.65% | 89.59% | 80.35% | 99.77% |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v2_ema_b8_run_001_hd_allpos_bf16/unified_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v2_ema_b8_run_001_hd_allpos_bf16/unified_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v2_ema_b8_run_001_hd_allpos_bf16/unified_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v2_ema_b8_run_001_hd_allpos_bf16/unified_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v2_ema_b8_run_001_hd_allpos_bf16/unified_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v2_ema_b8_run_001_hd_allpos_bf16/unified_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 2661.0 s |
| frozen_head_mlp | 2680.0 s |
| board_feature_extraction | 232.9 s |
| board_probe_linear | 12.2 s |
| board_probe_mlp | 16.2 s |
