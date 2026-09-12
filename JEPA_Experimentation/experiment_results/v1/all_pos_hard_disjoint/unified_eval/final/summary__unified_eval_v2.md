# Unified evaluation: jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16

- **Protocol:** `unified_eval_v2`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `jepa_v1_vicreg_hard_disjoint`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `768d228cb0a1aa8efea7cfc721cbfcc5147d45580e75fea573c868bdac703743`
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
| LINEAR | 98.99% | 95.66% | 90.97% | 86.83% |
| MLP | 98.29% | 95.07% | 90.52% | 91.33% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L2 | 0.250 | 68.95% | 68.80% | 75.51% | 53.24% | 100.00% |
| LINEAR | relative | L2 | 0.250 | 90.33% | 90.17% | 92.33% | 85.38% | 99.99% |
| MLP | absolute | L2 | 0.250 | 78.47% | 78.05% | 82.76% | 67.10% | 99.99% |
| MLP | relative | L2 | 0.250 | 89.44% | 89.23% | 91.65% | 84.09% | 99.96% |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/unified_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/unified_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/unified_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/unified_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/unified_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_allpos_bf16/unified_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| frozen_head_linear | 124.2 s |
| frozen_head_mlp | 127.0 s |
| board_feature_extraction | 70.3 s |
| board_probe_linear | 3.7 s |
| board_probe_mlp | 4.8 s |
