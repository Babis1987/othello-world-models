# Unified evaluation: transformer_jepa_v5_hd_infonce_allpos_b12

- **Protocol:** `unified_eval_v4`
- **Board:** `12x12`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `8cda977e9fd018c8b4943cd941885f673edd35df9ff45223b2f49aa0117c1462`
- **Split manifest SHA-256:** `0ef44a8ff8095d9eb17b6bd95af4ebac29288521851bdfc9edbfb580bd185e4e`
- **Data-manifest SHA-256:** `1ffe57c995aad5523da285757d3d9813a52c0eb2b1bca28294404e1f973c2c31`
- **Per-shard corpus identity:** `78ea34919eb892e91d8621569afba2ced653c7fe8b5b2c97bfdd01d369074610`
- **Project-source SHA-256:** `d6c865f5c3ef1ef7834aaadedb7a01e3bcdb20c134f4745164d6f2b8b41285c7`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 98.52% | 97.27% | 95.54% | 93.30% |
| MLP | 98.38% | 97.12% | 95.38% | 92.94% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 67.23% | 67.23% | 74.90% | 50.84% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 88.49% | 88.48% | 91.21% | 82.81% | 99.97% |
| MLP | absolute | L6 | 0.750 | 75.84% | 75.84% | 81.49% | 63.77% | 99.97% |
| MLP | relative | L6 | 0.750 | 87.42% | 87.40% | 90.38% | 81.24% | 99.92% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 63.70% | 55.88% | 65.00% | 57.53% |  |
| L1 | 68.66% | 60.99% | 72.73% | 66.29% |  |
| L2 | 73.29% | 65.54% | 80.69% | 75.20% |  |
| L3 | 74.60% | 66.89% | 83.83% | 78.87% |  |
| L4 | 74.89% | 67.21% | 85.44% | 80.91% |  |
| L5 | 74.90% | 67.23% | 89.92% | 86.77% | absolute |
| L6 | 74.81% | 67.12% | 91.21% | 88.48% | relative |
| L7 | 74.27% | 66.46% | 90.99% | 88.27% |  |
| L8 | 73.80% | 65.99% | 90.25% | 87.44% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.00% | 57.44% | 65.27% | 57.69% |  |
| L1 | 69.00% | 61.74% | 71.10% | 64.41% |  |
| L2 | 74.10% | 66.86% | 78.67% | 72.89% |  |
| L3 | 76.53% | 69.53% | 82.10% | 76.76% |  |
| L4 | 76.52% | 69.37% | 83.98% | 79.03% |  |
| L5 | 80.48% | 74.53% | 89.00% | 85.59% |  |
| L6 | 81.49% | 75.84% | 90.38% | 87.40% | absolute + relative |
| L7 | 79.58% | 73.45% | 89.79% | 86.76% |  |
| L8 | 76.38% | 69.46% | 88.72% | 85.54% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b12/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| head_tuning_linear | 506.4 s |
| frozen_head_linear | 7008.4 s |
| head_tuning_mlp | 5.5 s |
| frozen_head_mlp | 6266.2 s |
| board_feature_extraction | 570.2 s |
| board_probe_linear | 83.5 s |
| board_probe_mlp | 95.3 s |
