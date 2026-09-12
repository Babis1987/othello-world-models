# Unified evaluation: transformer_ar_b12

- **Protocol:** `unified_eval_v3`
- **Board:** `12x12`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `cc3270ad1e75d9198b247c3228fe1999f491816c151d49285a12e2f0d942f175`
- **Split manifest SHA-256:** `0ef44a8ff8095d9eb17b6bd95af4ebac29288521851bdfc9edbfb580bd185e4e`
- **Data-manifest SHA-256:** `1ffe57c995aad5523da285757d3d9813a52c0eb2b1bca28294404e1f973c2c31`
- **Per-shard corpus identity:** `78ea34919eb892e91d8621569afba2ced653c7fe8b5b2c97bfdd01d369074610`
- **Project-source SHA-256:** `d8e32f03a58dc21f2a5a7871756c819577c8aeea2292f2ec349fd23da4f0ce6d`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.42% | 98.12% | 96.34% | 93.13% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 67.31% | 67.30% | 74.96% | 50.96% | 99.99% |
| LINEAR | relative | L7 | 0.875 | 89.17% | 89.17% | 91.74% | 83.83% | 99.98% |
| MLP | absolute | L5 | 0.625 | 72.54% | 72.59% | 79.00% | 58.91% | 99.96% |
| MLP | relative | L7 | 0.875 | 88.02% | 88.01% | 90.86% | 82.15% | 99.94% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 63.75% | 55.94% | 65.03% | 57.56% |  |
| L1 | 70.20% | 62.45% | 74.63% | 68.23% |  |
| L2 | 74.28% | 66.54% | 79.75% | 73.65% |  |
| L3 | 74.86% | 67.19% | 82.20% | 76.71% |  |
| L4 | 74.88% | 67.21% | 86.28% | 82.03% |  |
| L5 | 74.96% | 67.30% | 90.12% | 87.04% | absolute |
| L6 | 74.89% | 67.21% | 91.29% | 88.57% |  |
| L7 | 74.76% | 67.04% | 91.74% | 89.17% | relative |
| L8 | 74.72% | 66.99% | 91.73% | 89.16% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.02% | 57.47% | 65.27% | 57.72% |  |
| L1 | 69.86% | 62.40% | 72.59% | 65.95% |  |
| L2 | 74.01% | 66.34% | 78.07% | 71.62% |  |
| L3 | 74.94% | 67.33% | 80.73% | 74.82% |  |
| L4 | 75.24% | 67.70% | 84.66% | 79.89% |  |
| L5 | 79.00% | 72.59% | 89.16% | 85.79% | absolute |
| L6 | 77.08% | 70.07% | 90.45% | 87.46% |  |
| L7 | 78.26% | 71.62% | 90.86% | 88.01% | relative |
| L8 | 75.90% | 68.53% | 90.80% | 87.93% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b12/thesis_eval/final/results__unified_eval_v3.json`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b12/thesis_eval/final/linear_board_probe__unified_eval_v3.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b12/thesis_eval/final/mlp_board_probe__unified_eval_v3.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b12/thesis_eval/final/position_manifest__unified_eval_v3.json`

| Stage | Wall time |
|---|---:|
| native_ar | 5451.0 s |
| board_feature_extraction | 570.9 s |
| board_probe_linear | 83.9 s |
| board_probe_mlp | 88.9 s |
