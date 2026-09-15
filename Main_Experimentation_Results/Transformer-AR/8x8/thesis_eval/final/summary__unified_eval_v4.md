# Unified evaluation: transformer_ar_b8_bf16

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `a8372296a542876e0b4e5cee22a1e6940f9d3de6772e12835314e982a52b56d2`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.73% | 96.78% | 92.45% | 98.48% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L3 | 0.375 | 69.01% | 68.88% | 75.57% | 53.37% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.66% | 96.91% | 97.59% | 95.41% | 99.99% |
| MLP | absolute | L5 | 0.625 | 93.79% | 93.88% | 95.20% | 90.83% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.34% | 96.61% | 97.38% | 95.01% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.69% | 55.79% | 64.79% | 58.21% |  |
| L1 | 75.21% | 68.50% | 87.72% | 84.31% |  |
| L2 | 75.46% | 68.75% | 91.74% | 89.39% |  |
| L3 | 75.57% | 68.88% | 94.54% | 92.99% | absolute |
| L4 | 75.60% | 68.91% | 96.09% | 94.98% |  |
| L5 | 75.54% | 68.83% | 97.23% | 96.45% |  |
| L6 | 75.38% | 68.64% | 97.59% | 96.91% | relative |
| L7 | 75.19% | 68.41% | 97.54% | 96.86% |  |
| L8 | 75.17% | 68.39% | 97.51% | 96.82% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.13% | 58.78% | 65.24% | 58.57% |  |
| L1 | 84.41% | 80.29% | 87.16% | 83.64% |  |
| L2 | 88.53% | 85.42% | 91.31% | 88.84% |  |
| L3 | 92.03% | 89.85% | 94.39% | 92.80% |  |
| L4 | 93.77% | 92.07% | 95.94% | 94.78% |  |
| L5 | 95.20% | 93.88% | 97.09% | 96.24% | absolute |
| L6 | 92.83% | 90.88% | 97.38% | 96.61% | relative |
| L7 | 87.01% | 83.48% | 97.22% | 96.44% |  |
| L8 | 85.57% | 81.65% | 97.17% | 96.38% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `2`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.40% | 89.60% | 2.292 |
| Random direction | 90.40% | 89.72% | 2.274 |
| Relative-board direction | 97.30% | 95.88% | 0.432 |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b8_bf16/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b8_bf16/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b8_bf16/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b8_bf16/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| native_ar | 1167.9 s |
| board_feature_extraction | 147.8 s |
| board_probe_linear | 61.9 s |
| board_probe_mlp | 66.9 s |
| causal_intervention | 65.5 s |
