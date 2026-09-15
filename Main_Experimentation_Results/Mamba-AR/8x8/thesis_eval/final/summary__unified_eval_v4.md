# Unified evaluation: mamba_ar_8x8_bf16

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `84968ae53eb2344db59aa9f1ee647fbdffdce86bb3010e4a70c361b4104c2b4d`
- **Split manifest SHA-256:** `9b203eaa68ee72bda13dbeae13566df1869b618995743b21c52397aaf9b5146f`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.85% | 96.90% | 92.57% | 99.34% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L11 | 0.733 | 68.82% | 68.78% | 75.50% | 53.22% | 100.00% |
| LINEAR | relative | L12 | 0.800 | 97.63% | 97.85% | 98.32% | 96.81% | 99.99% |
| MLP | absolute | L10 | 0.667 | 86.45% | 86.55% | 89.43% | 79.85% | 99.98% |
| MLP | relative | L12 | 0.800 | 97.37% | 97.62% | 98.16% | 96.52% | 99.97% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.59% | 55.55% | 64.78% | 58.15% |  |
| L1 | 73.42% | 66.59% | 77.85% | 72.00% |  |
| L2 | 72.21% | 65.37% | 85.95% | 82.76% |  |
| L3 | 71.99% | 65.14% | 88.12% | 85.53% |  |
| L4 | 73.51% | 66.61% | 90.71% | 88.38% |  |
| L5 | 73.50% | 66.57% | 91.62% | 89.54% |  |
| L6 | 73.92% | 67.02% | 92.95% | 91.17% |  |
| L7 | 74.07% | 67.17% | 93.88% | 92.33% |  |
| L8 | 74.58% | 67.73% | 94.78% | 93.39% |  |
| L9 | 75.41% | 68.67% | 95.98% | 94.85% |  |
| L10 | 75.44% | 68.70% | 97.13% | 96.34% |  |
| L11 | 75.50% | 68.78% | 98.09% | 97.55% | absolute |
| L12 | 75.45% | 68.72% | 98.32% | 97.85% | relative |
| L13 | 75.25% | 68.47% | 98.29% | 97.81% |  |
| L14 | 74.41% | 67.51% | 97.49% | 96.85% |  |
| L15 | 74.44% | 67.54% | 97.50% | 96.86% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.06% | 58.72% | 65.25% | 58.63% |  |
| L1 | 75.08% | 68.80% | 77.19% | 71.33% |  |
| L2 | 81.86% | 77.95% | 84.80% | 81.48% |  |
| L3 | 82.95% | 79.43% | 86.82% | 84.13% |  |
| L4 | 82.58% | 78.33% | 89.66% | 87.19% |  |
| L5 | 79.62% | 74.47% | 90.54% | 88.29% |  |
| L6 | 81.15% | 76.33% | 91.87% | 89.87% |  |
| L7 | 81.62% | 76.87% | 92.94% | 91.19% |  |
| L8 | 85.34% | 81.50% | 93.92% | 92.34% |  |
| L9 | 86.02% | 82.21% | 95.48% | 94.19% |  |
| L10 | 89.43% | 86.55% | 96.84% | 95.94% | absolute |
| L11 | 84.42% | 80.16% | 97.98% | 97.39% |  |
| L12 | 78.34% | 72.42% | 98.16% | 97.62% | relative |
| L13 | 76.09% | 69.54% | 98.01% | 97.43% |  |
| L14 | 74.22% | 67.26% | 96.91% | 96.11% |  |
| L15 | 74.26% | 67.31% | 96.89% | 96.08% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `4`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 92.70% | 90.27% | 2.262 |
| Random direction | 91.40% | 90.25% | 2.258 |
| Relative-board direction | 95.70% | 94.35% | 0.726 |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_8x8_bf16/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_8x8_bf16/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_8x8_bf16/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_8x8_bf16/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| native_ar | 1156.3 s |
| board_feature_extraction | 177.3 s |
| board_probe_linear | 96.5 s |
| board_probe_mlp | 116.6 s |
| causal_intervention | 106.5 s |
