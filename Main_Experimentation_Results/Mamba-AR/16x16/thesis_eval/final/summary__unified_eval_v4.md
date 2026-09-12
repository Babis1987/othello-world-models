# Unified evaluation: mamba_ar_b16

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `86b7b3c19c1f85fb348a36b756849495aa535427abe9834cbe6ddebe7e52f184`
- **Split manifest SHA-256:** `20dc4d4e6c57313038b202fb3f6f762a18beae2fe62617601bf77765f68a124e`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.05% | 98.24% | 97.17% | 90.73% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L11 | 0.733 | 66.67% | 66.58% | 74.62% | 49.88% | 99.98% |
| LINEAR | relative | L11 | 0.733 | 83.41% | 83.35% | 87.37% | 75.10% | 99.95% |
| MLP | absolute | L11 | 0.733 | 66.27% | 66.26% | 74.33% | 49.49% | 99.79% |
| MLP | relative | L11 | 0.733 | 81.46% | 81.41% | 85.80% | 72.49% | 99.43% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.75% | 56.63% | 65.67% | 57.84% |  |
| L1 | 70.24% | 62.12% | 71.37% | 63.56% |  |
| L2 | 69.65% | 61.63% | 72.98% | 66.04% |  |
| L3 | 67.96% | 59.89% | 80.04% | 75.71% |  |
| L4 | 68.76% | 60.70% | 80.41% | 76.04% |  |
| L5 | 69.69% | 61.64% | 81.24% | 76.89% |  |
| L6 | 70.20% | 62.15% | 81.87% | 77.57% |  |
| L7 | 70.81% | 62.73% | 82.77% | 78.56% |  |
| L8 | 71.58% | 63.52% | 83.82% | 79.71% |  |
| L9 | 73.20% | 65.15% | 85.60% | 81.53% |  |
| L10 | 73.80% | 65.76% | 86.44% | 82.41% |  |
| L11 | 74.62% | 66.58% | 87.37% | 83.35% | absolute + relative |
| L12 | 74.59% | 66.53% | 87.37% | 83.33% |  |
| L13 | 74.53% | 66.45% | 87.22% | 83.14% |  |
| L14 | 74.44% | 66.35% | 87.11% | 82.99% |  |
| L15 | 74.42% | 66.32% | 87.06% | 82.93% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.23% | 57.07% | 65.83% | 57.90% |  |
| L1 | 68.65% | 60.62% | 69.53% | 61.67% |  |
| L2 | 68.67% | 60.73% | 71.16% | 63.99% |  |
| L3 | 67.12% | 59.04% | 77.62% | 73.05% |  |
| L4 | 67.78% | 59.72% | 77.90% | 73.30% |  |
| L5 | 68.61% | 60.53% | 78.61% | 73.99% |  |
| L6 | 69.03% | 60.95% | 79.12% | 74.56% |  |
| L7 | 69.54% | 61.45% | 79.96% | 75.48% |  |
| L8 | 70.35% | 62.30% | 81.00% | 76.59% |  |
| L9 | 72.12% | 64.06% | 83.01% | 78.63% |  |
| L10 | 73.06% | 64.98% | 84.27% | 79.89% |  |
| L11 | 74.33% | 66.26% | 85.80% | 81.41% | absolute + relative |
| L12 | 74.33% | 66.23% | 85.82% | 81.37% |  |
| L13 | 74.18% | 66.02% | 85.51% | 80.94% |  |
| L14 | 74.09% | 65.91% | 85.30% | 80.67% |  |
| L15 | 74.10% | 65.92% | 85.29% | 80.66% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1672.5 s |
| board_probe_linear | 183.5 s |
| board_probe_mlp | 205.1 s |
| native_ar | 16718.4 s |
