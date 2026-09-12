# Unified evaluation: transformer_ar_b16

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `39ba5994de998b615353189561ee7f104209a48c2e27ae654a2fb1c47beb0623`
- **Split manifest SHA-256:** `20dc4d4e6c57313038b202fb3f6f762a18beae2fe62617601bf77765f68a124e`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 98.99% | 98.12% | 97.01% | 88.78% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 66.82% | 66.81% | 74.79% | 50.23% | 99.96% |
| LINEAR | relative | L8 | 1.000 | 82.87% | 82.88% | 87.02% | 74.39% | 99.97% |
| MLP | absolute | L7 | 0.875 | 66.97% | 67.03% | 74.94% | 50.60% | 99.89% |
| MLP | relative | L7 | 0.875 | 80.50% | 80.49% | 85.16% | 70.98% | 99.70% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.69% | 56.61% | 65.59% | 57.81% |  |
| L1 | 66.89% | 58.86% | 68.04% | 60.34% |  |
| L2 | 72.57% | 64.51% | 74.09% | 66.47% |  |
| L3 | 74.46% | 66.46% | 76.05% | 68.50% |  |
| L4 | 74.66% | 66.66% | 81.13% | 75.13% |  |
| L5 | 74.79% | 66.81% | 85.66% | 81.08% | absolute |
| L6 | 74.72% | 66.71% | 86.62% | 82.34% |  |
| L7 | 74.62% | 66.58% | 87.00% | 82.84% |  |
| L8 | 74.59% | 66.54% | 87.02% | 82.88% | relative |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.52% | 57.51% | 65.84% | 57.90% |  |
| L1 | 66.75% | 58.73% | 67.27% | 59.33% |  |
| L2 | 71.15% | 63.09% | 72.17% | 64.37% |  |
| L3 | 73.43% | 65.38% | 74.47% | 66.64% |  |
| L4 | 74.09% | 66.03% | 78.34% | 71.55% |  |
| L5 | 74.64% | 66.68% | 83.59% | 78.41% |  |
| L6 | 74.35% | 66.26% | 84.78% | 79.96% |  |
| L7 | 74.94% | 67.03% | 85.16% | 80.49% | absolute + relative |
| L8 | 74.59% | 66.57% | 85.12% | 80.43% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1665.3 s |
| board_probe_linear | 101.3 s |
| board_probe_mlp | 112.4 s |
| native_ar | 16758.0 s |
