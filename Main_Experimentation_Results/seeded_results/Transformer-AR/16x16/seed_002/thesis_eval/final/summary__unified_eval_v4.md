# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `2839b9b76efea48101943b3f7a8984b550ecdab9ed2097abde38421f6b46bd88`
- **Split manifest SHA-256:** `0393a24fbf8655140ce9672c9bd3ca3e917b1afdc76d71030e0a03dea8f59a3c`
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
| Native AR | 99.00% | 98.14% | 97.04% | 88.94% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 66.75% | 66.75% | 74.73% | 50.17% | 99.91% |
| LINEAR | relative | L8 | 1.000 | 83.06% | 83.01% | 87.12% | 74.58% | 99.97% |
| MLP | absolute | L5 | 0.625 | 66.69% | 66.67% | 74.62% | 50.17% | 99.67% |
| MLP | relative | L8 | 1.000 | 80.61% | 80.57% | 85.22% | 71.08% | 99.72% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.65% | 56.59% | 65.54% | 57.77% |  |
| L1 | 66.83% | 58.77% | 67.91% | 60.17% |  |
| L2 | 72.84% | 64.78% | 74.46% | 66.87% |  |
| L3 | 74.47% | 66.45% | 76.08% | 68.53% |  |
| L4 | 74.69% | 66.69% | 82.16% | 76.49% |  |
| L5 | 74.73% | 66.75% | 86.02% | 81.57% | absolute |
| L6 | 74.70% | 66.69% | 86.82% | 82.61% |  |
| L7 | 74.60% | 66.56% | 87.07% | 82.93% |  |
| L8 | 74.58% | 66.53% | 87.12% | 83.01% | relative |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.56% | 57.55% | 65.82% | 57.87% |  |
| L1 | 66.77% | 58.78% | 67.19% | 59.25% |  |
| L2 | 71.42% | 63.36% | 72.52% | 64.74% |  |
| L3 | 73.43% | 65.38% | 74.55% | 66.73% |  |
| L4 | 74.12% | 66.07% | 79.41% | 72.96% |  |
| L5 | 74.62% | 66.67% | 84.02% | 79.03% | absolute |
| L6 | 74.47% | 66.43% | 84.99% | 80.27% |  |
| L7 | 74.52% | 66.48% | 85.20% | 80.54% |  |
| L8 | 74.24% | 66.10% | 85.22% | 80.57% | relative |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b16_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b16_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1664.5 s |
| board_probe_linear | 105.6 s |
| board_probe_mlp | 115.6 s |
| native_ar | 16818.4 s |
