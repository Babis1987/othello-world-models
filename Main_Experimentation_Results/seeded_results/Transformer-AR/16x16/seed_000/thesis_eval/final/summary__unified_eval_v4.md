# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `1fab25e92011b84d29d64657b993ed04653436011cfa8a6870a31c8eec1aa4f9`
- **Split manifest SHA-256:** `fd0ba1f8526c19877cec8932bcad7949ead980cd037ef6c34fa5edbdf2cd1605`
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
| Native AR | 99.01% | 98.15% | 97.04% | 89.08% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 66.79% | 66.75% | 74.74% | 50.16% | 99.92% |
| LINEAR | relative | L8 | 1.000 | 82.97% | 83.05% | 87.16% | 74.65% | 99.97% |
| MLP | absolute | L5 | 0.625 | 66.76% | 66.75% | 74.68% | 50.27% | 99.68% |
| MLP | relative | L7 | 0.875 | 80.52% | 80.59% | 85.24% | 71.15% | 99.68% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.68% | 56.62% | 65.55% | 57.78% |  |
| L1 | 66.68% | 58.69% | 67.89% | 60.26% |  |
| L2 | 72.70% | 64.62% | 74.43% | 66.87% |  |
| L3 | 74.37% | 66.33% | 76.20% | 68.69% |  |
| L4 | 74.62% | 66.60% | 82.21% | 76.55% |  |
| L5 | 74.74% | 66.75% | 86.13% | 81.71% | absolute |
| L6 | 74.71% | 66.69% | 86.93% | 82.75% |  |
| L7 | 74.61% | 66.56% | 87.12% | 83.00% |  |
| L8 | 74.59% | 66.54% | 87.16% | 83.05% | relative |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.54% | 57.53% | 65.81% | 57.87% |  |
| L1 | 66.68% | 58.65% | 67.20% | 59.28% |  |
| L2 | 71.18% | 63.12% | 72.38% | 64.63% |  |
| L3 | 73.28% | 65.20% | 74.58% | 66.81% |  |
| L4 | 74.08% | 66.03% | 79.51% | 73.12% |  |
| L5 | 74.68% | 66.75% | 84.13% | 79.16% | absolute |
| L6 | 74.53% | 66.51% | 85.18% | 80.52% |  |
| L7 | 74.26% | 66.14% | 85.24% | 80.59% | relative |
| L8 | 74.30% | 66.19% | 85.23% | 80.57% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b16_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b16_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1658.2 s |
| board_probe_linear | 104.6 s |
| board_probe_mlp | 114.8 s |
| native_ar | 16890.4 s |
