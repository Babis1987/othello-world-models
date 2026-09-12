# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `e945a9010c1efb9ab251f997c9502a223fac094630418b005dab44368747c302`
- **Split manifest SHA-256:** `2dff6224f7e40802c7589d4dfd27a7fcabe6d51b795a25086b4af4e53c8c7f90`
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
| Native AR | 98.99% | 98.12% | 97.02% | 88.91% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 66.74% | 66.70% | 74.70% | 50.10% | 99.91% |
| LINEAR | relative | L8 | 1.000 | 82.95% | 83.00% | 87.12% | 74.57% | 99.97% |
| MLP | absolute | L6 | 0.750 | 66.65% | 66.60% | 74.60% | 49.97% | 99.84% |
| MLP | relative | L8 | 1.000 | 80.56% | 80.57% | 85.23% | 71.08% | 99.74% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.69% | 56.62% | 65.56% | 57.79% |  |
| L1 | 66.92% | 58.89% | 68.06% | 60.35% |  |
| L2 | 72.76% | 64.68% | 74.49% | 66.93% |  |
| L3 | 74.42% | 66.40% | 76.47% | 69.03% |  |
| L4 | 74.60% | 66.58% | 81.81% | 76.03% |  |
| L5 | 74.70% | 66.70% | 85.99% | 81.54% | absolute |
| L6 | 74.67% | 66.65% | 86.81% | 82.60% |  |
| L7 | 74.57% | 66.52% | 87.06% | 82.92% |  |
| L8 | 74.56% | 66.50% | 87.12% | 83.00% | relative |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.54% | 57.54% | 65.81% | 57.86% |  |
| L1 | 66.80% | 58.80% | 67.22% | 59.26% |  |
| L2 | 71.18% | 63.11% | 72.40% | 64.64% |  |
| L3 | 73.50% | 65.43% | 74.89% | 67.16% |  |
| L4 | 74.03% | 65.98% | 79.07% | 72.52% |  |
| L5 | 74.40% | 66.40% | 83.93% | 78.90% |  |
| L6 | 74.60% | 66.60% | 84.98% | 80.26% | absolute |
| L7 | 74.51% | 66.46% | 85.20% | 80.53% |  |
| L8 | 74.54% | 66.50% | 85.23% | 80.57% | relative |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b16_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b16_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b16_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1653.3 s |
| board_probe_linear | 100.3 s |
| board_probe_mlp | 110.6 s |
| native_ar | 17063.4 s |
