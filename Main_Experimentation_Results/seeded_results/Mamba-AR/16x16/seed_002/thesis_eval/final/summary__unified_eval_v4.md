# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `c9b9b03fe7d38540fad933324c3f27309c5d1d14c50069bdff11eb602e2ba1cc`
- **Split manifest SHA-256:** `f6ecf88a63852c48c22f6d4740ee6214f09e5afbd83e92b8352ffe84fb22fb0f`
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
| Native AR | 98.96% | 98.13% | 97.06% | 90.52% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L11 | 0.733 | 66.60% | 66.60% | 74.60% | 49.98% | 99.82% |
| LINEAR | relative | L12 | 0.800 | 83.26% | 83.26% | 87.29% | 74.99% | 99.90% |
| MLP | absolute | L12 | 0.800 | 66.20% | 66.13% | 74.23% | 49.29% | 99.79% |
| MLP | relative | L12 | 0.800 | 81.17% | 81.17% | 85.63% | 72.10% | 99.49% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.76% | 56.61% | 65.65% | 57.79% |  |
| L1 | 71.12% | 63.03% | 72.50% | 64.80% |  |
| L2 | 69.22% | 61.14% | 75.69% | 69.78% |  |
| L3 | 68.47% | 60.42% | 79.69% | 75.24% |  |
| L4 | 69.02% | 60.98% | 79.99% | 75.52% |  |
| L5 | 70.03% | 61.98% | 81.06% | 76.60% |  |
| L6 | 70.49% | 62.43% | 81.70% | 77.29% |  |
| L7 | 71.80% | 63.77% | 83.32% | 79.02% |  |
| L8 | 72.33% | 64.31% | 84.14% | 79.92% |  |
| L9 | 72.94% | 64.92% | 84.96% | 80.79% |  |
| L10 | 73.97% | 65.96% | 86.10% | 81.95% |  |
| L11 | 74.60% | 66.60% | 86.82% | 82.68% | absolute |
| L12 | 74.61% | 66.58% | 87.29% | 83.26% | relative |
| L13 | 74.60% | 66.56% | 87.19% | 83.11% |  |
| L14 | 74.50% | 66.43% | 87.03% | 82.91% |  |
| L15 | 74.48% | 66.40% | 86.99% | 82.85% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.24% | 57.10% | 65.82% | 57.88% |  |
| L1 | 69.41% | 61.36% | 70.51% | 62.69% |  |
| L2 | 68.48% | 60.61% | 73.19% | 67.00% |  |
| L3 | 67.61% | 59.62% | 77.15% | 72.47% |  |
| L4 | 67.93% | 59.94% | 77.42% | 72.71% |  |
| L5 | 68.86% | 60.83% | 78.44% | 73.76% |  |
| L6 | 69.33% | 61.29% | 78.99% | 74.32% |  |
| L7 | 70.68% | 62.66% | 80.69% | 76.08% |  |
| L8 | 71.29% | 63.25% | 81.61% | 77.08% |  |
| L9 | 72.04% | 64.00% | 82.62% | 78.15% |  |
| L10 | 73.39% | 65.35% | 84.21% | 79.71% |  |
| L11 | 74.21% | 66.17% | 85.16% | 80.62% |  |
| L12 | 74.23% | 66.13% | 85.63% | 81.17% | absolute + relative |
| L13 | 74.24% | 66.11% | 85.40% | 80.80% |  |
| L14 | 74.10% | 65.93% | 85.13% | 80.45% |  |
| L15 | 74.09% | 65.91% | 85.14% | 80.47% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/mamba_ar_b16_seed002/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed002/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/mamba_ar_b16_seed002/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/mamba_ar_b16_seed002/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1658.8 s |
| board_probe_linear | 179.4 s |
| board_probe_mlp | 200.8 s |
| native_ar | 16828.3 s |
