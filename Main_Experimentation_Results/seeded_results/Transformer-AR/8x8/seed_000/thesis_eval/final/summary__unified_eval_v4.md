# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `c38baa5b2c844b04ae3e53da49d3540136c531f1f0725f4659fca50224118770`
- **Split manifest SHA-256:** `a31db3fcea98fe257c40e6c83efdedcd707f96a0b36218fcecd525c892b061dc`
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
| Native AR | 99.71% | 96.77% | 92.44% | 98.42% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L4 | 0.500 | 68.95% | 68.94% | 75.62% | 53.45% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.66% | 96.88% | 97.57% | 95.37% | 99.99% |
| MLP | absolute | L5 | 0.625 | 93.56% | 93.74% | 95.08% | 90.61% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.28% | 96.55% | 97.33% | 94.92% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.72% | 55.81% | 64.77% | 58.16% |  |
| L1 | 75.05% | 68.33% | 87.23% | 83.70% |  |
| L2 | 75.58% | 68.90% | 92.23% | 90.03% |  |
| L3 | 75.58% | 68.89% | 94.26% | 92.63% |  |
| L4 | 75.62% | 68.94% | 95.98% | 94.85% | absolute |
| L5 | 75.62% | 68.94% | 97.13% | 96.32% |  |
| L6 | 75.45% | 68.73% | 97.57% | 96.88% | relative |
| L7 | 75.19% | 68.42% | 97.46% | 96.76% |  |
| L8 | 75.14% | 68.36% | 97.40% | 96.68% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.12% | 58.79% | 65.27% | 58.64% |  |
| L1 | 83.99% | 79.79% | 86.53% | 82.87% |  |
| L2 | 89.46% | 86.58% | 92.01% | 89.72% |  |
| L3 | 91.35% | 88.99% | 94.05% | 92.37% |  |
| L4 | 93.41% | 91.61% | 95.85% | 94.67% |  |
| L5 | 95.08% | 93.74% | 96.99% | 96.12% | absolute |
| L6 | 93.36% | 91.56% | 97.33% | 96.55% | relative |
| L7 | 87.12% | 83.63% | 97.14% | 96.34% |  |
| L8 | 85.37% | 81.41% | 97.05% | 96.24% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `2`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 91.30% | 89.59% | 2.254 |
| Random direction | 90.80% | 89.51% | 2.288 |
| Relative-board direction | 98.50% | 95.61% | 0.480 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b8_seed000/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed000/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed000/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b8_seed000/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 164.7 s |
| board_probe_linear | 59.8 s |
| board_probe_mlp | 64.7 s |
| causal_intervention | 62.5 s |
| native_ar | 1106.3 s |
