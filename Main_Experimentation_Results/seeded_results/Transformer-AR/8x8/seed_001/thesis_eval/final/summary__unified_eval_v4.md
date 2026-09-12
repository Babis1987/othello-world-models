# Unified evaluation: selected

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `19,999,840`)
- **Checkpoint SHA-256:** `ba14a88df9f37c2f78cc2ba24b26b70b74081f2722fc71708fa42cc25481c930`
- **Split manifest SHA-256:** `ebe05316ed531b94bc2e33f685ae79d118b96870025ebd9a10028d58e573a192`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `3de85ba0c54c02ccde18de35609415525e0553c52a3a009de764d5034018cf1d`
- **Exact pretraining budget:** `19,999,840` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Native AR next-move prediction

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| Native AR | 99.72% | 96.77% | 92.43% | 98.35% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L4 | 0.500 | 68.94% | 68.90% | 75.59% | 53.40% | 100.00% |
| LINEAR | relative | L6 | 0.750 | 96.72% | 96.94% | 97.61% | 95.45% | 99.99% |
| MLP | absolute | L5 | 0.625 | 94.32% | 94.43% | 95.63% | 91.65% | 100.00% |
| MLP | relative | L6 | 0.750 | 96.35% | 96.58% | 97.35% | 94.97% | 99.98% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.71% | 55.82% | 64.81% | 58.21% |  |
| L1 | 75.15% | 68.43% | 87.30% | 83.78% |  |
| L2 | 75.62% | 68.94% | 92.31% | 90.11% |  |
| L3 | 75.64% | 68.96% | 94.29% | 92.68% |  |
| L4 | 75.59% | 68.90% | 96.02% | 94.90% | absolute |
| L5 | 75.57% | 68.87% | 97.25% | 96.48% |  |
| L6 | 75.40% | 68.66% | 97.61% | 96.94% | relative |
| L7 | 75.20% | 68.43% | 97.55% | 96.88% |  |
| L8 | 75.13% | 68.34% | 97.51% | 96.82% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.12% | 58.78% | 65.23% | 58.59% |  |
| L1 | 84.01% | 79.80% | 86.59% | 82.91% |  |
| L2 | 89.64% | 86.81% | 92.06% | 89.79% |  |
| L3 | 91.84% | 89.60% | 94.08% | 92.41% |  |
| L4 | 94.07% | 92.46% | 95.88% | 94.71% |  |
| L5 | 95.63% | 94.43% | 97.13% | 96.29% | absolute |
| L6 | 93.16% | 91.31% | 97.35% | 96.58% | relative |
| L7 | 88.35% | 85.19% | 97.26% | 96.50% |  |
| L8 | 87.91% | 84.65% | 97.17% | 96.38% |  |

## Nanda-style causal intervention (8x8)

- **Readout:** `native_ar`
- **Selected alpha:** `2`
- **Interpretation scope:** native model behavior

| Control | Target top-1 legal | Target legal mass | Top-N FP+FN |
|---|---:|---:|---:|
| Null | 93.10% | 89.48% | 2.242 |
| Random direction | 90.40% | 89.34% | 2.288 |
| Relative-board direction | 97.90% | 95.32% | 0.480 |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_ar_b8_seed001/runs/selected/thesis_eval/final/results__unified_eval_v4.json`
- linear board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed001/runs/selected/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_ar_b8_seed001/runs/selected/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_ar_b8_seed001/runs/selected/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 161.0 s |
| board_probe_linear | 55.6 s |
| board_probe_mlp | 64.1 s |
| causal_intervention | 62.5 s |
| native_ar | 1091.8 s |
