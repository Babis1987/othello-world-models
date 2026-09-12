# Unified evaluation: selected__random_encoder_control

- **Protocol:** `unified_eval_v4`
- **Board:** `8x8`
- **Architecture/objective:** `transformer` / `architecture_matched_random_encoder_control`
- **Checkpoint:** `final.pt` (step `0`, games `0`)
- **Checkpoint SHA-256:** `8ec6b1b75deeed477048e0a21f7125b52aa690a718000926da5df72c4fa7c67d`
- **Split manifest SHA-256:** `4fd0bfa141d51f8ff969526a9b5128a1afb4b9e0afc730f547102bb2ad8f59fd`
- **Data-manifest SHA-256:** `a0994e3e27b032e593894e0fc9f2626094b4b75e91179f4803fd0f23939b1325`
- **Per-shard corpus identity:** `unavailable-counts-only`
- **Project-source SHA-256:** `3de85ba0c54c02ccde18de35609415525e0553c52a3a009de764d5034018cf1d`
- **Exact pretraining budget:** `0` games / `0` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 62.13% | 56.21% | 51.28% | 36.09% |
| MLP | 74.04% | 68.34% | 62.86% | 50.50% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L1 | 0.125 | 63.94% | 63.84% | 70.47% | 48.46% | 94.68% |
| LINEAR | relative | L1 | 0.125 | 65.92% | 65.87% | 72.20% | 51.90% | 94.53% |
| MLP | absolute | L1 | 0.125 | 64.74% | 64.70% | 71.07% | 49.93% | 94.32% |
| MLP | relative | L1 | 0.125 | 65.67% | 65.71% | 72.09% | 51.97% | 94.22% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 62.60% | 55.58% | 64.83% | 58.26% |  |
| L1 | 70.47% | 63.84% | 72.20% | 65.87% | absolute + relative |
| L2 | 70.08% | 63.42% | 71.73% | 65.31% |  |
| L3 | 69.65% | 62.99% | 71.31% | 64.84% |  |
| L4 | 69.37% | 62.69% | 71.03% | 64.51% |  |
| L5 | 69.10% | 62.38% | 70.78% | 64.20% |  |
| L6 | 68.97% | 62.25% | 70.62% | 64.01% |  |
| L7 | 68.74% | 62.00% | 70.47% | 63.79% |  |
| L8 | 68.63% | 61.92% | 70.34% | 63.66% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.65% | 58.13% | 65.16% | 58.51% |  |
| L1 | 71.07% | 64.70% | 72.09% | 65.71% | absolute + relative |
| L2 | 70.38% | 63.89% | 71.65% | 65.21% |  |
| L3 | 69.95% | 63.44% | 71.22% | 64.78% |  |
| L4 | 69.57% | 63.03% | 70.95% | 64.43% |  |
| L5 | 69.40% | 62.92% | 70.72% | 64.28% |  |
| L6 | 69.21% | 62.68% | 70.56% | 64.00% |  |
| L7 | 68.90% | 62.31% | 70.39% | 63.85% |  |
| L8 | 68.82% | 62.25% | 70.30% | 63.71% |  |

## Artifacts and timing

- Results JSON: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/random_encoder_control/results__unified_eval_v4.json`
- linear next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/random_encoder_control/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/random_encoder_control/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/random_encoder_control/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/random_encoder_control/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/seed_evaluation_views/transformer_jepa_b8_seed000/runs/selected/thesis_eval/final/random_encoder_control/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 158.8 s |
| board_probe_linear | 54.4 s |
| board_probe_mlp | 64.6 s |
| frozen_head_linear | 1954.0 s |
| frozen_head_mlp | 2766.9 s |
| head_tuning_linear | 90.9 s |
| head_tuning_mlp | 2.1 s |
