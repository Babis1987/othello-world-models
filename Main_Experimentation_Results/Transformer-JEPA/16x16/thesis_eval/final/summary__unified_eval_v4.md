# Unified evaluation: transformer_jepa_v5_hd_infonce_allpos_b16

- **Protocol:** `unified_eval_v4`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `jepa_v5_infonce_hard_disjoint_all_position`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `b11199e3483a8a0a2da0fb225d0a7aedcb33a67fd968356542c70a78e812a571`
- **Split manifest SHA-256:** `20dc4d4e6c57313038b202fb3f6f762a18beae2fe62617601bf77765f68a124e`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `9e5740023d9dcdeff04fd8aae7f60bd51382e5f6062cd0c6ec2941a8c2de6225`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 97.54% | 96.49% | 95.30% | 87.93% |
| MLP | 97.14% | 96.09% | 94.91% | 87.61% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L6 | 0.750 | 66.30% | 66.26% | 74.36% | 49.45% | 99.89% |
| LINEAR | relative | L7 | 0.875 | 78.28% | 78.37% | 83.49% | 67.91% | 99.46% |
| MLP | absolute | L7 | 0.875 | 65.74% | 65.70% | 73.82% | 48.84% | 99.43% |
| MLP | relative | L7 | 0.875 | 75.90% | 75.99% | 81.56% | 64.61% | 98.92% |

## Board-state probe accuracy by layer

Each layer table reports untouched test-split accuracy. `Selected` marks validation-selected layers.

### LINEAR board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 64.56% | 56.49% | 65.46% | 57.68% |  |
| L1 | 65.66% | 57.59% | 66.96% | 59.30% |  |
| L2 | 68.26% | 60.15% | 69.58% | 61.87% |  |
| L3 | 70.67% | 62.58% | 72.18% | 64.55% |  |
| L4 | 72.38% | 64.26% | 74.95% | 67.63% |  |
| L5 | 73.94% | 65.82% | 79.90% | 73.66% |  |
| L6 | 74.36% | 66.26% | 83.36% | 78.11% | absolute |
| L7 | 74.06% | 65.94% | 83.49% | 78.37% | relative |
| L8 | 73.61% | 65.50% | 82.47% | 77.21% |  |

### MLP board-state probe

| Layer | Absolute accuracy | Absolute macro | Relative accuracy | Relative macro | Selected |
|---:|---:|---:|---:|---:|:---|
| L0 | 65.50% | 57.49% | 65.78% | 57.81% |  |
| L1 | 66.11% | 58.04% | 66.61% | 58.66% |  |
| L2 | 67.55% | 59.48% | 68.26% | 60.34% |  |
| L3 | 69.36% | 61.38% | 70.30% | 62.55% |  |
| L4 | 70.96% | 62.95% | 72.61% | 65.08% |  |
| L5 | 72.85% | 64.73% | 77.39% | 70.75% |  |
| L6 | 73.85% | 65.72% | 81.34% | 75.67% |  |
| L7 | 73.82% | 65.70% | 81.56% | 75.99% | absolute + relative |
| L8 | 73.08% | 64.90% | 80.21% | 74.41% |  |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/results__unified_eval_v4.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/linear_next_move_head__unified_eval_v4.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/mlp_next_move_head__unified_eval_v4.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/linear_board_probe__unified_eval_v4.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/mlp_board_probe__unified_eval_v4.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-JEPA/transformer_jepa_v5_hd_infonce_allpos_b16/thesis_eval/final/position_manifest__unified_eval_v4.json`

| Stage | Wall time |
|---|---:|
| board_feature_extraction | 1651.6 s |
| board_probe_linear | 102.4 s |
| board_probe_mlp | 116.2 s |
| frozen_head_linear | 20805.1 s |
| frozen_head_mlp | 19397.6 s |
| head_tuning_linear | 1591.9 s |
| head_tuning_mlp | 10.6 s |
