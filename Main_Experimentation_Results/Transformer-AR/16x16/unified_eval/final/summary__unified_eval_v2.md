# Unified evaluation: transformer_ar_b16

- **Protocol:** `unified_eval_v2`
- **Board:** `16x16`
- **Architecture/objective:** `transformer` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `39ba5994de998b615353189561ee7f104209a48c2e27ae654a2fb1c47beb0623`
- **Split manifest SHA-256:** `20dc4d4e6c57313038b202fb3f6f762a18beae2fe62617601bf77765f68a124e`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `daf598da3c0cc3a0948ef1464fbc870b795d8a2fd7304fabd4276f682d5cd3d7`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored

## Frozen-encoder next-move readouts (primary comparison)

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 98.24% | 97.30% | 96.16% | 89.39% |
| MLP | 96.78% | 95.52% | 94.12% | 80.30% |

## Native AR head (separate end-to-end control)

This head was jointly trained with the encoder for the full AR budget; it is not mixed into the frozen-readout comparison above.

| Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---:|---:|---:|---:|
| 98.97% | 98.11% | 97.01% | 88.82% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L5 | 0.625 | 66.44% | 66.52% | 74.54% | 49.84% | 99.87% |
| LINEAR | relative | L7 | 0.875 | 81.85% | 81.80% | 86.19% | 72.79% | 99.94% |
| MLP | absolute | L6 | 0.750 | 64.58% | 64.55% | 72.75% | 47.51% | 98.62% |
| MLP | relative | L8 | 1.000 | 75.97% | 75.89% | 81.32% | 64.75% | 98.31% |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/unified_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/unified_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/unified_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/unified_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/unified_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Transformer-AR/transformer_ar_b16/unified_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| native_ar | 635.4 s |
| frozen_head_linear | 1201.1 s |
| frozen_head_mlp | 1190.1 s |
| board_feature_extraction | 918.9 s |
| board_probe_linear | 3.8 s |
| board_probe_mlp | 5.3 s |
