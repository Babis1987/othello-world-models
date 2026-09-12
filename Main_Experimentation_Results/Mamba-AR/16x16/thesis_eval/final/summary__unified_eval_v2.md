# Unified evaluation: mamba_ar_b16

- **Protocol:** `unified_eval_v2`
- **Board:** `16x16`
- **Architecture/objective:** `mamba` / `ar`
- **Checkpoint:** `final.pt` (step `78200`, games `20,000,000`)
- **Checkpoint SHA-256:** `86b7b3c19c1f85fb348a36b756849495aa535427abe9834cbe6ddebe7e52f184`
- **Split manifest SHA-256:** `20dc4d4e6c57313038b202fb3f6f762a18beae2fe62617601bf77765f68a124e`
- **Data-manifest SHA-256:** `a5db6b8cde1b971c63b0afcaaea866b5d3c4b6f7a71c843883e63b0e20ee7f57`
- **Per-shard corpus identity:** `aa2936a4105559f5e89723eb7ea4f8530bd68e10129f5c27c199d18f91467f35`
- **Project-source SHA-256:** `f6d85c1fed548dc383e645efa0eed0c313d0efcd33b922e2b62451c603f2d826`
- **Exact pretraining budget:** `20,000,000` games / `200` shards
- **Checkpoint-reported training precision:** `bf16`
- **Precision:** frozen encoder bf16; cached features/readouts/metrics fp32
- **Pass policy:** effective side-to-move; implicit-pass positions are scored
- **Mamba runtime:** `mamba-ssm None` / `causal-conv1d None`

## Frozen-encoder next-move readouts (primary comparison)

| Readout | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---|---:|---:|---:|---:|
| LINEAR | 98.40% | 97.51% | 96.41% | 90.97% |
| MLP | 97.00% | 95.89% | 94.63% | 82.87% |

## Native AR head (separate end-to-end control)

This head was jointly trained with the encoder for the full AR budget; it is not mixed into the frozen-readout comparison above.

| Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal mass |
|---:|---:|---:|---:|
| 99.05% | 98.24% | 97.18% | 90.77% |

## Phase-stratified board-state probes

Layers are selected only on the selection split; every number in the `Test` columns comes from the untouched final test split.

| Probe | Labels | Selected layer | Depth | Val macro | Test macro | Test overall | Occupied | Empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LINEAR | absolute | L11 | 0.733 | 66.23% | 66.31% | 74.40% | 49.50% | 99.94% |
| LINEAR | relative | L11 | 0.733 | 82.72% | 82.59% | 86.77% | 74.04% | 99.84% |
| MLP | absolute | L12 | 0.800 | 64.73% | 64.68% | 72.95% | 47.50% | 99.05% |
| MLP | relative | L11 | 0.733 | 77.86% | 77.63% | 82.56% | 67.60% | 97.90% |

## Artifacts and timing

- Results JSON: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/results__unified_eval_v2.json`
- linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/linear_next_move_head__unified_eval_v2.pt`
- mlp next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/mlp_next_move_head__unified_eval_v2.pt`
- linear board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/linear_board_probe__unified_eval_v2.pt`
- mlp board probe: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/mlp_board_probe__unified_eval_v2.pt`
- Position manifest: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/Mamba-AR/mamba_ar_b16/thesis_eval/final/position_manifest__unified_eval_v2.json`

| Stage | Wall time |
|---|---:|
| native_ar | 638.9 s |
| frozen_head_linear | 1161.6 s |
| frozen_head_mlp | 1132.7 s |
| board_feature_extraction | 925.9 s |
| board_probe_linear | 6.9 s |
| board_probe_mlp | 8.6 s |
