# JEPA Smoke Evaluation: jepa_vicreg_b8_run_001_hard_disjoint_bf16

- **Checkpoint:** `checkpoint_games_015M.pt`
- **Objective:** `jepa`
- **Checkpoint step:** `59041`
- **Games seen:** `15,099,866`
- **Downstream training chunks:** `20`
- **Board probe training games:** `5,000` (`250` per chunk)
- **Validation split begins after pretraining chunk:** `200`

## Next-Move Heads: Legal-Move Evaluation

The heads are trained with next-move cross-entropy. These metrics evaluate whether their predictions are legal, not whether they match the sampled continuation.

| Head | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal probability mass |
|---|---:|---:|---:|---:|
| Linear | 71.68% | 63.82% | 57.37% | 41.46% |
| MLP | 77.86% | 69.66% | 62.59% | 46.28% |

- **Board probe protocol:** `normalized_v2` (input LayerNorm=True, precision=bf16)

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L7 | 66.55% | L5 | 69.40% |
| MLP | L8 | 66.44% | L5 | 69.02% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 62.72% | 65.00% |
| L1 | 64.41% | 67.56% |
| L2 | 65.51% | 68.71% |
| L3 | 65.81% | 68.91% |
| L4 | 66.29% | 69.34% |
| L5 | 66.44% | 69.40% |
| L6 | 66.46% | 69.33% |
| L7 | 66.55% | 69.31% |
| L8 | 66.50% | 69.12% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 64.59% | 65.14% |
| L1 | 65.03% | 67.73% |
| L2 | 65.76% | 68.52% |
| L3 | 65.87% | 68.65% |
| L4 | 66.25% | 68.99% |
| L5 | 66.35% | 69.02% |
| L6 | 66.34% | 68.95% |
| L7 | 66.37% | 68.95% |
| L8 | 66.44% | 68.80% |

## Matched-Conditions Board Probes (pass-free, last position, t in [4,44])

Probes trained only on hidden states at position t-1 of pass-free prefixes — the exact conditions prefix-based objectives supervise. Compare against the all-position tables above.

| Layer | Linear abs | Linear rel | MLP abs | MLP rel |
|---:|---:|---:|---:|---:|
| L0 | 64.86% | 67.06% | 67.37% | 67.63% |
| L1 | 68.21% | 70.97% | 69.35% | 71.54% |
| L2 | 69.43% | 72.56% | 70.38% | 72.83% |
| L3 | 69.98% | 72.90% | 70.32% | 72.98% |
| L4 | 70.89% | 73.54% | 71.11% | 73.42% |
| L5 | 70.99% | 73.72% | 71.04% | 73.57% |
| L6 | 71.16% | 73.76% | 71.11% | 73.56% |
| L7 | 71.28% | 73.74% | 71.14% | 73.46% |
| L8 | 71.19% | 73.67% | 71.04% | 73.46% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_next_move_head__checkpoint_games_015M.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_next_move_head__checkpoint_games_015M.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_board_probes_normalized_v2__checkpoint_games_015M.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_board_probes_normalized_v2__checkpoint_games_015M.pt`
