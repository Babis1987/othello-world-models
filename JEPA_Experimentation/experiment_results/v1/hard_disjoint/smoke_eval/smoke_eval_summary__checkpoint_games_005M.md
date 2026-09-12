# JEPA Smoke Evaluation: jepa_vicreg_b8_run_001_hard_disjoint_bf16

- **Checkpoint:** `checkpoint_games_005M.pt`
- **Objective:** `jepa`
- **Checkpoint step:** `19941`
- **Games seen:** `5,099,956`
- **Downstream training chunks:** `20`
- **Board probe training games:** `5,000` (`250` per chunk)
- **Validation split begins after pretraining chunk:** `200`

## Next-Move Heads: Legal-Move Evaluation

The heads are trained with next-move cross-entropy. These metrics evaluate whether their predictions are legal, not whether they match the sampled continuation.

| Head | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal probability mass |
|---|---:|---:|---:|---:|
| Linear | 62.15% | 54.75% | 49.37% | 35.01% |
| MLP | 68.95% | 60.72% | 54.27% | 39.03% |

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L5 | 61.23% | L8 | 62.29% |
| MLP | L2 | 63.11% | L1 | 64.53% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 59.36% | 61.95% |
| L1 | 60.31% | 62.27% |
| L2 | 60.80% | 62.18% |
| L3 | 59.97% | 61.08% |
| L4 | 60.73% | 61.65% |
| L5 | 61.23% | 61.51% |
| L6 | 60.19% | 61.82% |
| L7 | 60.90% | 61.66% |
| L8 | 60.83% | 62.29% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 61.05% | 62.93% |
| L1 | 63.00% | 64.53% |
| L2 | 63.11% | 64.22% |
| L3 | 62.65% | 64.16% |
| L4 | 62.17% | 63.41% |
| L5 | 61.53% | 63.00% |
| L6 | 61.47% | 61.82% |
| L7 | 59.04% | 60.38% |
| L8 | 59.01% | 57.62% |

## Matched-Conditions Board Probes (pass-free, last position, t in [4,44])

Probes trained only on hidden states at position t-1 of pass-free prefixes — the exact conditions prefix-based objectives supervise. Compare against the all-position tables above.

| Layer | Linear abs | Linear rel | MLP abs | MLP rel |
|---:|---:|---:|---:|---:|
| L0 | 64.50% | 66.87% | 64.45% | 66.75% |
| L1 | 62.16% | 62.95% | 66.56% | 67.83% |
| L2 | 63.10% | 64.32% | 66.91% | 68.08% |
| L3 | 63.06% | 64.33% | 66.53% | 67.83% |
| L4 | 63.64% | 64.53% | 66.51% | 67.60% |
| L5 | 64.02% | 63.41% | 66.10% | 67.42% |
| L6 | 63.91% | 63.35% | 65.71% | 66.42% |
| L7 | 64.49% | 63.68% | 63.28% | 64.93% |
| L8 | 63.73% | 64.90% | 62.33% | 63.55% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_next_move_head__checkpoint_games_005M.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_next_move_head__checkpoint_games_005M.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_board_probes__checkpoint_games_005M.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_board_probes__checkpoint_games_005M.pt`
