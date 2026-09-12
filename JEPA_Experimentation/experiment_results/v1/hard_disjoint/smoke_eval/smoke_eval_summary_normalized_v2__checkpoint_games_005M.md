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
| Linear | 61.92% | 54.61% | 49.31% | 34.97% |
| MLP | 69.27% | 60.81% | 54.30% | 38.84% |

- **Board probe protocol:** `normalized_v2` (input LayerNorm=True, precision=bf16)

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L7 | 65.34% | L7 | 67.18% |
| MLP | L7 | 65.35% | L5 | 67.12% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 62.81% | 65.04% |
| L1 | 63.43% | 65.46% |
| L2 | 64.41% | 66.38% |
| L3 | 64.48% | 66.50% |
| L4 | 64.96% | 66.84% |
| L5 | 65.20% | 67.09% |
| L6 | 65.29% | 67.15% |
| L7 | 65.34% | 67.18% |
| L8 | 65.28% | 67.11% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 64.05% | 65.14% |
| L1 | 63.99% | 65.87% |
| L2 | 64.67% | 66.68% |
| L3 | 64.77% | 66.66% |
| L4 | 65.08% | 66.95% |
| L5 | 65.34% | 67.12% |
| L6 | 65.33% | 67.07% |
| L7 | 65.35% | 67.05% |
| L8 | 65.33% | 67.04% |

## Matched-Conditions Board Probes (pass-free, last position, t in [4,44])

Probes trained only on hidden states at position t-1 of pass-free prefixes — the exact conditions prefix-based objectives supervise. Compare against the all-position tables above.

| Layer | Linear abs | Linear rel | MLP abs | MLP rel |
|---:|---:|---:|---:|---:|
| L0 | 64.91% | 67.01% | 67.00% | 67.59% |
| L1 | 66.80% | 68.14% | 67.45% | 68.91% |
| L2 | 67.84% | 69.59% | 68.48% | 70.12% |
| L3 | 68.05% | 69.62% | 68.42% | 70.05% |
| L4 | 68.84% | 70.23% | 69.19% | 70.59% |
| L5 | 69.07% | 70.80% | 69.46% | 71.03% |
| L6 | 69.41% | 70.66% | 69.54% | 70.93% |
| L7 | 69.32% | 70.87% | 69.65% | 71.01% |
| L8 | 69.66% | 70.60% | 69.61% | 70.87% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_next_move_head__checkpoint_games_005M.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_next_move_head__checkpoint_games_005M.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/linear_board_probes_normalized_v2__checkpoint_games_005M.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_vicreg_b8_run_001_hard_disjoint_bf16/smoke_eval/mlp_board_probes_normalized_v2__checkpoint_games_005M.pt`
