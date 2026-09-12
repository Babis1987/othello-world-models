# JEPA Smoke Evaluation: jepa_transformer_b8_run_001_hard_disjoint

- **Checkpoint:** `final.pt`
- **Objective:** `jepa`
- **Checkpoint step:** `9800`
- **Games seen:** `4,999,952`
- **Downstream training chunks:** `20`
- **Board probe training games:** `5,000` (`250` per chunk)
- **Validation split begins after pretraining chunk:** `50`

## Next-Move Heads: Legal-Move Evaluation

The heads are trained with next-move cross-entropy. These metrics evaluate whether their predictions are legal, not whether they match the sampled continuation.

| Head | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal probability mass |
|---|---:|---:|---:|---:|
| Linear | 31.22% | 28.82% | 27.13% | 18.91% |
| MLP | 37.84% | 34.48% | 31.80% | 22.58% |

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L0 | 56.79% | L0 | 58.85% |
| MLP | L0 | 60.62% | L0 | 62.01% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 56.79% | 58.85% |
| L1 | 49.02% | 50.85% |
| L2 | 50.13% | 51.06% |
| L3 | 50.29% | 49.52% |
| L4 | 48.98% | 49.66% |
| L5 | 48.50% | 49.29% |
| L6 | 50.85% | 51.27% |
| L7 | 50.98% | 50.18% |
| L8 | 48.96% | 51.45% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 60.62% | 62.01% |
| L1 | 54.67% | 55.31% |
| L2 | 55.88% | 57.09% |
| L3 | 56.34% | 56.40% |
| L4 | 55.83% | 55.64% |
| L5 | 55.79% | 55.62% |
| L6 | 56.49% | 56.36% |
| L7 | 56.40% | 57.07% |
| L8 | 56.94% | 57.40% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_transformer_b8_run_001_hard_disjoint/smoke_eval/linear_next_move_head__final.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_transformer_b8_run_001_hard_disjoint/smoke_eval/mlp_next_move_head__final.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_transformer_b8_run_001_hard_disjoint/smoke_eval/linear_board_probes__final.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_transformer_b8_run_001_hard_disjoint/smoke_eval/mlp_board_probes__final.pt`
