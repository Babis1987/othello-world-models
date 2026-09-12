# JEPA Smoke Evaluation: jepa_v4_multi_pos_mlp_k4_b8_run_001_hard_disjoint_vicreg

- **Checkpoint:** `final.pt`
- **Objective:** `jepa`
- **Checkpoint step:** `9800`
- **Games seen:** `4,999,956`
- **Downstream training chunks:** `20`
- **Board probe training games:** `5,000` (`250` per chunk)
- **Validation split begins after pretraining chunk:** `50`

## Next-Move Heads: Legal-Move Evaluation

The heads are trained with next-move cross-entropy. These metrics evaluate whether their predictions are legal, not whether they match the sampled continuation.

| Head | Top-1 legal | Top-3 legal fraction | Top-5 legal fraction | Legal probability mass |
|---|---:|---:|---:|---:|
| Linear | 22.13% | 21.97% | 21.76% | 15.86% |
| MLP | 25.27% | 24.22% | 23.86% | 18.03% |

## Board-State Probes: Best Validation Layers

| Probe | Best absolute layer | Absolute accuracy | Best relative layer | Relative accuracy |
|---|---:|---:|---:|---:|
| Linear | L0 | 54.56% | L0 | 55.05% |
| MLP | L0 | 59.78% | L0 | 60.45% |

## Linear Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 54.56% | 55.05% |
| L1 | 52.22% | 53.41% |
| L2 | 49.57% | 50.79% |
| L3 | 48.42% | 50.01% |
| L4 | 48.22% | 49.36% |
| L5 | 46.36% | 47.24% |
| L6 | 44.38% | 46.24% |
| L7 | 43.77% | 43.41% |
| L8 | 43.22% | 44.93% |

## MLP Board-State Probe: Per-Layer Validation Accuracy

| Layer | Absolute accuracy | Relative accuracy |
|---:|---:|---:|
| L0 | 59.78% | 60.45% |
| L1 | 55.06% | 55.86% |
| L2 | 52.80% | 53.76% |
| L3 | 53.07% | 53.83% |
| L4 | 52.31% | 53.64% |
| L5 | 52.63% | 53.73% |
| L6 | 53.01% | 53.57% |
| L7 | 51.92% | 52.38% |
| L8 | 52.80% | 52.52% |

## Artifacts

- Linear next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_mlp_k4_b8_run_001_hard_disjoint_vicreg/smoke_eval/linear_next_move_head__final.pt`
- MLP next-move head: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_mlp_k4_b8_run_001_hard_disjoint_vicreg/smoke_eval/mlp_next_move_head__final.pt`
- Linear board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_mlp_k4_b8_run_001_hard_disjoint_vicreg/smoke_eval/linear_board_probes__final.pt`
- MLP board probes: `/content/drive/MyDrive/Master_Thesis_Artifacts/runs/jepa_v4_multi_pos_mlp_k4_b8_run_001_hard_disjoint_vicreg/smoke_eval/mlp_board_probes__final.pt`
