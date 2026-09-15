# Causal-intervention comparison

- Condition: `transformer` / `jepa` / `16x16`
- Profile: `full`
- Claim scope: `objective_board_extension`
- Shared counterfactuals: occupied-square colour flips that change the legal-move set.
- JEPA readout: `linear`; policy `best`, chosen only from the selection split by `best_selection_legal_probability_mass` (linear=0.878847, mlp=0.875718).

| Method | Published operation | Local intervention | Top-N FP+FN | Target legal mass |
|---|---|---|---:|---:|
| li | Adam optimization through nonlinear absolute probe | `li_nonlinear_activation_optimization_v1` | 10.246 | 0.8689 |
| nanda | fixed normalized linear direction in residual branch | `nanda_target_class_attention_branch_v1` | 5.336 | 0.8651 |
| adapted | target-minus-source edits after every block | `adapted_linear_sequential_residual_v2` | 5.124 | 0.8718 |

The primary adapted-method random control applies one random signed permutation per case at every layer. It therefore preserves the norms and all cross-layer angles of the probe directions while breaking their alignment with the learned feature coordinates.

| Adapted comparison | Delta top-1 (pp) | Delta target mass (pp) | Delta Top-N FP+FN |
|---|---:|---:|---:|
| Targeted - geometry-matched random | +0.40 | +1.41 | -0.902 |
| Targeted - independent-layer random (sensitivity only) | +0.90 | +0.93 | -0.620 |

The 8x8 Transformer-AR condition follows the released intervention algorithms and constants, but uses this thesis checkpoint, corpus, and probe split. Each JEPA run uses the one recorded post-hoc readout for all three methods; automatic selection uses validation evidence and never the final causal test. Mamba patches the mixer branch corresponding to Nanda's attention branch. Those conditions, and larger boards, are extensions.

`smoke` is only an execution gate. Thesis conclusions require `full`.
