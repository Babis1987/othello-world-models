# Causal-intervention comparison

- Condition: `mamba` / `jepa` / `12x12`
- Profile: `full`
- Claim scope: `architecture_objective_board_extension`
- Shared counterfactuals: occupied-square colour flips that change the legal-move set.
- JEPA readout: `linear`; policy `best`, chosen only from the selection split by `best_selection_legal_probability_mass` (linear=0.955027, mlp=0.952094).

| Method | Published operation | Local intervention | Top-N FP+FN | Target legal mass |
|---|---|---|---:|---:|
| li | Adam optimization through nonlinear absolute probe | `li_nonlinear_activation_optimization_v1` | 2.876 | 0.9223 |
| nanda | fixed normalized linear direction in residual branch | `nanda_target_class_attention_branch_v1` | 2.848 | 0.9221 |
| adapted | target-minus-source edits after every block | `adapted_linear_sequential_residual_v2` | 1.238 | 0.9474 |

The primary adapted-method random control applies one random signed permutation per case at every layer. It therefore preserves the norms and all cross-layer angles of the probe directions while breaking their alignment with the learned feature coordinates.

| Adapted comparison | Delta top-1 (pp) | Delta target mass (pp) | Delta Top-N FP+FN |
|---|---:|---:|---:|
| Targeted - geometry-matched random | +4.80 | +3.18 | -1.732 |
| Targeted - independent-layer random (sensitivity only) | +3.80 | +2.63 | -1.634 |

The 8x8 Transformer-AR condition follows the released intervention algorithms and constants, but uses this thesis checkpoint, corpus, and probe split. Each JEPA run uses the one recorded post-hoc readout for all three methods; automatic selection uses validation evidence and never the final causal test. Mamba patches the mixer branch corresponding to Nanda's attention branch. Those conditions, and larger boards, are extensions.

`smoke` is only an execution gate. Thesis conclusions require `full`.
