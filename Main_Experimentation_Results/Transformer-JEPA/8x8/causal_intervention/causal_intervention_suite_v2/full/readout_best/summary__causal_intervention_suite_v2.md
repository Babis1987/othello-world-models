# Causal-intervention comparison

- Condition: `transformer` / `jepa` / `8x8`
- Profile: `full`
- Claim scope: `objective_extension`
- Shared counterfactuals: occupied-square colour flips that change the legal-move set.
- JEPA readout: `linear`; policy `best`, chosen only from the selection split by `best_selection_legal_probability_mass` (linear=0.978020, mlp=0.976177).

| Method | Published operation | Local intervention | Top-N FP+FN | Target legal mass |
|---|---|---|---:|---:|
| li | Adam optimization through nonlinear absolute probe | `li_nonlinear_activation_optimization_v1` | 1.792 | 0.9245 |
| nanda | fixed normalized linear direction in residual branch | `nanda_target_class_attention_branch_v1` | 2.156 | 0.9072 |
| adapted | target-minus-source edits after every block | `adapted_linear_sequential_residual_v2` | 0.364 | 0.9611 |

The primary adapted-method random control applies one random signed permutation per case at every layer. It therefore preserves the norms and all cross-layer angles of the probe directions while breaking their alignment with the learned feature coordinates.

| Adapted comparison | Delta top-1 (pp) | Delta target mass (pp) | Delta Top-N FP+FN |
|---|---:|---:|---:|
| Targeted - geometry-matched random | +7.50 | +7.28 | -2.106 |
| Targeted - independent-layer random (sensitivity only) | +7.70 | +6.14 | -2.124 |

The 8x8 Transformer-AR condition follows the released intervention algorithms and constants, but uses this thesis checkpoint, corpus, and probe split. Each JEPA run uses the one recorded post-hoc readout for all three methods; automatic selection uses validation evidence and never the final causal test. Mamba patches the mixer branch corresponding to Nanda's attention branch. Those conditions, and larger boards, are extensions.

`smoke` is only an execution gate. Thesis conclusions require `full`.
