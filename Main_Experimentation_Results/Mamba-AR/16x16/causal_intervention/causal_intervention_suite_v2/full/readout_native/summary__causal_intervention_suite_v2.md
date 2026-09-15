# Causal-intervention comparison

- Condition: `mamba` / `ar` / `16x16`
- Profile: `full`
- Claim scope: `architecture_board_extension`
- Shared counterfactuals: occupied-square colour flips that change the legal-move set.
- Readout: native AR language-model head.

| Method | Published operation | Local intervention | Top-N FP+FN | Target legal mass |
|---|---|---|---:|---:|
| li | Adam optimization through nonlinear absolute probe | `li_nonlinear_activation_optimization_v1` | 4.848 | 0.8945 |
| nanda | fixed normalized linear direction in residual branch | `nanda_target_class_attention_branch_v1` | 4.676 | 0.8889 |
| adapted | target-minus-source edits after every block | `adapted_linear_sequential_residual_v2` | 3.870 | 0.9036 |

The primary adapted-method random control applies one random signed permutation per case at every layer. It therefore preserves the norms and all cross-layer angles of the probe directions while breaking their alignment with the learned feature coordinates.

| Adapted comparison | Delta top-1 (pp) | Delta target mass (pp) | Delta Top-N FP+FN |
|---|---:|---:|---:|
| Targeted - geometry-matched random | +5.10 | +1.73 | -1.258 |
| Targeted - independent-layer random (sensitivity only) | +3.00 | +1.47 | -0.944 |

The 8x8 Transformer-AR condition follows the released intervention algorithms and constants, but uses this thesis checkpoint, corpus, and probe split. Each JEPA run uses the one recorded post-hoc readout for all three methods; automatic selection uses validation evidence and never the final causal test. Mamba patches the mixer branch corresponding to Nanda's attention branch. Those conditions, and larger boards, are extensions.

`smoke` is only an execution gate. Thesis conclusions require `full`.
