# Causal-intervention comparison

- Condition: `transformer` / `jepa` / `8x8`
- Profile: `full`
- Claim scope: `objective_extension`
- Shared counterfactuals: occupied-square colour flips that change the legal-move set.

| Method | Published operation | Local intervention | Top-N FP+FN | Target legal mass |
|---|---|---|---:|---:|
| li | Adam optimization through nonlinear absolute probe | `li_nonlinear_activation_optimization_v1` | 1.800 | 0.9231 |
| nanda | fixed normalized linear direction in residual branch | `nanda_target_class_attention_branch_v1` | 2.168 | 0.9044 |
| adapted | target-minus-source edits after every block | `adapted_linear_sequential_residual_v1` | 0.382 | 0.9545 |

The 8x8 Transformer-AR condition follows the released intervention algorithms and constants, but uses this thesis checkpoint, corpus, and probe split. JEPA uses one fixed post-hoc readout for all methods; Mamba patches the mixer branch corresponding to Nanda's attention branch. Those conditions, and larger boards, are extensions.

`smoke` is only an execution gate. Thesis conclusions require `full`.
