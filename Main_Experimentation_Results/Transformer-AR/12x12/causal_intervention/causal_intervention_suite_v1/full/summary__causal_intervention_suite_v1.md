# Causal-intervention comparison

- Condition: `transformer` / `ar` / `12x12`
- Profile: `full`
- Claim scope: `board_extension`
- Shared counterfactuals: occupied-square colour flips that change the legal-move set.

| Method | Published operation | Local intervention | Top-N FP+FN | Target legal mass |
|---|---|---|---:|---:|
| li | Adam optimization through nonlinear absolute probe | `li_nonlinear_activation_optimization_v1` | 5.594 | 0.8994 |
| nanda | fixed normalized linear direction in residual branch | `nanda_target_class_attention_branch_v1` | 2.300 | 0.9082 |
| adapted | target-minus-source edits after every block | `adapted_linear_sequential_residual_v1` | 1.688 | 0.9316 |

The 8x8 Transformer-AR condition follows the released intervention algorithms and constants, but uses this thesis checkpoint, corpus, and probe split. JEPA uses one fixed post-hoc readout for all methods; Mamba patches the mixer branch corresponding to Nanda's attention branch. Those conditions, and larger boards, are extensions.

`smoke` is only an execution gate. Thesis conclusions require `full`.
