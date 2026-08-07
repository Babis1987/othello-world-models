# Mamba-JEPA pipeline audit and result verification

Date: 2026-07-14. Scope: diagnostics only; no training or existing evaluation code was modified as part of this audit.

## Executive conclusion

**H1 is confirmed: the published Mamba next-move numbers are invalid.** The common legal-move evaluator unconditionally enters CUDA fp16 autocast (`othello_research/evaluation/legal_moves.py:71-74, 340`). The trained Mamba residual stream reaches roughly `5e5`, beyond fp16's maximum finite value (65,504), so its forward produces non-finite logits. The raw result JSON records `legal_prob_mass_per_token: null` for both heads (a serialized NaN), while the scorer turns both non-finite probability rows into the same deterministic ranking. That explains all three identical headline metrics (14.39/14.04/13.39) without head collapse or a summary copy bug.

The saved heads are structurally and numerically different: linear weight/bias norms are 70.755/0.943; the MLP tensors have norms 88.382, 1.638, 30.297, and 0.531. Therefore “identical learned heads” is rejected. A small corrected-precision Colab check already recorded in the diagnostics notebook reached about 94.3% linear and 94.0% MLP top-1 legality on 64 games. The full corrected evaluation remains to be run by the user.

VERDICT: **FAIL — supports H1.** Mark every previous Mamba next-move legality number void pending bf16/fp32 re-evaluation.

## Task 1 — static audit

### Feature provenance

| Consumer | Tensor and positions | Layer / stream | Final norm |
|---|---|---|---|
| Linear next-move head | `h[:, t]`, paired with dataset `y[:, t] = token[t+1]`; padding ignored with `TARGET_PAD` | output after all 15 Mamba blocks | `encoder.ln_f` applied |
| MLP next-move head | identical encoder path and positions; only downstream head differs | output after all 15 blocks | `encoder.ln_f` applied |
| Linear board probes | hook activation at every configured layer; layer 0 is embedding/drop stream, layers 1–15 are block residual outputs; all valid game positions | post-block residual stream | no `ln_f` on the last hooked block |
| MLP board probes | same cached tensors as linear probes | same | no `ln_f` |
| Matched-conditions probes | same `ActivationCache` tensors, filtered/scored through matched-condition indices | same | no `ln_f` |

The Mamba encoder deliberately preserves the Transformer-facing attributes (`wte`, `wpe`, `drop`, `blocks`, `ln_f`), so the notebook's manual forward is a valid path rather than a missing adaptation. The invariant test proves `head_path == encode_hidden == ln_f(last_hook)` within `1e-5`. The only mismatch is documented and expected: probes see the raw pre-final-norm residual, heads see its final LayerNorm. Hook provenance is implemented by `ActivationCache` at `board_state.py:191-226`.

### Alignment, target branch, objective, and precision

- Causality: the torch reference causal convolution uses left-effect padding `d_conv-1` and slices back to the original length (`mamba_ar.py:163-168, 194-199`); the scan is forward. Tests show every prefix activation is invariant to changes in later tokens.
- Supervision: `OthelloChunkDataset` shifts the sequence by one and preserves `TARGET_PAD=-100` only on the target side (`dataset.py:44, 130-141`). `hidden[:,t]` therefore predicts token `t+1`; no convolution off-by-one was found.
- Length 1: the causal convolution returns length `1` after cropping and the scan is well-defined. All 60 move-token prototypes were finite and distinct; minimum pairwise distance was 5.9313 and every feature dimension had nonzero token-wise standard deviation.
- VICReg: v1 has no separate EMA target encoder; the hard-disjoint target is encoded by the shared context encoder (`jepa.py:498-502`). No target detach occurs on this v1 path. Variance and covariance are computed on both prediction and target branches by `vicreg_loss` (`jepa.py:171-215, 540-546`). Detaches at 211-215 are only returned logging statistics.
- Training precision was **bf16**, verified from the run's synced `source_config.yml` and `resolved_train_config.yml` (not fp16). The failure is eval-side: legal-move evaluation forces fp16 (`legal_moves.py:71-74`); board hooks immediately cast captured values to fp32 (`board_state.py:226`), explaining why board probes remained usable.
- Summary generation reads the separate `SMOKE_RESULTS['next_move_legal']['linear']` and `['mlp']` entries. Independent regeneration from raw JSON produces the same rows. This is not a shared-dict/copy-paste problem; the raw arrays themselves are identical after the shared non-finite eval failure.

### Ranked suspects

1. **Confirmed — hardcoded fp16 in legal-move evaluation** (`legal_moves.py:71-74, 340`): overflows trained Mamba activations and invalidates both next-move rows. Proposed patch (not applied): accept an explicit eval dtype, default to bf16 for Mamba/CUDA or disable autocast; additionally fail fast when logits/probabilities are non-finite.
2. **Confirmed downstream optimization issue — raw-scale MLP board probes**: hooks expose pre-`ln_f` residuals (`board_state.py:203-226`), whose scale is architecture-dependent. A linear classifier can survive this better than the Transformer-tuned MLP recipe. Proposed evaluation change: sweep LR and add input LayerNorm, as Section D of the notebook does.
3. **Rejected — head/probe extraction mismatch**: exact equivalence up to the explicit final norm was tested.
4. **Rejected — target/label off-by-one or noncausal padding**: causality and alignment tests pass.
5. **Rejected — summary dict reuse or identical checkpoints**: raw JSON is already identical due to NaNs; saved heads have different architectures and nontrivial weights.
6. **Still requires full control — random-reservoir explanation for board probes**: Section C of the notebook must be run before interpreting trained-vs-random board-state accuracy.

## Task 2 — CPU invariant tests

Command: `python -m pytest tests/test_mamba_jepa_invariants.py -q -s`

| Check | Expected | Observed | Verdict | Hypothesis |
|---|---|---|---|---|
| Causality at every layer | prefix unchanged by future tokens | max delta below tolerance | PASS | none |
| Token/target alignment | reacts to token t, ignores t+1; y[t]=token t+1 | satisfied, TARGET_PAD correct | PASS | rejects H1 alignment bug |
| Length-1 prototypes | finite, distinct, nonzero per-dim std | min distance 5.9313; mean std 0.7159 | PASS | rejects H2 structural impossibility |
| Extraction equivalence | head equals normalized hook path | max delta <1e-5; raw pre-norm gap 3.2357 | PASS | rejects H1 provenance bug |
| Right padding | scored prefix unchanged | all layers pass | PASS | rejects padding artifact |

Overall: **6 passed, 1 expected torch-backend warning** in 0.24 s.

VERDICT: **PASS — supports none directly; narrows H1 to evaluation precision.**

## Task 3 — checkpoint forensics

| Check | Expected | Observed | Verdict | Hypothesis |
|---|---|---|---|---|
| Head identity | distinct trained models | different architectures/shapes and nonzero input weights | PASS | rejects head collapse/identity |
| Collapse indicator | input weights nontrivial vs bias | linear 70.755 vs 0.943; MLP input 88.382 vs biases 1.638/0.531 | PASS | rejects bias-only prior |
| Raw eval finiteness | finite probability mass | both `legal_prob_mass_per_token = null`; per-position arrays identical | FAIL | confirms H1 |
| Summary regeneration | separate raw records | same rows regenerated; no shared dict needed | PASS | rejects writer bug |
| Corrected precision micro-check | heads separate and exceed chance | ~94.3% / ~94.0% top-1 legality on 64 games | PASS (screening) | confirms H1; full rerun needed |
| Right checkpoint | trained weights loaded | final.pt reports step 78,200 / 19,999,840 games; full init-stat comparison is in notebook | INCONCLUSIVE locally | Task 4C |
| Fixed-prior baseline | match only if heads truly prior-shaped | unnecessary after direct NaN mechanism; retained as optional notebook control | INCONCLUSIVE | none |

VERDICT: **FAIL — supports H1.** The original metrics measure a non-finite-evaluation fallback, not either learned head.

## Task 4 deliverable

`notebooks/mamba_jepa_diagnostics.ipynb` is generated and organized into independently skippable sections A–E. It adds a precision sweep and a direct inherited-fp16 versus forced-bf16/fp32 reproduction before the requested objective-solved, reservoir-control, hook-head-retrain, and summary-regeneration checks. Helper code is isolated in `othello_research/diagnostics/mamba_jepa_audit.py`; existing evaluation/training modules remain untouched.

## Decision tree conclusion

The first branch fires: an evaluation bug is demonstrated. **Conclusion: H1.** The fix is to make evaluation precision explicit and finite-safe; for this trained bf16 Mamba checkpoint use bf16 or fp32, not fp16. All previously reported Mamba next-move numbers are void until rerun. H2 is not supported by the current evidence, and H3 cannot be claimed because the apparent chance-level affordance result disappears under corrected precision. Board-state claims must still be stated relative to the random-init reservoir control.

## What to rerun (lowest cost first)

1. Notebook A2 and B0: reproduce fp16 overflow and evaluate both saved heads in bf16/fp32 on a small validation subset.
2. Full B0: regenerate the official legal-move metrics for both saved heads with finite checks enabled.
3. Section E: write a corrected summary from the new raw metrics; archive the old next-move rows as invalid.
4. Section A and B: audit the training curves and verify the objective directly with target prototypes.
5. Section C: trained-versus-random Mamba and Transformer board probes (required before thesis interpretation).
6. Section D: hook-path MLP LR/LayerNorm sweep, mainly to resolve the board-probe optimization anomaly.

