# Transformer-JEPA and Mamba-JEPA full pipeline audit

Date: 2026-07-14

## Bottom line

The codebase is substantially healthier than the anomalous Mamba smoke summary suggested: the full local suite passes (165 tests), the train/validation split is disjoint, Transformer and Mamba use the same downstream data, and the v1 hard-disjoint alignment/causality invariants pass. Nevertheless, the current evidence is **not yet sufficient for an unconditional thesis claim that Mamba-JEPA outperforms Transformer-JEPA by 21.4pp**. The effect is promising and very large, but the official Mamba smoke result is invalid, evaluation precision is not symmetric, downstream runs are unseeded, and architecture-specific probe controls are missing.

The claim becomes thesis-grade after a short controlled rerun protocol listed at the end.

## Severity-ranked findings

### Critical — official Mamba next-move rows are invalid

`othello_research/evaluation/legal_moves.py:71-74` unconditionally forces CUDA fp16. The trained Mamba forward overflows in this context; the raw JSON contains `legal_prob_mass_per_token = null` for both heads. NaN rows then yield deterministic `argmax/topk` outputs and the identical 14.39/14.04/13.39 rows. JSON conversion silently maps NaN to `null`, so summary generation does not fail.

The diagnostic bf16/fp32 results (linear top-1 legality 94.27%) strongly recover the intended measurement, but they are not yet the official smoke artifact.

Impact: old Mamba next-move rows are void. H1 is confirmed; this does not imply a training failure.

Required fix: make eval dtype explicit, reject any non-finite logits/probabilities/aggregates, and regenerate the raw JSON and summary.

### High — Transformer/Mamba comparison currently uses asymmetric evaluation precision

The published Transformer smoke metrics were produced by the common fp16 evaluator and are finite (72.90% linear, 79.43% MLP). The corrected Mamba metrics use an internal fp32/bf16 override. Even though Transformer fp16 is stable, a controlled architecture comparison must run **both saved heads through the same fp32 and bf16 evaluator**. Otherwise precision remains a procedural confound.

Training precision also differs: the 20M Transformer v1 checkpoint records fp16, while Mamba records bf16. This is a legitimate engineering choice but a confound when attributing the entire difference solely to architecture.

### High — downstream head/probe training is not seeded

Neither smoke notebook calls `torch.manual_seed`, `torch.cuda.manual_seed_all`, or supplies a seeded DataLoader generator. Head initialization, probe initialization, dropout, and shuffle order therefore vary by notebook execution. The matched-condition helper is correctly seeded (`matched_prefix_probe.py:122-143, 257-301`), but the main heads and all-position probes are not.

Impact: each reported head/probe number is one stochastic downstream draw with no error bar. A 21pp effect is unlikely to disappear, but its magnitude and MLP/linear gaps cannot be reported as precise architecture effects without repeated seeds.

Required control: 3–5 fixed downstream seeds for both frozen checkpoints; report mean, standard deviation, and preferably paired differences on identical validation games.

### High — board-probe comparison is not scale-robust across architectures

`ActivationCache` hooks raw embedding/post-block residual outputs (`board_state.py:191-226`), not final-normalized features. Transformer and Mamba residual scales differ greatly. The same MLP optimizer was tuned on Transformer features; the Mamba MLP probe's ~61–62% plateau below its linear probe is therefore an optimization/normalization failure, not evidence that nonlinear information is absent.

The linear Mamba probe result may still be real, but a recurrent random network can serve as a reservoir. Until the random-init Mamba and Transformer controls are run, the 80–84% Mamba board score cannot be attributed to JEPA learning.

Required controls: input LayerNorm/standardization; seeded LR sweep; random-init encoders; trained-minus-random deltas. Compare final-normalized features separately from raw residual-stream probes.

### Medium — reported “next-move accuracy” is legality, not continuation accuracy

`legal_moves.py:126-180` checks whether ranked predictions belong to the legal set before the next move. It does not ask whether the argmax equals the sampled next move. The 94.27% result is **top-1 legality**, not exact next-token accuracy. Legal probability mass is also not calibrated accuracy.

Impact: thesis wording must use “top-1 legal-move rate” or “legality”, not “next-move accuracy”. Exact continuation top-1 and cross-entropy should be reported separately if desired.

### Medium — model depth indices are not directly comparable

Transformer probes use L0–L8 (embedding plus 8 blocks); Mamba uses L0–L15 (embedding plus 15 blocks). L7 in the two models is not equivalent depth. Compare normalized depth, final layer, peak layer, and area-under-depth profile rather than identical integer layer labels.

The Mamba model has 15 blocks, not 16; L0 is the embedding stream.

### Medium — exact historical code provenance is incomplete

Checkpoints save model/train configs and artifacts point back to `final.pt`, which is good. However, runs do not consistently embed a git commit and source snapshot sufficient to reconstruct the exact code used. Evaluation notebooks install the mutable Drive checkout in editable mode. A later code change can therefore evaluate an old checkpoint with new semantics while retaining the same run name.

Required fix for future runs: save git commit, dirty-worktree diff/hash, package versions, dataset manifest/hashes, notebook hash, and evaluator version into every checkpoint/results JSON.

### Medium — validation protocol is deterministic but narrow

Both v1 runs use identical 20 head-training chunks and identical 38 held-out validation chunks, with zero path overlap. Legal evaluation uses the first 5,000 held-out games in deterministic chunk order. Board probes train on 5,000 games but validate on only 512 games. This is comparable, not leakage, but it samples one fixed prefix of the validation distribution.

Required control: bootstrap confidence intervals over games and repeat on multiple disjoint validation slices.

### Low — Mamba notebook finite-loss error path has a typo

In the Mamba smoke notebook, `train_board_probe_bank` raises an error referencing undefined `head_type` when a board-probe loss is non-finite (cell 12). It would produce `NameError` instead of the intended diagnostic exception. It did not alter completed finite probe results, but should reference `probe_type`.

### Low — top-1 is exposed through two aggregation fields

The evaluator returns dedicated `top1_legal_per_token` and also `topk_legal[1]`. They should be mathematically identical; tiny differences in stored results warrant keeping one canonical field plus an invariant assertion. This is not responsible for the architecture gap.

## Checks that passed

- Full repository suite: **165 passed**, 7 non-failing warnings.
- Mamba-specific causality, alignment, length-1 target, extraction equivalence, and padding tests: all pass.
- Dataset uses `TARGET_PAD=-100` only for targets and a valid input pad embedding (`dataset.py:44, 130-141`).
- Train/validation chunk resolution checks path overlap (`train_jepa.py:388-430`).
- Both 20M v1 runs use the same downstream train/validation chunk basenames; no train/validation path overlap.
- Both saved head artifacts point to their corresponding `final.pt` and use 20 training chunks.
- Transformer raw metrics are finite; its linear/MLP predictions are distinct.
- A sweep of all available smoke result JSON files found non-finite head metrics only in the original Mamba v1 smoke evaluation.
- Mamba checkpoint state dict contains real SSM parameters (`A_log`, `D`, `conv1d`, `x_proj`, `dt_proj`); Transformer contains attention/MLP parameters.
- v1 hard-disjoint target semantics are implemented as future-only target tokens with absolute positions (`jepa.py:559-601`), shared encoder, and VICReg on both branches (`jepa.py:171-215, 540-546`).
- The v1 saved checkpoints agree on objective settings: hard-disjoint future, K=1, linear predictor, VICReg 25/25/1.

## Experiment-family interpretation boundaries

The repository's JEPA runs are not a single controlled axis. They should be grouped before comparison:

| Family | Key semantic difference | Directly comparable to v1 architecture test? |
|---|---|---|
| v1 | shared encoder, no stop-gradient, VICReg | yes, when data/budget/precision/eval are matched |
| v2 | EMA frozen target, Smooth L1 | no; target topology and loss differ |
| v3 | nested K=8, MLP predictor | no |
| v4 | disjoint/multi-position predictor, often different K/EMA choice | no |
| v5 | contrastive InfoNCE and prefix masking/action view | no |
| v6 | hard-disjoint action margin objective/grouped branching data | no |
| v7 | action-conditioned; Smooth L1/InfoNCE/hybrid, sometimes CE injection | compare only within matched v7 ablations |
| v8 | order-aware pair-index sampling and hard negatives/hybrid variants | compare only with identical pair index and sampling budget |
| v9 | EMA multi-action latent rollout, horizon weighting, Smooth L1/InfoNCE | no |

Several configs also differ in pretraining chunks (10, 50, or 200), precision, batch construction, and effective optimizer steps. Raw smoke rankings across all v1–v9 runs must not be presented as architecture-only or objective-only effects without controlling those axes.

## Current matched v1 evidence

| Item | Transformer v1 | Mamba v1 | Status |
|---|---:|---:|---|
| Games seen | 19,999,836 | 19,999,840 | matched |
| Downstream head chunks | 20 | 20 | identical basenames |
| Validation chunks | 38 | 38 | identical basenames |
| Head training precision | bf16 | bf16 | matched |
| Encoder training precision | fp16 | bf16 | confounded |
| Official eval precision | forced fp16 | forced fp16 (overflow) | invalid for Mamba |
| Corrected linear top-1 legality | not rerun yet | 94.27% fp32/bf16 | comparison pending symmetric rerun |
| Stored linear top-1 legality | 72.90% finite | 14.39% non-finite artifact | do not compare |
| Stored MLP top-1 legality | 79.43% finite | 14.39% non-finite artifact | do not compare |

## Minimum rerun protocol for a thesis-grade claim

1. Freeze one evaluator version and run **both** Transformer and Mamba saved linear/MLP heads in fp32 and bf16 with fail-fast finite assertions.
2. Report top-1 legality, top-k legal fraction, legal probability mass, exact continuation top-1, and next-token cross-entropy on the same ordered game IDs.
3. Retrain downstream heads for seeds `{0,1,2,3,4}` with identical initialization policy, chunk order, DataLoader generator, optimizer, and precision; report paired mean ± SD and bootstrap 95% CIs.
4. Regenerate raw JSON and summaries; mark the original Mamba summary invalid rather than overwriting it silently.
5. Run random-init Transformer/Mamba board-probe controls and normalized-feature probe variants before interpreting board-state superiority.
6. Repeat legality evaluation on at least three disjoint validation slices.
7. Record code commit/diff, CUDA, PyTorch, mamba-ssm, causal-conv1d, dataset manifest, checkpoint SHA-256, and evaluator settings in the final artifact.

## Audit verdict

**The architecture finding is credible but provisional.** The corrected Mamba linear legality result is numerically stable between fp32 and bf16 and the gap versus the finite Transformer result is large. No alignment, split, checkpoint-identity, or causality bug currently explains it. However, symmetric precision evaluation and seeded repetitions are required before stating the magnitude as a definitive Transformer-vs-SSM result. Board-state superiority remains explicitly provisional until random-init and normalization controls are complete.
