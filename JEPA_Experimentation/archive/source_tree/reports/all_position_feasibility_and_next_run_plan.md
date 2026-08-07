# All-position JEPA feasibility and next-run plan

## Scope and terminology

The current codebase defines public JEPA variants **v1 through v9**. There is
no v10 implementation or config. `family_infonce` is a separate, indexed,
step-based experiment family; it should not be called v10 unless the thesis
explicitly introduces that naming convention.

Here, *all-position* means that every valid causal boundary of every sampled
game contributes objective signal. It does **not** mean flattening positions
indiscriminately. Batch-statistical objectives must compare examples at the
same absolute boundary, and grouped objectives must preserve their positive
and negative-set semantics.

## Feasibility matrix

| Variant | Existing objective and sampling unit | Can all-position preserve the objective? | Main risk / required implementation | Expected cost relative to its random-boundary version | Thesis value | Recommendation |
|---|---|---|---|---|---|---|
| v1 | Hard-disjoint K=1, shared encoder, VICReg; one shared random boundary per ordinary minibatch | **Yes; implemented** | VICReg variance/covariance must be computed across games separately at each position. Flattening positions creates positional-embedding shortcuts. | About 58x more latent pairs on 8x8, but observed wall time is much less than 58x because the causal context pass is shared | **Very high**: direct supervision-density ablation against AR | **Run now** on both architectures and selected board sizes |
| v2 | EMA target, Smooth L1, usually hard-disjoint K=4 with Transformer predictor | **Yes, technically straightforward** | Batch all K-token target windows while sharing the causal context pass. The existing reference used only 50 chunks, so density cannot be isolated without also creating a matched 20M random-boundary baseline | Roughly 55 windows/game instead of one; target/predictor work is substantially higher | Medium: tests whether density helps EMA regression too | **Second-wave candidate**, not tomorrow |
| v3 | Nested EMA target, Smooth L1, weak MLP predictor, K=8 | **Yes, with a new vectorized path** | Run context and EMA target encoders once causally, then gather all `(t,k)` targets. A naive prefix loop would be quadratic. Long-horizon ambiguity remains part of the objective | High: roughly 51 starting boundaries x 8 predicted horizons/game | Low-to-medium because v2 already represents EMA/Smooth-L1 and v3 changes predictor plus horizon | **Do not prioritize** |
| v4 | Multi-position K=4 predictor; configs include EMA/Smooth-L1 and shared-encoder VICReg | **Yes, but variant-dependent** | EMA version is similar to v2. VICReg version requires statistics stratified by both boundary and horizon; overlapping windows from one game must not be treated as independent covariance samples | High | Medium as a multi-step-density ablation, but interpretation is less clean | **Only after v2**, if multi-step prediction becomes a thesis question |
| v5 | Hard-disjoint action InfoNCE with exact-prefix grouped batches; same-prefix rows are positives | **No simple all-position conversion** | Every boundary would need a corpus-wide exact-prefix regrouping/index. Flattening game positions destroys positive groups and creates false negatives; self-positive-only batches change the objective | Offline index and training cost potentially very large | Low for the main 2x2 thesis; high risk of becoming a new objective | **Do not convert mechanically** |
| v6 | Hard-disjoint action-margin loss with prefix groups and multi-mode predictor | **No simple all-position conversion** | Same grouping problem as v5. Pull/push sets and phantom-token deduplication are defined at one grouped prefix, not across arbitrary positions | Very high and data-index dominated | Low; existing v6 already changes several axes simultaneously | **Do not run all-position** |
| v7 Smooth-L1 | One-step action-conditioned predictor, nested EMA target, ordinary random-boundary minibatches | **Yes; best structured candidate** | Share full causal context/EMA-target passes and evaluate every transition `(state_t, action_t) -> state_{t+1}`. No contrastive negatives are required | About 58 transitions/game on 8x8; predictor work increases but both encoder passes can be shared | **High**: tests dense transition learning, distinct from v1 next-action latent prediction | **Best optional follow-up after v1** |
| v7 InfoNCE / hybrid | Same-(t, action) grouped InfoNCE, optionally with CE grounding | **No simple conversion** | Must rebuild grouped batches for every `(t, action)`. Naive all-position flattening changes negatives and the hybrid CE term confounds the density question | High plus grouping/index overhead | Medium only as a separate grouped-objective study | **Do not include in the first all-position sweep** |
| v8 | Order-aware indexed positives and hard negatives at selected prefixes | **Not meaningfully as ordinary all-position** | Only prefixes with valid cross-order relations have the intended semantics. Exhaustive indexing is possible but is a *denser pair-index experiment*, not the v1-style all-position intervention | Very high offline index/storage cost | Low for supervision-density claim; existing v8-improvement configs already test index density | **Keep as indexed order-aware experiments** |
| v9A | Multi-action EMA latent rollout with weighted Smooth L1 and anti-collapse terms | **Yes, but complex** | Enumerate every rollout start, share causal encoders, and stratify variance/covariance by start position and rollout step. Overlapping rollouts must not inflate independent-sample counts | Very high: O(valid starts x horizon) predictor work | Medium-high if multi-step world-model rollout becomes central | **Third-wave candidate** |
| v9B | Multi-action rollout with per-step InfoNCE | **Possible only with careful stratification** | Negatives must be restricted to the same absolute start and rollout step. Otherwise position embeddings solve the contrastive task and overlapping same-game rollouts create false negatives | Very high memory/compute | Medium, but difficult to interpret cleanly | **Do not prioritize** |
| family-InfoNCE (separate family; not v10) | Offline transposition-family index, family-balanced step-based InfoNCE | **Already position-indexed in a different sense** | It samples valid transposition families rather than games. "Every game position" would mostly produce no positive family and would change the sampling distribution completely | Dominated by exhaustive index construction | Separate research question, not a v1 density control | **Do not relabel or convert as v10** |

## Recommended experimental scope

### Required for the main thesis

1. v1 all-position Transformer and Mamba on 8x8.
2. v1 all-position Transformer and Mamba on 12x12.
3. Compare against the already trained Transformer-AR and Mamba-AR controls
   using one identical legal-move and normalized board-probe evaluator.

This is sufficient to test the central claim: whether architecture and board
size interact with sparse versus dense JEPA supervision.

### Optional extension after the required matrix is complete

Implement **v7 Smooth-L1 all-position** as the single structured representative.
If time remains, implement **v2 EMA/Smooth-L1 all-position**. Do not launch the
all-position checkpoint alone: density claims require a random-boundary
baseline with the same 20M budget, batch size, precision, model, and horizon.

Do not run v5, v6, grouped v7, or v8 under an "all-position" label. Doing so
would require new grouping/index semantics and would no longer be a controlled
supervision-density ablation. Leave v9 for a later multi-step-rollout study.

## Training-time protocol for the thesis

Every reported runtime row must include:

- GPU model and VRAM;
- software backend (`torch`, CUDA, and `mamba_ssm` versions where applicable);
- precision, batch size, board size, parameter count, optimizer steps;
- games processed and **effective supervision units** processed;
- median, p10, and p90 seconds per training-only chunk;
- games/second and supervision-units/second;
- estimated pure-training hours and observed training+validation loop hours;
- whether checkpoint serialization, Drive sync, installation, corpus loading,
  evaluation, and probe training are included.

The existing `metrics.csv` files already preserve `dt_seconds`, `chunk_games`,
`chunk_tokens`, `games_seen`, and `tokens_seen`. The timer starts before the
chunk is loaded and ends after its optional validation. It normally excludes
the per-chunk `latest.pt` serialization and Drive sync, which happen after the
row is measured. Step-fraction checkpoints may occasionally be saved inside a
timed training pass. Therefore:

- use rows with `val_games == 0` for training-only chunk timing;
- use the median rather than a single smoke chunk;
- compare models only on the same GPU class;
- report both games/s and supervision-units/s, because v1 all-position sees
  about 58x more latent pairs than random-boundary v1 on 8x8;
- time downstream evaluation/probe training separately from pretraining.

Use:

```bash
python scripts/summarize_training_times.py RUN_DIR_1 RUN_DIR_2 --json-out reports/training_times.json
```

The generated `estimated_training_hours` is median training-only chunk time
multiplied by the number of chunks. `observed_train_plus_validation_loop_hours`
is the sum of recorded chunk timers. Neither should be described as end-to-end
notebook wall time.

### Provisional inventory from existing 8x8 metrics

The following values prove that the historical CSVs are usable, but they are
**not yet a fair architecture runtime table**, because GPU model/session is not
stored with all old runs:

| Run | Median non-validation chunk | Estimated training-loop time | Effective supervision units |
|---|---:|---:|---:|
| Transformer-JEPA v1 random-boundary | 9.92 s | 0.55 h | 19,991,681 |
| Transformer-JEPA v1 all-position | 28.72 s | 1.60 h | 1,159,519,681 |
| Mamba-JEPA v1 random-boundary | 58.89 s | 3.27 h | 19,991,706 |
| Transformer-AR | 101.14 s | 5.62 h | 1,179,328,592 |
| Mamba-AR | 36.13 s | 2.83 h | 1,179,328,592 |

These numbers must not be used to claim that one architecture is faster until
the GPU class, batch settings, precision, and backend are matched. The useful
within-run finding is that Transformer v1 all-position processed about 58x as
many supervision pairs as random-boundary v1 while its median recorded chunk
time was about 2.9x larger; this is an implementation-efficiency observation,
not yet a hardware-controlled architecture comparison.

## Next-run checklist

1. Record the Colab GPU name before launching anything. Use the same GPU class
   for runs that will appear in the same runtime table.
2. Preserve the completed Transformer-v1-all-position 8x8 artifacts and final
   summary; do not rerun it unless GPU-matched timing is required.
3. Train and evaluate Mamba-v1-all-position 8x8.
4. Train and evaluate Transformer-v1-all-position 12x12.
5. Train and evaluate Mamba-v1-all-position 12x12.
6. Evaluate the completed Mamba-AR 12x12 checkpoint.
7. Use identical `normalized_v2` linear/MLP probe settings for all four model
   families. For cross-board bounded diagnostics use phase-matched ranges:
   8x8 `[4,44]`, 12x12 `[10,105]`, 16x16 `[17,190]`.
8. Generate timing summaries immediately after every completed training run
   and store the JSON with the run reports.
9. Only after the required matrix is complete, decide whether to implement
   v7 Smooth-L1 all-position and then v2 all-position.
