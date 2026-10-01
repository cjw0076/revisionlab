# Research proposal: chaotic data and delayed correction

**Unverified hypothesis:** Cosmos may benefit more from structured, difficult dynamics than from heavily curated data. This is a proposed experiment, not a result or a property established by RevisionLab. The current release contains fixed-capacity residual/RLS correctors and delayed-feedback contracts; native Cosmos expansion, v14 splitting, and new training are outside this note.

The useful question is whether an advantage comes from deterministic chaos, changing regimes, observation corruption, or preprocessing. Do not combine those conditions into one “messy data” benchmark and attribute any gain to chaos.

## Separate the experimental factors

| Factor | Proposed condition | What the comparison isolates |
| --- | --- | --- |
| Deterministic dynamics | Lorenz and Mackey–Glass candidates, with preregistered parameter settings spanning contrasting dynamics | Dependence on the underlying dynamics. |
| Nonstationarity | Explicit parameter/regime switches on otherwise matched streams | Adaptation to changed dynamics rather than chaos alone. |
| Observation noise | Independently seeded measurement corruption at locked levels | Robustness to limited observations. |
| Outliers/missingness | Separate corruption masks and magnitudes | Recovery from isolated disruptions. |
| Curation/preprocessing | Raw observations, train-fit normalization, and separately specified smoothing/filtering | Effects of observation transforms. |

Lorenz's original study describes deterministic trajectories that can diverge after small changes in initial conditions, motivating a predictability limit rather than a promise that difficult dynamics always favor a particular learner. See [Lorenz, *Deterministic Nonperiodic Flow*](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2).

Normalization is not automatic evidence that chaos has been removed. Keep the latent trajectory, sampling times, observation channels, forecast horizons, and feedback delays identical when comparing preprocessing. Fit transforms on training data only and report errors in original target units. Train each arm's base predictor equivalently in that arm's coordinates, or map every arm back to declared common coordinates before using one shared base; lock this choice in advance. Feeding normalized inputs to a base trained only on raw inputs would manufacture a mismatch, not test curation. Record clipping/update units and map forecasts consistently back to target units. Filtering may change available information or effective delay; record both instead of treating it as interchangeable with rescaling.

## Lock the protocol before looking at held-out results

1. **Generator manifest:** specify equations/reference implementation, parameters, initial conditions/history, solver, tolerances, integration step, burn-in, sample interval, observed coordinates, forecast targets, and trajectory duration. Verify numerical stability independently of learning. Do not label every parameter setting chaotic without a stated diagnostic.
2. **Seed and parameter manifest:** use multiple initial-condition, corruption, and model seeds with distinct roles. A proposed starting grid is at least three parameter settings per system and five trajectory seeds per setting; lock exact values and any training seeds before evaluation. Reserve parameter settings and trajectories for held-out transfer, not only new windows from the same trajectory.
3. **Chronology:** split each designated stream into ordered train/validation/test periods and declare warmup/boundary handling. Remove overlapping target/lookback windows and prevent future-label leakage across boundaries. Tune on validation only. Lock hyperparameters before test; if online adaptation is allowed during test, it may consume only already-matured feedback and must score each forecast before its write.
4. **Matched observations:** reuse the same latent trajectory and sampling for raw/normalized/filtered pairs. Add regime changes, noise, and outliers as separate factors before considering interactions. Keep a clean stationary condition and simple periodic/persistence controls.
5. **Delay and predictability:** declare forecast lead and label-arrival delay separately, both in sample counts and physical time. Choose a locked delay grid, such as 1/4/16 samples, with adequate stream length. Estimate local divergence/Lyapunov-related timescales with an independent diagnostic and report sensitivity to that estimate; do not convert it into a universal forecastability threshold. Report results before and beyond the measured useful prediction range.
6. **Freeze the decision rule:** specify a practically meaningful effect size, uncertainty procedure, test families, resource cap, and stop/failure rules before opening held-out results. Multiple windows from one trajectory are correlated observations, not independent experimental seeds.

## Comparisons and resource accounting

Start with current fixed-capacity methods: frozen/no-write, normalized LMS residual memory, RLS with validation-tuned forgetting, reset/adaptive-rate LMS, and wrong-address writes. Reset/adaptive LMS are proposed baselines, not advertised v0.1 features. Within each observation arm, all correctors share its equivalently trained and then frozen base predictor, inputs, transforms, released labels, and event clock; score each method's own issued forecasts. Across preprocessing arms, follow the coordinate/training rule above with equal base-training budgets. Give correctors equal validation-search budgets and disclose reset triggers and tuning choices.

Report both equal-slot comparisons and a separate total-memory-budget comparison. RLS covariance, replay pending tickets, settled audit IDs, preprocessing state, and diagnostic overhead count toward total memory; equal coefficient counts do not imply equal memory cost. Measure prediction/write time, update counts, and accumulated event-clock lag on the same hardware/dtype. Keep diagnostic work out of the learner or count it explicitly if used for decisions.

Only after a native split mechanism exists, add split/no-split/mixture controls under declared capacity, tuning, and total-resource budgets. Preserve prediction-time owner/address version and define how pending tickets survive a split. Include sham splits and a matched mixture baseline so extra capacity, resets, or routing cannot automatically be credited to native expansion. This later phase is deferred and cannot be inferred from current fixed-memory results.

## Outcomes and falsification

Primary outcome: held-out, horizon-resolved prediction error in original units versus the strongest budget-matched baseline, with preregistered uncertainty and an effect-size threshold. Select one reference baseline family using a locked joint-validation rule before test, keep that same family in both sides of each paired condition contrast, and tune/train it equivalently within each arm. Do not choose a different favorable comparator after inspecting test results.

“Benefits more” requires a method-by-condition interaction: compare the candidate's relative error reduction over that same reference baseline in chaotic versus control dynamics, and separately in raw versus normalized/filtered arms. Preregister the interaction direction, practical margin, aggregation weights, and uncertainty procedure. A lower error on a chaotic stream alone, or a larger absolute error reduction caused by a worse baseline error scale, does not establish the hypothesis. Fixed-memory tests can audit this protocol, but cannot confirm a Cosmos-specific interaction until an actual Cosmos implementation is evaluated.

Also report recovery time after a regime switch/corruption event, extreme-event error under a threshold fixed from training, write failures, and runtime/memory/clock cost. Show each factor and parameter setting separately, including failures; aggregate scores must not hide a failed regime.

The hypothesis is weakened or falsified if:

- tuned forgetting/reset/adaptive LMS or a matched mixture removes the claimed advantage;
- gains follow regime changes or outlier handling but do not persist on clean chaotic streams;
- matched raw versus normalized observations show no relevant difference, contradicting a claim that curation itself caused the disadvantage;
- the preregistered method-by-chaos or method-by-preprocessing interaction fails its held-out practical-effect criterion against the same reference baseline;
- improvement requires more state, compute, privileged features, premature labels, or unmatched tuning;
- gains fail on held-out parameter settings/trajectory seeds or appear only where all forecasts are beyond useful predictability;
- shuffled/no-write controls perform similarly, or an effect disappears when leakage and ownership errors are removed.

Independent random-noise targets are a negative control, not a promised opportunity for Cosmos. An apparent advantage there requires a leakage/finite-sample audit before interpretation. A successful synthetic study would justify a next experiment; it would not establish superiority on natural chaotic systems, global BPTT credit assignment, or the unimplemented Cosmos architecture.

This note is an authoring proposal. A separate scientific reviewer must evaluate the locked protocol before any experiment begins; no reviewer approval or results are asserted here.
