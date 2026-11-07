# Active-learning process draft

This document describes the three active-learning processes and the metrics
used to evaluate them. The configured hit tolerance is shared by acquisition
and final QM labelling.

## Shared tolerance rule

The AL configuration supports property-specific relative and absolute
tolerances:

```yaml
al:
  tolerances:
    default:
      relative: 0.20
      absolute: 0.0
    HLgap:
      relative: 0.10
      absolute: 0.05
    eMBD:
      relative: 0.05
      absolute: 0.02
```

For a target value `t`, the allowed absolute error is:

```text
delta = max(relative * abs(t), absolute)
```

This same `delta` is used by:

1. acquisition hit probability;
2. final QM hit labelling;
3. hit-rate evaluation.

If a property is not configured, its relative tolerance falls back to
`query.filter_threshold`.

## 1. Search using KDE

### Purpose

KDE chooses property-space targets for the next generation iteration. It does
not select individual molecules.

### Call path

```text
src/scripts/forward.py
  -> active_learning.plan_block.run_plan()
  -> active_learning.query_planner.plan_targets()
  -> plan_kde_map()
  -> score_targets()
  -> Corridor.sample()
```

### When it runs

It runs in the forward job before dataset splitting, model training, and
molecule generation. The result is written to the iteration's `targets.json`.

### Inputs

- trajectory property rows;
- cumulative HDF5 labels loaded by `state.load_state()`;
- previous requested/achieved properties and hit flags;
- `al.density.*` weights and KDE bandwidth;
- corridor padding and lookahead.

### Scoring components

```text
target score =
    w_sparse  * sparse-region score
  + w_err     * previous target-error score
  + w_explore * under-explored-region score
```

Targets with sufficiently many previous observations and low estimated
reachability can be excluded. Selected targets are separated using repulsion
so one iteration does not spend all its budget at the same point.

### Outputs

`targets.json` stores each target's properties, sample allocation, strategy,
KDE score, effective sample count, and reachability estimate.

### Related metrics

- **Corridor coverage:** estimated after QM labels are appended.
- **Hit rate:** measures whether KDE-selected targets were achievable.
- **Per-target hit rate:** identifies difficult regions hidden by an aggregate
  hit rate.

## 2. Acquisition function

### Purpose

Acquisition chooses which generated molecules receive expensive relaxation and
QM calculations.

### Call path

```text
src/scripts/backward.py
  -> active_learning.acquire_block.run_acquire()
  -> active_learning.surrogate.build_surrogate()
  -> active_learning.acquisition.select_batch()
  -> p_hit()
```

### When it runs

It runs after generation and score filtering but before geometry relaxation and
property calculation.

### Candidate score

```text
score =
    w_hit * predicted hit probability
  + w_unc * surrogate uncertainty
  + w_nov * property novelty
  + w_div * molecular diversity
  - w_cost * estimated cost
```

The batch is selected separately per planned target, subject to the global
acquisition budget.

### Tolerance use

For each surrogate property, `p_hit()` receives the configured property-specific
`delta`. This makes acquisition probability use the same success definition as
the final QM label.

### Outputs

`acquisition.csv` records predictions, uncertainty, novelty, diversity, cost,
score, status, and selection flags for every candidate.

### Related metrics

- selected count and budget usage;
- mean acquisition score;
- surrogate nMAE and RMSE after QM truth is known;
- uncertainty calibration;
- hit rate of selected molecules.

## 3. Evaluation and benchmark metrics

### Call path

```text
src/scripts/backward.py
  -> active_learning.hdf5_block.prepare_al_labels()
  -> pipeline_blocks.to_hdf5.run_to_hdf5()
  -> active_learning.evaluate_block.run_evaluate()
  -> active_learning.metrics.compute_metrics()
  -> append_metrics()
  -> early_stop()
```

### When it runs

Evaluation runs after valid QM results are appended to cumulative HDF5. The
row is appended to the experiment's `metrics.csv`.

### Metrics

#### Hit rate

```text
hits / valid labelled molecules in the current iteration
```

Higher is better. A molecule is a hit only when every non-`atNUM` property is
within its configured tolerance.

#### Corridor coverage

The corridor is sampled and `kernel_regress()` estimates the effective number
of nearby requested targets. A sample is covered when:

```text
n_eff >= al.min_neff
```

Coverage is the fraction of covered samples. Higher is better.

#### Surrogate error

For selected molecules with QM truth:

```text
surrogate_nmae = mean(abs(prediction - truth) / robust_property_scale)
surrogate_rmse = sqrt(mean(((prediction - truth) / robust_property_scale)^2))
```

Lower is better.

#### Uncertainty calibration

The ensemble uncertainty is compared with the QM error:

```text
calib_1sigma = fraction(error <= sigma)
calib_2sigma = fraction(error <= 2 * sigma)
```

For approximately Gaussian uncertainty, useful reference values are about
68% and 95%. Calibration is best when the observed values are close to those
references, not when they are simply maximized.

### Benchmark design

To test whether KDE and acquisition improve the system, compare identical
experiments with the same initial HDF5 data, seeds, generation budget, QM
budget, and iterations:

```text
trajectory + random acquisition
trajectory + score acquisition
kde_map + random acquisition
kde_map + score acquisition
```

Compare hit rate, corridor coverage, surrogate error, calibration, and
computational cost across multiple seeds.
