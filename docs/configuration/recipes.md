# Recipes

All snippets go into `Experiments/<experiment>/configs/experiment_config.yaml`.

## Compute properties with DFT (FHI-aims)

```yaml
properties:
  calculator: FHI-aims
system:
  fih_aims_exc: /path/to/aims.x   # overrides global_config.yaml for this experiment
```

Also load the FHI-aims modules in the backward job script ([Running on SLURM](../user-guide/running-on-slurm.md#adapt-the-job-scripts)). The control settings come from `src/configs/nctr1.in`.

## Condition on other properties

```yaml
EQdiffParam:
  properties_list: ['HLgap', 'DIP']
```

Set this before iteration 0. The initial dataset must contain these properties, and the trajectory CSV needs them as columns, plus `atNUM`; `query.props` follows automatically.

## Plan more and wider KDE targets

```yaml
al:
  anchor_box: 0.2
  anchor_n_candidates: 500
  anchor_n_targets: 4
  anchor_extra_fraction: 0.25
```

## Tighten the hit rule for one property

```yaml
al:
  tolerances:
    HLgap: 0.05                            # relative tolerance only
    eMBD: {relative: 0.1, absolute: 0.05}
```

## Keep off-target molecules

```yaml
pipeline:
  blocks: [split, train, plan, generate, dedup, score_filter, relax_geo, prop_calc, hdf5, evaluate]
al:
  keep_off_target: true
```

Misses are then stored with `al_hit = 0`. Their QM labels are still correct, so they also join the next training set. `evaluate` logs the hit rate per iteration.

## Explore the whole trajectory corridor

```yaml
al:
  kde_region: corridor
  targets_per_iter: 4
```

This needs at least two waypoints and uses the [adaptive score](../user-guide/kde-planning.md#adaptive-score). Until the molecules of a first iteration have been labelled, it requests the plain waypoint.

## Run without active learning

```yaml
al:
  enabled: false
```

The trajectory then needs one row per iteration, from 0 to `loop.num_iterations`.

## Start from a pretrained model

```yaml
EQdiffParam:
  load_ckpt_from_pretrained: /path/to/model.ckpt
```

## Keep experiment data on another filesystem

```yaml
Data_path: /scratch/<project>/qalpa-data   # top level; all data paths derive from it
```

Set it before the first job of the experiment.

## Screen more strictly before the QM steps

```yaml
query:
  scores: ["SAScore", "QED"]   # check both scores
  score_relax:
    QED: 0.1                   # tighter than the template's 0.3
    SAScore: 0.1
```

A molecule passes if QED > (1 − r) × mean QED and SA < (1 + r) × mean SA of the initial dataset. Smaller r keeps fewer, more drug-like molecules for the expensive QM steps.
