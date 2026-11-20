# Active learning

Defaults: {repo}`src/configs/al.yaml`. The `al` section of the experiment configuration is merged over this file when a job starts, key by key inside nested sections such as `density`, so the experiment only needs the keys it changes.

In the tables, **`al.yaml`** is the default in that file and **Template** is the value in the experiment template; – means the template does not set the key, so the `al.yaml` value applies. Edit labels are explained in [Reading the reference tables](index.md#reading-the-reference-tables).

The template runs the KDE anchor planner: `query_strategy: kde_map`, `kde_region: anchor`, `anchor_score: relative_pct`. [KDE target planning](../user-guide/kde-planning.md) explains how it works.

:::{dropdown} Full file: `src/configs/al.yaml`
```{literalinclude} ../../src/configs/al.yaml
:language: yaml
```
:::

## Switch and strategy

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `enabled` | `false` | `true` | Turn active learning on: adds the `plan` and `dedup` blocks and labels every molecule against its own target | **Mandatory** decision |
| `query_strategy` | `kde_map` | `kde_map` | `kde_map` (KDE planning), `frontier` (advance along the trajectory by hit rate) or `trajectory` (waypoint *n* only) | Optional |
| `kde_region` | `corridor` | `anchor` | `kde_map` only: `anchor` samples a box around the current waypoint; `corridor` samples a padded band along the trajectory | Optional |
| `generation_budget_hours` | `12` | `6` | No further targets are started once generation has used this much wall time | Optional |

## Anchor mode

These keys apply with `kde_region: anchor`.

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `include_anchor` | `true` | `true` | Waypoint *n* itself is target 0 | Optional |
| `anchor_box` | `0.1` | `0.1` | Candidates are sampled within ± this fraction of each anchor value; for values near 0, the trajectory span on that axis is used | Optional |
| `anchor_n_candidates` | `2000` | `200` | Candidates sampled in the box; `null` uses `density.n_candidates` | Optional |
| `anchor_n_targets` | `3` | `2` | Additional targets per iteration; the anchor comes on top | Optional |
| `anchor_score` | `kde` | `relative_pct` | `relative_pct`: distance to the anchor plus density difference, with farthest-point selection. `kde`: the [adaptive score](#kde-and-adaptive-scoring), with repulsion | Optional |
| `anchor_min_datnum` | `1` | `1` | `relative_pct` only: minimum atom-count difference to the anchor. `anchor_box × atNUM` must allow it, or no candidate qualifies | Optional |
| `anchor_extra_fraction` | `null` | `0.2` | Molecules per additional target = round(fraction × `num_samples`), on top of `num_samples` at the anchor. `null` divides `num_samples` among all targets, by score with `kde` and equally with `relative_pct` | Optional |

## Corridor and frontier modes

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `targets_per_iter` | `4` | – | Corridor: targets planned per iteration | Optional |
| `corridor_pad` | `0.1` | – | Corridor: padding around the trajectory polyline, as a fraction of its span on each axis | Optional |
| `corridor_lookahead` | `null` | – | Corridor: `null` uses the whole trajectory; `k` uses the waypoints up to `current_iter + k` | Optional |
| `advance_hit_rate` | `0.3` | – | Frontier: hit rate a waypoint must reach before the next one is used | Optional |

## KDE and adaptive scoring

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `density.bandwidth` | `scott` | `scott` | KDE bandwidth in standardized units: `scott` (Scott's rule) or a number. Along `atNUM`, it is never narrower than one atom | Optional |
| `density.density_percentiles` | `null` | `[5.0, 15.0]` | Keep only candidates whose density under the KDE of the initial dataset lies between these percentiles of the initial dataset's own leave-one-out densities. Computed once and cached in `density_reference.npy`; `null` applies no band | Optional |
| `density.n_candidates` | `2000` | – | Corridor: candidates sampled per iteration; anchor mode uses it when `anchor_n_candidates` is `null` | Optional |
| `density.w_sparse` | `0.5` | – | Adaptive score: weight of data sparsity, the negative log density of all labelled molecules | Optional |
| `density.w_err` | `0.5` | – | Adaptive score: weight of the generator's past error near the candidate | Optional |
| `density.w_explore` | `0.5` | – | Adaptive score: weight of 1 / √(n_eff + 1), which favours rarely requested regions | Optional |
| `density.min_reach` | `0.05` | `0.0` | Drop candidates whose estimated hit rate is below this value once more than `min_neff` past requests lie nearby; `0` disables it | Optional |
| `density.logp_floor` | `-20.0` | – | Lower clip of the log density | Optional |
| `min_neff` | `5` | `5` | Kernel-weighted number of nearby past requests needed before a region can be judged unreachable; also the coverage threshold of `evaluate` | Optional |

## Hit rule and labelling

A molecule is a *hit* if, for every conditioning property, abs(achieved − target) ≤ max(relative × abs(target), absolute). The atom count is not part of the rule ([details](../user-guide/kde-planning.md#closing-the-loop-the-hit-rule)).

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `tolerances.default.relative` | `null` | `null` | Relative tolerance; `null` uses `query.filter_threshold` | Optional |
| `tolerances.default.absolute` | `0.0` | `0.0` | Absolute tolerance | Optional |
| `tolerances.<property>` | – | – | Per-property override: a number (relative tolerance) or `{relative: …, absolute: …}`. Tolerances must be non-negative | Optional |
| `keep_off_target` | `true` | `true` | Without the `prop_filter` block, keep misses in the HDF5 files with `al_hit = 0`; `false` drops them. With `prop_filter`, misses are always dropped | Optional |

## Deduplication

Deduplication runs at the end of the forward job, right after `generate`.

| Parameter | `al.yaml` | Template | Explanation | Edit |
|---|---|---|---|---|
| `dedup.enabled` | `false` | `true` | Remove molecules with identical connectivity | Optional |
| `dedup.scope` | `all` | `all` | `all`: one copy across all targets, and the lowest target id wins. `target`: one copy within each target | Optional |
| `dedup.against_initial` | `true` | `true` | Also remove molecules whose connectivity occurs in the initial dataset; those connectivities are cached next to the input HDF5 | Optional |
| `dedup.cov_factor` | `1.3` | – | Two atoms are bonded if their distance is below `cov_factor` × the sum of their covalent radii | Optional |
