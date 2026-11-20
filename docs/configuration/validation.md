# Validation

These checks run when a job starts, or when the block that needs the setting runs. A failed check stops the job with an error that names the setting.

| Setting | Requirement |
|---|---|
| `loop.current_iter` | At most `loop.num_iterations` |
| `query.trajectory_csv` | A matching file exists and contains every `query.props` column (and, in `trajectory` mode, every `query.scores` column) |
| `query.props` | With active learning: not empty, and only `HLgap`, `eMBD`, `DIP`, `eH`, `eL` |
| `query.score_reference`, `query.score_relax` | `initial_mean` or `trajectory`; relax keys `QED` and `SAScore` with values in (−1, 1) |
| `query.scores` | In `initial_mean` mode: contains `SAScore`, `QED` or both |
| `EQdiffParam.context_mapping` | Needs a non-empty `properties_list` |
| `EQdiffParam.concat_datasets` | Needs `hdf5.hdf5_iter_flag: True`; from iteration 1, the previous backward job must have written `iter_<n−1>/hdf5/output.hdf5` |
| `EQdiffParam.alpha`, `beta`, `min_original_share` | `alpha` and `beta` in [0, 1]; `min_original_share` in [0, 1) and small enough to leave a share for the newest dataset |
| `hdf5.appending_flag` | `True` with active learning |
| `al.query_strategy`, `al.kde_region`, `al.anchor_score`, `al.dedup.scope` | One of the listed options |
| `al.kde_region: corridor` | At least two waypoints |
| `al.anchor_box` | Greater than 0 |
| `al.anchor_n_targets`, `al.targets_per_iter` | At least 0 with the anchor, otherwise at least 1; at most the number of candidates |
| Number of targets | At most `EQdiffParam.num_samples` (including the anchor), unless `anchor_extra_fraction` is set |
| `al.anchor_extra_fraction` | Greater than 0; needs `include_anchor: true`; round(fraction × `num_samples`) ≥ 1 |
| `al.anchor_score: relative_pct` | An `atNUM` column in the trajectory with `query.fix_num_nodes: true`; `anchor_min_datnum` ≥ 0 |
| `al.density.density_percentiles` | `[low, high]` with 0 ≤ low < high ≤ 100 |
| `al.tolerances` | Non-negative values |
| `QALPA_START_AT` | A block listed in `pipeline.blocks` |
