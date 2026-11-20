# Resuming and re-planning

`pipeline.start_at` restarts the current iteration (`loop.current_iter`) from any block; the blocks before it are skipped. An existing `targets.json` is always reused, and `split` keeps a dataset that is already prepared.

| Goal | Set in `experiment_config.yaml` | Then submit |
|---|---|---|
| Resume after a failed block | `pipeline.start_at: <failed block>` | The job that owns the block, then the remaining chain |
| Re-plan targets without retraining | Delete `iter_<n>/targets.json`; `pipeline.start_at: plan` | Forward, then backward |
| Regenerate for the same targets | `pipeline.start_at: generate` (reuses `targets.json`) | Forward, then backward |
| Redo the backward step without regenerating | `pipeline.start_at: dedup` (refills `query/` from the generated molecules) | Forward, then backward |

These restarts are meant for an iteration whose `hdf5` block has not run yet; after that, its molecules are already part of the cumulative dataset.

```{important}
`pipeline.start_at` is not reset automatically. Set it back to `split` once the restarted iteration has finished, or the next iteration will skip its first blocks.
```

## Restart one job without editing the config

To restart a single job, set the block for that job only, in either of two ways:

- In the Navigator, use **Start at** in step 6 Run (see [Restart from a block](../navigator/running-jobs.md#restart-from-a-block-start-at)).
- On the command line, set the environment variable `QALPA_START_AT`, for example:

  ```bash
  sbatch --export=ALL,QALPA_START_AT=prop_calc qalpa_backward.sh
  ```

The variable takes priority over `pipeline.start_at` for that job. The config is not changed, so the jobs after it run from `pipeline.start_at` (normally `split`) and nothing needs resetting. The job fails at once if the block is not in the experiment's `pipeline.blocks`.

## Cleaning a failed iteration

Before rerunning a stage, remove its partial outputs from `Data/<experiment>/iter_<n>/`. The Navigator's [Clean](../navigator/running-jobs.md#clean) tab groups the iteration's folders by pipeline stage and empties the ones you select.
