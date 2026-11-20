# Using the app

Steps 1 to 5 prepare an experiment: create it, draw the trajectory, save it, and review the configuration. [Running jobs](running-jobs.md) covers step 6.

When you open the app, the **Welcome** tab shows the QALPA workflow and the six steps, with a link to this documentation. The steps themselves are in the panel on the left. Pressing **Show map** switches to the **Map** tab.

## Step 1: Experiment

- **New:** enter a name, pick the initial dataset (`.hdf5`) with **Browse…** and press **Create experiment**. This runs `init_experiment.py` and creates `Experiments/<name>_<n>/`. The log is shown when it finishes.
- **Existing:** choose an experiment from the list and press **Use this experiment**. **Refresh** reloads the list.

The experiment's dataset is selected for the map automatically.

## Step 2: Map

1. Choose the properties for the **x axis**, **y axis** and **colour**.
   - An axis can be **none**: it is left out of the trajectory.
   - An axis can be **constant**: pick the property name and type its value, and every waypoint gets that value.
2. Set **Molecules to read**. The default is 1000; 0 reads all of them.
3. Press **Show map**.

The progress bar shows how many molecules have been read, out of the total in the file. To read more, raise the number and press **Show map** again.

## Step 3: Waypoints

Click anywhere on the map to add waypoints, from the start to the end of the trajectory. The colour value of each waypoint is taken from the nearest molecules.

- **Undo** removes the last waypoint.
- **Clear** removes all of them.

With `atNUM` as the colour, every waypoint also gets an atom count, which the template's KDE planner needs ([The trajectory file](../user-guide/trajectory.md#the-trajectory-file)).

## Step 4: Export

- **Save to experiment** writes `trajectory_<x>_<y>_<colour>.csv` into `Experiments/<exp>/configs/`. It also sets `EQdiffParam.properties_list` in the experiment's config. Older `trajectory_*.csv` files there are renamed to `.bak`, because the pipeline uses the first trajectory file it finds.
- **Save as…**, **Download CSV** and **Download PNG** save copies elsewhere.

## Step 5: Configuration

**Edit config** opens `Experiments/<exp>/configs/experiment_config.yaml` in the **Config** tab.

- **Save** (or Ctrl+S) checks the YAML syntax before writing. The previous version is kept as `experiment_config.yaml.bak`.
- **Reload** discards your unsaved edits.

```{warning}
Avoid editing the config while a backward job is running: that job rewrites the file when it finishes ({ref}`write-back <write-back>`), and the page shows a warning meanwhile.
```

Every key is described in the [Configuration](../configuration/index.md) section.
