# Designing a trajectory

A trajectory is a list of waypoints through property space, stored as a CSV file. Each iteration *n* of the loop is anchored at waypoint *n*. You draw it on a map of the initial dataset, in one of three ways:

| Tool | Runs | Best for |
|---|---|---|
| [Navigator app](../navigator/index.md) | In your browser, with a small server on the cluster | The whole workflow: create the experiment, draw, save straight into it, edit the config, run the jobs |
| {repo}`Navigator/design_trajectory.ipynb` | In Jupyter | Drawing a trajectory inside a notebook |
| {repo}`Navigator/navigator.html` | Double-click, no server | Drawing on a dataset copied to your own computer |

## In the Navigator app

Start the server on the cluster and open the link it prints:

```bash
python Navigator/navigator_server.py      # any QALPA conda environment
```

The app walks you through **1 Experiment** (create or open), **2 Map**, **3 Waypoints** and **4 Save to experiment**, which writes the trajectory into `Experiments/<experiment>/configs/` and sets `EQdiffParam.properties_list` in that experiment's config. Setup and every step are described in the [Navigator](../navigator/index.md) section.

## In the Jupyter notebook

Open {repo}`Navigator/design_trajectory.ipynb` and set `file_path` to your initial dataset. Choose the x, y and colour properties, for example `HLgap`, `eMBD` and `atNUM`, and press **Show Plot**. Left-click the waypoints from start to end (**Undo** and **Clear** correct mistakes). Each click snaps to the nearest hexagon and records its mean colour value, so with `atNUM` as the colour every waypoint also gets an atom count. **Save Plot + CSV** writes `trajectory_<x>_<y>_<colour>.csv`.

Copy the file into the experiment:

```bash
cp trajectory_HLgap_eMBD_atNUM.csv Experiments/my_exp_0/configs/
```

(trajectory-csv)=
## The trajectory file

Example, {repo}`Navigator/trajectory_HLgap_eMBD_atNUM.csv`:

```{literalinclude} ../../Navigator/trajectory_HLgap_eMBD_atNUM.csv
:language: text
```

- Each row is one waypoint.
- The file needs one column per conditioning property, named as in `EQdiffParam.properties_list`.
- An `atNUM` column holds the atom count. The template's `relative_pct` planner requires it.
- Score columns such as `SAScore` are read only with `query.score_reference: trajectory`.
- With active learning, iterations beyond the last row reuse it; without it, provide one row per iteration.

The pipeline reads `query.trajectory_csv`, which by default matches `Experiments/<experiment>/configs/trajectory_*.csv`. Keep exactly one such file there: if several match, the first in alphabetical order is used. The Navigator's **Save to experiment** renames older ones to `.bak` for this reason.
