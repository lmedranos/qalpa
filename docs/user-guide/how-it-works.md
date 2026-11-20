# How it works

QALPA runs in three phases.

| Phase | What you do | Tool |
|---|---|---|
| **I · Trajectory design** | Draw waypoints through property space on top of the initial dataset and export them as `trajectory_*.csv` | [Navigator](../navigator/index.md) or {repo}`Navigator/design_trajectory.ipynb` |
| **II · Active-learning loop** | Run one forward (GPU) and one backward (CPU) job per iteration | {repo}`forward.py <src/scripts/forward.py>`, {repo}`backward.py <src/scripts/backward.py>` |
| **III · Analysis** | Inspect targets, hit rates, molecules and training curves | `Data/<experiment>/`, {repo}`analysis/`, MLflow, Navigator |

```{image} ../../img/navigator-trajectory.png
:class: only-light
:width: 520px
:align: center
:alt: Navigator, the example trajectory: five orange waypoints from q = 0 to q = 4, drawn on the HLgap–eMBD map of the initial dataset; hexagons are coloured by mean atom count, light blue for small and dark blue for large molecules.
```

```{image} ../../img/navigator-trajectory-dark.png
:class: only-dark
:width: 520px
:align: center
:alt: Navigator, the example trajectory (dark version of the figure above).
```

<p align="center"><sub>Phase I: waypoints drawn in the Navigator on the HLgap–eMBD map of the initial dataset, coloured by atom count.</sub></p>

(active-learning-loop)=
## The active-learning loop

Each iteration *n* is one **forward** job on a GPU followed by one **backward** job on CPUs. Orange blocks belong to active learning (`plan`, `dedup`) or change behaviour with it (`prop_filter`).

```{mermaid}
flowchart LR
    INIT[("Initial dataset<br/>HDF5")]:::data
    TRAJ[/"Trajectory<br/>trajectory_*.csv"/]:::data

    subgraph FWD["Forward job · GPU"]
        direction TB
        S["<b>split</b><br/>last iteration's molecules<br/>become dataset_n"]:::step
        T["<b>train</b><br/>fine-tune EQDIFF on a<br/>weighted replay of all datasets"]:::step
        P["<b>plan</b><br/>KDE target planning<br/>→ targets.json"]:::al
        G["<b>generate</b><br/>molecules at every target"]:::step
        D["<b>dedup</b><br/>unique connectivity,<br/>not in the initial dataset"]:::al
        S --> T --> P --> G --> D
    end

    subgraph BWD["Backward job · CPU"]
        direction TB
        SF["<b>score_filter</b><br/>QED and SA vs.<br/>the initial dataset"]:::step
        R["<b>relax_geo</b><br/>EquiDTB · DFTB+ · FHI-aims"]:::step
        Q["<b>prop_calc</b><br/>QM properties"]:::step
        PF["<b>prop_filter</b><br/>hit or miss vs. the<br/>molecule's own target"]:::al
        H["<b>hdf5</b><br/>append molecules<br/>with target and hit flag"]:::step
        SF --> R --> Q --> PF --> H
    end

    OUTDS[("Augmented dataset<br/>cumulative HDF5")]:::data

    INIT --> FWD
    TRAJ --> FWD
    FWD -->|"generated molecules"| BWD
    BWD -.->|"next iteration"| FWD
    BWD --> OUTDS

    classDef step fill:#FFFFFF,stroke:#78909C,color:#212121
    classDef al fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#212121
    classDef data fill:#F3E5F5,stroke:#8E24AA,color:#212121
    style FWD fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    style BWD fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    linkStyle default stroke:#607D8B,stroke-width:1.5px
```

(pipeline-blocks)=
## Pipeline blocks

| # | Block | Job | What happens | Written to `Data/<experiment>/` |
|:-:|---|:-:|---|---|
| 1 | `split` | Forward | The molecules labelled in iteration *n − 1* become dataset `dataset_n` (iteration 0: the initial dataset) and are split into train, validation and test sets | `diffusion_wdir/data/dataset_<n>/` |
| 2 | `train` | Forward | EQDIFF is fine-tuned from the previous checkpoint on a weighted replay of all datasets | `diffusion_wdir/model_registry/model_iter_<n>/` |
| 3 | `plan` | Forward | The KDE planner chooses this iteration's targets ([details](kde-planning.md)) | `iter_<n>/targets.json` |
| 4 | `generate` | Forward | Molecules are generated at every target, conditioned on its properties and atom count | `iter_<n>/generated/t<id>/` |
| 5 | `dedup` | Forward | Molecules with identical connectivity, or already in the initial dataset, are removed | `iter_<n>/dedup_report.csv` |
| 6 | `score_filter` | Backward | Molecules are screened against the initial dataset's mean QED and SA score (`query.score_relax` sets the margin) | `iter_<n>/score_filtered/` |
| 7 | `relax_geo` | Backward | Geometries are optimized with EquiDTB, DFTB+ or FHI-aims | `iter_<n>/relaxed/` |
| 8 | `prop_calc` | Backward | QM properties are computed | `iter_<n>/prop_calc/` |
| 9 | `prop_filter` | Backward | Each molecule is labelled *hit* or *miss* against **the target it was generated for**; misses are dropped | `iter_<n>/prop_filter_report.json` |
| 10 | `hdf5` | Backward | Labelled molecules are appended, with their target and hit flag, to the cumulative dataset and saved as the next iteration's dataset | `hdf5/<name>-output.hdf5`, `iter_<n>/hdf5/` |

At the end of the backward job, `loop.current_iter` is advanced in `experiment_config.yaml`, so the next forward job continues with iteration *n + 1*. Add `evaluate` after `hdf5` in `pipeline.blocks` to also log each iteration's hit rate and coverage to `Experiments/<experiment>/al_metrics.csv`.

When each block runs, and which settings switch it on or off, is listed under [Loop and pipeline](../configuration/experiment.md#loop-and-pipeline).

## Training on a weighted replay

With `EQdiffParam.concat_datasets: true` (template), every iteration keeps its molecules as a separate dataset, and training draws from all of them. The newest dataset gets a share of 1 − `alpha`; the older ones share `alpha`, with the initial dataset weighted `beta` relative to each intermediate one. `min_original_share` keeps a floor under the initial dataset's share to limit forgetting. The exact formula is in [Generative model](../configuration/experiment.md#generative-model).
