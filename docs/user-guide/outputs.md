# Outputs and analysis

## Output files

```text
Data/my_exp_0/
├── hdf5/
│   ├── <name>-input.hdf5            # copy of the initial dataset: KDE, QED/SA and dedup reference
│   └── <name>-output.hdf5           # cumulative dataset: initial molecules + every labelled molecule
├── diffusion_wdir/
│   ├── data/dataset_<k>/            # train/validation/test split of each dataset
│   └── model_registry/model_iter_<n>/    # checkpoints: best_mol_stab.ckpt, run0/last.ckpt
└── iter_<n>/
    ├── targets.json                 # planned targets and their metadata
    ├── anchor_candidates.csv        # every candidate: score components, eligibility, selection order
    ├── generated/t<id>/             # raw molecules per target
    ├── dedup_report.csv             # every molecule: kept, duplicate or in_initial
    ├── score_filtered/              # QED/SA scores (scores.tsv) and rejected molecules
    ├── relaxed/  prop_calc/         # optimized geometries and QM results
    ├── prop_filter_properties.csv   # requested vs. computed properties of every molecule
    ├── prop_filter_report.json      # hit rate per target, misses included
    ├── rejected/                    # removed molecules, reasons in reasons.json
    └── hdf5/output.hdf5             # this iteration's molecules, the next iteration's dataset

Experiments/my_exp_0/
├── al_state.json                    # planner state
├── density_reference.npy            # cached KDE reference densities of the initial dataset
└── al_metrics.csv                   # hit rate and coverage per iteration (with the evaluate block)
```

In the cumulative HDF5, each generated molecule carries these fields next to its properties, so every molecule can be traced back to the request that produced it:

| Field | Meaning |
|---|---|
| `al_iteration` | Iteration that generated the molecule |
| `al_target_id` | Target it was generated for (0 = the anchor) |
| `al_requested` | The target's property values |
| `al_hit` | 1 if it hit its target, 0 if not (see the [hit rule](kde-planning.md#closing-the-loop-the-hit-rule)) |

## Where to look first

| Question | File |
|---|---|
| What was requested this iteration? | `iter_<n>/targets.json` |
| Why were these targets chosen? | `iter_<n>/anchor_candidates.csv` |
| How many molecules survived each stage? | `dedup_report.csv`, `score_filtered/`, `rejected/reasons.json` |
| How many hit their target? | `iter_<n>/prop_filter_report.json` |
| How did the hit rate develop? | `Experiments/<experiment>/al_metrics.csv` (with `evaluate`) |

The Navigator's **Evaluation** tab shows all of these as charts, including a KDE map of targets and hits: see [Evaluation](../navigator/running-jobs.md#evaluation).

## Analysis notebooks

For Phase III, the notebooks in {repo}`analysis/` work on these outputs. Set the paths in each notebook's configuration cell first.

| Notebook | Use it to |
|---|---|
| {repo}`iteration_analysis.ipynb <analysis/iteration_analysis.ipynb>` | Follow SA-score distributions and model-evaluation metrics across iterations, and compare experiments |
| {repo}`augmented_data_analysis.ipynb <analysis/augmented_data_analysis.ipynb>` | Compare the property distributions of the initial and the augmented dataset (2D KDE) |
| {repo}`accuracy_analysis.ipynb <analysis/accuracy_analysis.ipynb>` | Compare predicted and reference properties for the models of successive iterations |
| {repo}`hdf5_analysis.ipynb <analysis/hdf5_analysis.ipynb>` | Inspect the structure and property distributions of any HDF5 dataset |
