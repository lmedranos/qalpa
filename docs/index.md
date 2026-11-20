# QALPA

**Quantum-Aware Learning for Property-Space Augmentation**

Steer a property-conditioned diffusion model through chemical space with quantum-mechanical feedback and active learning planned by kernel density estimation.

```{image} ../img/qalpa-workflow.png
:class: only-light
:width: 100%
:alt: QALPA workflow. Forward job on a GPU: (1) fine-tune the EQDIFF diffusion model on a weighted replay of all data; (2) plan targets with KDE around the current trajectory waypoint; (3) generate molecules at every target and remove duplicates. Backward job on CPUs: (4) screen QED and SA score against the initial dataset; (5) relax geometries with EquiDTB, DFTB+ or FHI-aims; (6) compute QM properties; (7) keep the molecules that hit their own target. The hits are added to the property space, and the loop repeats.
```

```{image} ../img/qalpa-workflow-dark.png
:class: only-dark
:width: 100%
:alt: QALPA workflow (dark version of the figure above).
```

## About QALPA

A conditional generative model is most reliable where its training data is dense. QALPA widens that region step by step. You draw a **trajectory** through a quantum-mechanical (QM) property space, for example HOMO–LUMO gap, many-body dispersion (MBD) energy and molecule size, on top of an initial dataset. QALPA then walks the trajectory one iteration at a time: an equivariant diffusion model (EQDIFF) generates molecules at property targets, QM calculations verify them, and the verified molecules join the training data before the model is fine-tuned for the next step.

The targets of every iteration are planned with **kernel density estimation (KDE)**. Besides the current waypoint, QALPA places additional targets nearby where the initial dataset is sparse and spreads them apart, so each iteration adds the data the model lacks most. Every labelled molecule records the target it was generated for and whether it hit it.

| Feature | What QALPA provides |
|---|---|
| **Property-guided generation** | 3D molecules from an equivariant diffusion model conditioned on QM properties and atom count |
| **KDE-planned active learning** | Per-iteration targets in sparse regions around each waypoint, scored with a Gaussian KDE of the initial dataset and spread out by farthest-point selection |
| **QM labelling** | Geometry optimization and properties with EquiDTB (machine-learned), DFTB+, or FHI-aims (DFT) |
| **Quality control** | Connectivity-based deduplication, QED/SA screening against the initial dataset, per-target property-accuracy filter |
| **Continual learning** | Fine-tuning on a weighted replay of the initial data and every iteration's new molecules |
| **HPC-ready and traceable** | Split GPU/CPU SLURM jobs, Hydra configuration, restart from any block; targets, hits and reports stored per iteration |

## Where to start

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: getting-started/installation
:link-type: doc

Create the GPU and CPU environments on a SLURM cluster.
:::

:::{grid-item-card} {octicon}`rocket` Quick start
:link: getting-started/quickstart
:link-type: doc

From an initial dataset to the first iteration, in six steps.
:::

:::{grid-item-card} {octicon}`git-branch` How it works
:link: user-guide/how-it-works
:link-type: doc

The three phases, the active-learning loop and its pipeline blocks.
:::

:::{grid-item-card} {octicon}`gear` Configuration
:link: configuration/index
:link-type: doc

How the configuration is assembled, and every key with its default.
:::

:::{grid-item-card} {octicon}`browser` Navigator
:link: navigator/index
:link-type: doc

Browser app to design trajectories, edit the config and run the jobs.
:::

:::{grid-item-card} {octicon}`tools` Recipes
:link: configuration/recipes
:link-type: doc

Ready-made config snippets for common changes.
:::
::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:caption: User guide

user-guide/how-it-works
user-guide/kde-planning
user-guide/trajectory
user-guide/running-on-slurm
user-guide/outputs
user-guide/resuming
user-guide/mlflow
```

```{toctree}
:hidden:
:caption: Configuration

configuration/index
configuration/shared
configuration/diffusion-model
configuration/experiment
configuration/active-learning
configuration/validation
configuration/recipes
```

```{toctree}
:hidden:
:caption: Navigator

navigator/index
navigator/using-the-app
navigator/running-jobs
navigator/troubleshooting
```

```{toctree}
:hidden:
:caption: Reference

reference/init-experiment
reference/repository-layout
reference/citation
reference/building-docs
```
