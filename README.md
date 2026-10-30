# QALPA - Quantum-Aware Learning for Property-Space Augmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

QALPA combines diffusion-based molecular generation with quantum-mechanical feedback through active learning to explore property space and improve generative models for molecular design. 

<p align="center">
<img width="600" alt="workflow" src="img/qalpa.png" />
</p>

## Overview

The QALPA framework combines diffusion models with quantum-mechanical (QM) calculations to generate novel molecules with desired properties. The system leverages active learning (AL) to iteratively improve molecular generation by incorporating quantum property data feedback, enabling efficient exploration of chemical space.

**Key Features:**
- Molecular trajectory planning in property space
- Iterative active learning with DFT and EquiDTB feedback
- HPC-optimized workflow orchestration
- Comprehensive analysis and visualization tools
- Property-guided molecular generation
- The QM packages currently implemented are FHI-aims and DFTB+.

## WorkFlow

The software system is organized into a modular architecture that follows a three-phase workflow, guiding the process from molecular trajectory design to data analysis. 

```mermaid
flowchart LR
    subgraph phase1["Phase I: Trajectory Planning"]
        A["Define Property Space<br/>(HOMO-LUMO, MBD, etc.)"]
        B["Interactive Planner"]
        C["Trajectory CSV"]
        A --> B --> C
    end
    
    subgraph phase2["Phase II: Active Diffusion"]
        T["Train Model"]
        D["Generate Molecules<br/>(Diffusion Model)"]
        G["Geometry Optimization"]
        R["Retrain Model"]
        S["Scoring"]
        F["Active Learning<br/>Selection"]
        E["DFT Calculation"]
        
        T --> D --> G --> S--> E --> F --> R --> D
    end
    
    subgraph phase3["Phase III:  Analysis"]
        H["HDF5 Output"]
        I["Accuracy Analysis"]
        J["Iteration Analysis"]
        K["Visualization"]
        H --> I
        H --> J
        I --> K
        J --> K
    end
    
    C ==> D
    C ==> F
    phase2 --> H
    
    style phase1 fill:#FFE082,stroke:#F57C00,stroke-width: 2px
    style phase2 fill:#81C784,stroke:#388E3C,stroke-width: 2px
    style phase3 fill:#90CAF9,stroke:#1976D2,stroke-width: 2px
```

### Phase I – Trajectory Planning

This phase provides an interactive environment for defining molecular trajectories within a property space (e.g., HOMO-LUMO gap, many-body dispersion energy, atomic number). The resulting trajectory files (`trajectory_*.csv`) serve as input for the active learning process. 

<img width="400" alt="trajectory_planer" src="Navigator/example_trajectory.png" />

**Tools:** Interactive trajectory planner in `Navigator/design_trajectory.ipynb`

### Phase II – Active Diffusion Execution

This represents the core operational layer of the system, responsible for orchestrating the iterative molecular generation, evaluation, and feedback process. 

#### Active learning (`al:` block in `experiment_config.yaml`)

With `al.enabled: true` the loop chooses what to evaluate instead of following one trajectory row per iteration:

1. **plan** – picks `targets_per_iter` property targets inside the trajectory corridor. `kde_map` favors sparse regions (KDE of labelled data), places where the generator missed its past targets, and under-explored targets. `trajectory` and `frontier` are simpler alternatives.
2. **generate** – one generator call per target, `num_samples * oversample` candidates in total.
3. **acquire** – a fingerprint random-forest ensemble predicts each candidate's properties and uncertainty; candidates are ranked by hit probability, uncertainty, novelty, diversity and cost, and only `budget` of them go on to relax and QM.
4. **hdf5** – every molecule with a valid QM result is kept (even if it missed its target) with its true properties, so the generator is retrained on what it actually produced. Set `keep_off_target: false` to keep hits only.
5. **evaluate** – appends hit rate, surrogate error, calibration and coverage to `<experiment>/metrics.csv`; optional early stopping sets `loop.stop`.

Run with `al.enabled: false` to get the original fixed-trajectory loop. `kde_map` needs at least two trajectory rows and `hdf5.appending_flag: true`. Forward and backward remain separate jobs, so submit them alternately as before. Use `al.acquisition.mode: random` for a baseline.


### Phase III – Output Analysis

The final phase focuses on analyzing and visualizing the output. Jupyter notebooks in the `Navigator/` directory allow users to inspect the active diffusion results, model performance, and generated molecular properties (HDF5 outputs).


## Active Diffusion Architecture

The architectural elements are organized as follows:

### Surface Layer – Experiment Orchestration Layer
Responsible for workflow coordination, configuration management, and resource allocation across HPC environments. It defines the experiment lifecycle.

### Deep Layer – Functional Core Layer 
Contains the domain-specific components that implement the scientific and computational functionality of the diffusion process, including molecular generation and property evaluation.

### Utilities Subsystem 
Provides cross-cutting services that support both layers by handling data management, task parallelization, and trajectory synchronization.


```mermaid
graph TB
    subgraph orchestration["🔷 Orchestration Layer"]
        hpc["HPC Resource Allocator"]
        config["Configuration"]
        orc["Orchestrator"]
        Expsc["Experimentation Scripts"]
        initsc["Initiation"]
        forsc["Forward"]
        bacsc["Backward"]
        Evalsc["Benchmarking Scripts"]
        gensc["Generate"]
        calsc["Calculate"]
        evalsc["Evaluate"]

    end

    Expsc --> initsc
    Expsc --> forsc
    Expsc --> bacsc
    Evalsc --> gensc
    Evalsc --> calsc
    Evalsc --> evalsc
    orc --> Expsc
    orc --> Evalsc
   

    subgraph utilities["🟢 Utilities Subsystem"]
        data["Data Manager"]
        parallel["Task Parallelizer"]
        sync["Trajectory Synchronizer"]
    end
    
    subgraph functional["🔴 Functional Core Layer"]
        Diff["Diffusion Model<br/>(Pytorch Geometry/PytorchLightning)"]
        train["Model Trainer"]
        gen["Molecular Generator"]
        Pipeb["Pipeline Blocks"]
        eval["Property Evaluator<br/>(DFT/FHI-aims)"]
        opt["Geometry Optimizer<br/>(MLFF)"]
        sco["Scoring Schemes Calculators<br/>(SAScore/QED)"]
        sel["Molecular Filters"]
        Aug["Dataset Augmentation"]
    end
    Diff --> train
    Diff --> gen
    Pipeb --> eval
    Pipeb --> opt
    Pipeb --> sco
    Pipeb --> sel
    Pipeb --> Aug
    
    style orchestration fill:#B3E5FC,stroke:#0277BD,stroke-width:2px,color:#000
    style utilities fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px,color:#000
    style functional fill:#FFCCBC,stroke:#D84315,stroke-width:2px,color:#000
    linkStyle default stroke:#424242,stroke-width:2px
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Mamba or Conda package manager
- Access to HPC cluster with SLURM
- CUDA-capable GPU (for forward pipeline)
- Multi-core CPU system (for backward pipeline)

### Installation Options

The project supports two environment configuration strategies:

| Approach | Environments | Use Case | Pros | Cons |
|----------|---|----------|------|------|
| **Split** | `env-qalpa-forward` + `env-qalpa-backward` | HPC clusters, production workflows (recommended) | Smaller per-node installations, hardware-specific optimization, better scalability | Requires managing two environments |
| **Combined** | Single `env-activeloop` | Development, single-machine testing | Simpler setup, unified environment | Larger download, slower installation |

**Recommended:** Use **Split** environments for production workflows and HPC clusters. 

### Installation Steps

#### Default option: Split Environments (Recommended) ⭐

For production workflows on HPC systems and general use, use separate forward (GPU) and backward (CPU) environments:

For production workflows on HPC systems, use separate forward (GPU) and backward (CPU) environments:

1. **Clone the repository:**

```bash
git clone <repository-url>
cd qapla
```

2. **Create the forward environment (GPU cluster):**

```bash
mamba env create -f environment_f.yml -p conda_envs/env-qalpa-forward/
```

3. **Create the backward environment (CPU cluster):**

```bash
mamba env create -f environment_b.yml -p conda_envs/env-qalpa-backward/
```

4. **Patch the backward environment's ASE DFTB calculator:**

```bash
ASE_DFTB="./conda_envs/env-qalpa-backward/lib/python3.10/site-packages/ase/calculators/dftb.py"
cp "$ASE_DFTB" "${ASE_DFTB}.bak"
cp src/dftb.py "$ASE_DFTB"
```

5. **Export the SKfiles path (for both environments):**

```bash
export DFTB_PREFIX="$PWD/src/configs/SKfiles/3ob-3-1/"
```

6. **Update your job scripts to use split environments:**

Edit `Experiments/templates/jobs/eqmgen_forward.sh`:
```bash
conda activate $REPO_ROOT/conda_envs/env-qalpa-forward/
```

Edit `Experiments/templates/jobs/eqmgen_backward.sh`:
```bash
conda activate $REPO_ROOT/conda_envs/env-qalpa-backward/
```

> **Note:** Three environment configurations are available:
> - `environment-al.yml`: Single combined environment with GPU/diffusion and chemistry/DFT stacks
> - `environment_f.yml`: GPU-optimized forward pipeline (PyTorch, PyTorch Geometric, MACE)
> - `environment_b.yml`: CPU-optimized backward pipeline (ASE, RDKit, xTB, DFTB+)

## Quick Start

1. **Initialize an experiment**

```bash
python src/scripts/init_experiment.py --name my_exp --dataset-path /path/to/base_dataset.hdf5
```

This creates the experiment directory with template job scripts:

```
Experiments/my_exp_0/
├── configs/
│   └── experiment_config.yaml
├── jobs/
│   ├── eqmgen_forward.sh      # runs on the GPU cluster
│   └── eqmgen_backward.sh     # runs on the heavy CPU cluster
├── logs/
└── results/
```

2. **Edit the experiment config**

Update `Experiments/my_exp_0/configs/experiment_config.yaml`.

3. **Customize the SLURM job scripts for your cluster**

The copied job scripts must be adapted to your cluster before submitting. See [HPC Cluster Configuration (SLURM)](#hpc-cluster-configuration-slurm) below.

4. **Copy Navigator trajectory CSV files**

Copy the trajectory CSV files into `Experiments/my_exp_0/configs/`.

5. **Submit the jobs**

```bash
cd Experiments/my_exp_0/jobs
sbatch eqmgen_forward.sh
sbatch eqmgen_backward.sh
```

6. **Monitor job status**

```bash
squeue -u $USER
squeue -j <job_id>          # check a specific job
```

7. **Analyze results**

Generated iteration data and HDF5 outputs are written under `Data/my_exp_0/`.

## HPC Cluster Configuration (SLURM)

The QAPLA workflow is designed to run on HPC clusters using SLURM. It is split into two complementary job types that target different hardware. When you initialize an experiment, these scripts are copied from `Experiments/templates/jobs/` into your experiment directory — **you must customize them for your specific cluster before running experiments.**

### Forward Pipeline (GPU Cluster)

**Script:** `Experiments/templates/jobs/eqmgen_forward.sh`

The forward pipeline runs **on GPU clusters** and handles:
- Diffusion model training
- Molecular generation
- Model inference

**Default SLURM Configuration:**
- **Partition:** `capella` (GPU partition)
- **GPU:** 1 GPU per job
- **CPU:** 6 cores per task
- **Memory:** 8000 MB per CPU
- **Nodes:** 1
- **Time:** 20 hours

**Update cluster parameters before running:**

Edit `Experiments/templates/jobs/eqmgen_forward.sh` to match your GPU cluster specifications:

```bash
#SBATCH --partition=your_gpu_partition    # Change to your GPU partition name
#SBATCH --gres=gpu:X                       # Adjust number of GPUs (1, 2, 4, etc.)
#SBATCH --cpus-per-task=6                  # Adjust CPU cores
#SBATCH --mem-per-cpu=8000                 # Adjust memory per CPU
#SBATCH --time=HH:MM:SS                    # Adjust time limit
```

### Backward Pipeline (Heavy CPU Cluster)

**Script:** `Experiments/templates/jobs/eqmgen_backward.sh`

The backward pipeline runs **on heavy CPU clusters** and handles:
- DFT calculations (FHI-aims)
- Geometry optimization
- Property evaluation
- DFTB+ calculations

**Default SLURM Configuration:**
- **Partition:** `barnard` (CPU partition)
- **CPU:** 32 tasks per node
- **Nodes:** 10 (distributed computing)
- **Memory:** 12000 MB per CPU
- **Account:** (set your SLURM project account)
- **Time:** 5 hours
- **Module dependencies:** FHI-aims, Intel compiler, MPI

**Update cluster parameters before running:**

Edit `Experiments/templates/jobs/eqmgen_backward.sh` to match your CPU cluster specifications:

```bash
#SBATCH -A your_account              # Change to your SLURM account
#SBATCH --partition=your_cpu_partition
#SBATCH --cpus-per-task=X            # Adjust CPUs per task
#SBATCH --mem-per-cpu=XXXX           # Adjust memory per CPU
#SBATCH --nodes=X                    # Adjust number of nodes
#SBATCH --ntasks-per-node=X          # Adjust tasks per node
#SBATCH --time=HH:MM:SS              # Adjust time limit
```

Additionally, update the module load commands if your cluster has different module names:

```bash
# Original (example):
module load release/24.04 intel-compilers/2023.2.1 impi/2021.10.0
module load FHI-aims/240507

# Update to your cluster's available modules:
module load <release> <compiler> <mpi>
module load <fhi-aims-module>
```

### Switching Between Environment Strategies

By default, the job script templates use **split environments** (`env-qalpa-forward` and `env-qalpa-backward`). To switch strategies:

**To use the combined environment:**

Edit both `Experiments/templates/jobs/eqmgen_forward.sh` and `Experiments/templates/jobs/eqmgen_backward.sh`:

```bash
# Replace:
conda activate $REPO_ROOT/conda_envs/env-qalpa-forward/
conda activate $REPO_ROOT/conda_envs/env-qalpa-backward/

# With:
conda activate $REPO_ROOT/conda_envs/env-activeloop/
```

**To use split environments** (default):

No changes needed. The templates already reference:
- Forward: `$REPO_ROOT/conda_envs/env-qalpa-forward/`
- Backward: `$REPO_ROOT/conda_envs/env-qalpa-backward/`

> **Recommendation:** Use split environments for production HPC workflows to optimize hardware utilization and reduce environment sizes on each node.

## Project Structure

```
qapla/
├── Data/                            # Experiment data root (created/updated at runtime)
├── src/
│   ├── configs/                     # Framework configs (eqdiff_model.yaml, eqmgen_paths.yaml, SKfiles)
│   ├── eqdiff/                      # Diffusion model code
│   ├── pipeline_blocks/             # Geometry/props/filter/hdf5 blocks
│   ├── scripts/                     # init_experiment.py, forward.py, backward.py
│   └── utils/                       # Data manager, trajectory, parallelization helpers
├── Experiments/
│   └── templates/                   # Template config and job scripts for initialization
├── Navigator/                       # Interactive trajectory planner (design_trajectory.ipynb)
├── environment-al.yml               # Combined conda environment specification
├── environment_f.yml                # Forward (GPU) pipeline environment
├── environment_b.yml                # Backward (CPU) pipeline environment
├── global_config.yaml               # Root/shared config imported by experiments
└── README.md
```

## Requirements

All dependencies are installed via one of the environment specification files:

**Environment Options:**
- `environment-al.yml` – Combined environment with GPU/diffusion and chemistry/DFT stacks (recommended for single-machine setup)
- `environment_f.yml` – Forward pipeline (GPU) – PyTorch, PyTorch Geometric, MACE
- `environment_b.yml` – Backward pipeline (CPU) – ASE, RDKit, xTB, DFTB+

**Core Stack Components:**

**Deep learning / diffusion model** (in `environment-al.yml` and `environment_f.yml`)
- Python 3.10
- PyTorch 2.0.1
- PyTorch Geometric 2.3.0 (with `pytorch-scatter`, `pytorch-sparse`, `pytorch-cluster`)
- PyTorch Lightning 2.0.6
- `torch_ema`, TensorBoard
- MACE (machine-learning force field, from source)
- JAX (CUDA 12)

**Quantum chemistry / structure** (in `environment-al.yml` and `environment_b.yml`)
- ASE (Atomic Simulation Environment)
- FHI-aims (DFT, loaded as an HPC module — not via conda)
- DFTB+ 24.1 (`dftbplus`, `dftbplus-tools`, `dftbplus-python`)
- xTB (`xtb`, `xtb-python`)
- RDKit 2023.03.2
- Open Babel

**Configuration / data handling** (in all environments)
- Hydra / OmegaConf (experiment configuration)
- NumPy (<2), Pandas, h5py, LMDB

**Visualization / analysis** (in all environments)
- Matplotlib, Seaborn
- Jupyter / Notebook
- py3Dmol, nglview, RMSD, Biopython

See the environment YAML files for the complete, version-pinned lists.

## Troubleshooting

### `ModuleNotFoundError: Neither tensorboard nor tensorboardX is available`

`tensorboard` is declared in the environment files but may be missing if the environment was created from an older version. Fix it with:

**For combined environment:**
```bash
conda_envs/env-activeloop/bin/pip3 install tensorboard
```

**For split environments:**
```bash
conda_envs/env-qalpa-forward/bin/pip3 install tensorboard
conda_envs/env-qalpa-backward/bin/pip3 install tensorboard
```

Then re-submit the job.

### `ModuleNotFoundError: No module named 'pkg_resources'`

`setuptools>=70` removed `pkg_resources` as a top-level module, which breaks `pytorch-lightning` and several other packages. The environment files pin `setuptools<70` to prevent this. If your environment was created without this pin, fix it with:

**For combined environment:**
```bash
conda_envs/env-activeloop/bin/python -m pip install "setuptools<70" --force-reinstall
```

**For split environments:**
```bash
conda_envs/env-qalpa-forward/bin/python -m pip install "setuptools<70" --force-reinstall
conda_envs/env-qalpa-backward/bin/python -m pip install "setuptools<70" --force-reinstall
```

## Citations
If you use parts of the code please cite
```
@article{qalpa,
title={QALPA: Property-guided diffusion modeling for efficient exploration of chemical spaces of flexible molecules}, 
author={Michael Hanna and Julian Cremer and Zekiye Erarslan and Leonardo Medrano Sandonas},
year={2026},
journal={arXiv preprint arXiv:2609.16527},
url={https://arxiv.org/abs/2609.16527}, 
}
```
