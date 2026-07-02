# EQMgen: Efficient Quantum-Mechanical AI model GENerator

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This model uses the Diffusion model and Density Functional Theory (DFT) through active learning to navigate chemical space and improve our generative model for molecular design.


## Overview

The EQMgen framework combines diffusion models with quantum mechanical calculations to generate novel molecules with desired properties. The system leverages active learning to iteratively improve molecular generation by incorporating DFT feedback, enabling efficient exploration of chemical space.

**Key Features:**
- Molecular trajectory planning in property space
- Iterative active learning with DFT feedback
- HPC-optimized workflow orchestration
- Comprehensive analysis and visualization tools
- Property-guided molecular generation

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

<img width="400" height="400" alt="trajectory_planer" src="Navigator/example_trajectory.png" />

**Tools:** Interactive trajectory planner in `Navigator/design_trajectory.ipynb`

### Phase II – Active Diffusion Execution

This represents the core operational layer of the system, responsible for orchestrating the iterative molecular generation, evaluation, and feedback process. 


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

### Installation Steps

1. **Clone the repository:**

```bash
git clone <repository-url>
cd EQMgen
```

2. **Create the conda environment:**

```bash
mamba env create -f environment-al.yml -p conda_envs/env-activeloop/
```

Alternatively, if you prefer to manage the environment separately:

```bash
mamba env create -f environment-al.yml -n env-activeloop
```

3. **Activate the environment:**

```bash
conda activate ./conda_envs/env-activeloop/
# or if using named environment:
# conda activate env-activeloop
```

4. **Replace the installed ASE DFTB calculator file with the repository patch:**

```bash
ASE_DFTB="EQMgen/conda_envs/env-activeloop/lib/python3.10/site-packages/ase/calculators/dftb.py"
cp "$ASE_DFTB" "${ASE_DFTB}.bak"
cp EQMgen/dftb.py "$ASE_DFTB"
```

> **Note:** Run this from the directory where you cloned the repo (the parent of `EQMgen/`). The original ASE file is backed up as `dftb.py.bak` in the same location.

5. **Export the SKfiles path:**

```bash
export DFTB_PREFIX="$PWD/EQMgen/configs/SKfiles/3ob-3-1/"
```

6. **Ensure SKfiles are available under the code config directory:**

```bash
test -d "$CONFIG/SKfiles/3ob-3-1" && echo "SKfiles found under $CONFIG"
```

If SKfiles are stored elsewhere, copy them into `$CONFIG/SKfiles/` before running forward/backward jobs.

> **Note:** `environment-al.yml` produces a single combined environment (`env-activeloop`) that contains both the GPU/diffusion stack (PyTorch, PyTorch Geometric, MACE) and the chemistry/DFT stack (ASE, RDKit, xTB, DFTB+). Both the forward and backward job scripts activate this same environment.

> **Troubleshooting:** If you encounter `ModuleNotFoundError: No module named 'pkg_resources'` when running the forward pipeline, install `setuptools` into the environment: `conda activate ./conda_envs/env-activeloop/ && conda install setuptools`. This package is already included in `environment-al.yml` as of this version.

## Quick Start

1. **Initialize an experiment**

```bash
python EQMgen/scripts/init_experiment.py --name my_exp --dataset-path /path/to/base_dataset.hdf5
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

The EQMgen workflow is designed to run on HPC clusters using SLURM. It is split into two complementary job types that target different hardware. When you initialize an experiment, these scripts are copied from `Experiments/templates/jobs/` into your experiment directory — **you must customize them for your specific cluster before running experiments.**

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

> **Tip:** Both scripts activate the same combined environment. If you installed it under a different name or location, update the `conda activate $REPO_ROOT/conda_envs/env-activeloop` line in each script accordingly.

## Project Structure

```
EQMgen/
├── Data/                            # Experiment data root (created/updated at runtime)
├── EQMgen/
│   ├── configs/                     # Framework configs (eqdiff_model.yaml, eqmgen_paths.yaml, SKfiles)
│   ├── eqdiff/                      # Diffusion model code
│   ├── pipeline_blocks/             # Geometry/props/filter/hdf5 blocks
│   ├── scripts/                     # init_experiment.py, forward.py, backward.py
│   └── utils/                       # Data manager, trajectory, parallelization helpers
├── Experiments/
│   └── templates/                   # Template config and job scripts for initialization
├── Navigator/                       # Interactive trajectory planner (design_trajectory.ipynb)
├── environment-al.yml               # Combined conda environment specification
├── global_config.yaml               # Root/shared config imported by experiments
└── README.md
```

## Requirements

All dependencies are installed via `environment-al.yml`. The core stack includes:

**Deep learning / diffusion model**
- Python 3.10
- PyTorch 2.0.1
- PyTorch Geometric 2.3.0 (with `pytorch-scatter`, `pytorch-sparse`, `pytorch-cluster`)
- PyTorch Lightning 2.0.6
- `torch_ema`, TensorBoard
- MACE (machine-learning force field, from source)
- JAX (CUDA 12)

**Quantum chemistry / structure**
- ASE (Atomic Simulation Environment)
- FHI-aims (DFT, loaded as an HPC module — not via conda)
- DFTB+ 24.1 (`dftbplus`, `dftbplus-tools`, `dftbplus-python`)
- xTB (`xtb`, `xtb-python`)
- RDKit 2023.03.2
- Open Babel

**Configuration / data handling**
- Hydra / OmegaConf (experiment configuration)
- NumPy (<2), Pandas, h5py, LMDB

**Visualization / analysis**
- Matplotlib, Seaborn
- Jupyter / Notebook
- py3Dmol, nglview, RMSD, Biopython

See `environment-al.yml` for the complete, version-pinned list.

