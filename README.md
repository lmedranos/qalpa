# QM Active Learning Diffusion for Molecular Design

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This model uses the Diffusion model and Density Functional Theory (DFT) through active learning to navigate chemical space and improve our generative model for molecular design.


## Overview

The QMActiveDiffusion framework combines diffusion models with quantum mechanical calculations to generate novel molecules with desired properties. The system leverages active learning to iteratively improve molecular generation by incorporating DFT feedback, enabling efficient exploration of chemical space.

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
        T["Train Model<br/>(PytorchLightning)"]
        D["Generate Molecules<br/>(Diffusion Model)"]
        G["Geometry Optimization<br/>(MLFF-MACE)"]
        R["Retrain Model"]
        S["Scoring<br/>(SAScore/QED)"]
        F["Active Learning<br/>Selection"]
        E["DFT Calculation<br/>(FHI-aims)"]
        
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
    
    C ==> T
    phase2 --> H
    
    style phase1 fill:#FFE082,stroke:#F57C00,stroke-width: 2px
    style phase2 fill:#81C784,stroke:#388E3C,stroke-width: 2px
    style phase3 fill:#90CAF9,stroke:#1976D2,stroke-width: 2px
```

### Phase I – Trajectory Planning

This phase provides an interactive environment for defining molecular trajectories within a property space (e.g., HOMO-LUMO gap, many-body dispersion energy, atomic number). The resulting trajectory files (`trajectory_*.csv`) serve as input for the active learning process. 

<img width="200" height="200" alt="trajectory_planer" src="https://github.com/user-attachments/assets/a827726d-cfd1-411e-80ce-79d87f5f7dbe" />

**Tools:** Interactive trajectory planner in the `trajectory/` directory

### Phase II – Active Diffusion Execution

This represents the core operational layer of the system, responsible for orchestrating the iterative molecular generation, evaluation, and feedback process. 


### Phase III – Output Analysis

The final phase focuses on analyzing and visualizing the output. Jupyter notebooks (`accuracy_analysis.ipynb`, `iteration_analysis.ipynb`, `hdf5_analysis.ipynb`) allow users to inspect the active diffusion results, model performance, and generated molecular properties.

<img width="400" height="400" alt="embd_model_eval" src="https://github.com/user-attachments/assets/c23ebf26-e553-4d12-bcee-e8302bc121f0" />

<img width="400" height="400" alt="eval_embd_exp" src="https://github.com/user-attachments/assets/16936e0f-e4e8-4dcc-82b7-91b445ec5684" />


## Active Diffusion Architecture

The architectural elements are organized as follows:

### Surface Layer – Experiment Orchestration Layer (blue box)
Responsible for workflow coordination, configuration management, and resource allocation across HPC environments. It defines the experiment lifecycle.

### Deep Layer – Functional Core Layer (red box)
Contains the domain-specific components that implement the scientific and computational functionality of the diffusion process, including molecular generation and property evaluation.

### Utilities Subsystem (green box)
Provides cross-cutting services that support both layers by handling data management, task parallelization, and trajectory synchronization.


```mermaid
graph TB
    subgraph orchestration["🔷 Experiment Orchestration Layer"]
        config["Configuration Manager"]
        workflow["Workflow Coordinator"]
        hpc["HPC Resource Allocator"]
    end
    
    subgraph utilities["🟢 Utilities Subsystem"]
        data["Data Manager"]
        parallel["Task Parallelizer"]
        sync["Trajectory Synchronizer"]
    end
    
    subgraph functional["🔴 Functional Core Layer"]
        gen["Molecular Generator<br/>(Diffusion Model)"]
        eval["Property Evaluator<br/>(DFT/QM)"]
        select["Active Selector"]
        train["Model Trainer"]
    end
    
    orchestration --> utilities
    utilities --> functional
    functional --> utilities
    
    gen --> eval
    eval --> select
    select --> train
    train --> gen
    
    style orchestration stroke:#4A90E2,stroke-width:3px,color:#fff
    style utilities stroke:#7CB342,stroke-width: 3px,color:#fff
    style functional stroke:#E53935,stroke-width:3px,color:#fff
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- HPC environment (optional, for large-scale experiments)


## Quick Start

1. **Plan a trajectory:**

2. **Run active diffusion:**

3. **Analyze results:**


## Project Structure

```
QMActiveDiffusion/
├── actdiff_experiments/    # Experiment orchestration scripts
├── analysis/               # Jupyter notebooks for result analysis
├── archive/                # Archived experiments and data
├── configs/                # Configuration files
├── diffusion_code/         # Diffusion model implementation
├── pipeline_blocks/        # Modular pipeline components
├── results/                # Experiment outputs
├── scripts/                # Utility scripts
├── trajectory/             # Trajectory planning tools
├── utils/                  # Helper functions and utilities
└── README. md
```

## Requirements

Core dependencies include:
- PyTorch
- NumPy
- Pandas
- RDKit
- ASE (Atomic Simulation Environment)
- Matplotlib/Seaborn (visualization)
- Jupyter (analysis notebooks)

See `requirements.txt` for a complete list.

## Acknowledgments

- This work was part of a research project under the supervision of Dr. Leonardo Medrano.
