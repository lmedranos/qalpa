# QM Active Learning Diffusion for Molecular Design

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This model uses the Diffusion model and Density Functional Theory (DFT) through active learning to navigate chemical space and improve our generative model for molecular design.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Phase I – Trajectory Planning](#phase-i--trajectory-planning)
  - [Phase II – Active Diffusion Execution](#phase-ii--active-diffusion-execution)
  - [Phase III – Output Analysis](#phase-iii--output-analysis)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Overview

The QMActiveDiffusion framework combines diffusion models with quantum mechanical calculations to generate novel molecules with desired properties. The system leverages active learning to iteratively improve molecular generation by incorporating DFT feedback, enabling efficient exploration of chemical space.

**Key Features:**
- 🧬 Molecular trajectory planning in property space
- 🔄 Iterative active learning with DFT feedback
- 🚀 HPC-optimized workflow orchestration
- 📊 Comprehensive analysis and visualization tools
- 🎯 Property-guided molecular generation

## Architecture

The software system is organized into a modular architecture that follows a three-phase workflow, guiding the process from molecular trajectory design to data analysis. 

### Phase I – Trajectory Planning

This phase provides an interactive environment for defining molecular trajectories within a property space (e.g., HOMO-LUMO gap, many-body dispersion energy, atomic number). The resulting trajectory files (`trajectory_*.csv`) serve as input for the active learning process. 

<img width="200" height="200" alt="trajectory_planer" src="https://github.com/user-attachments/assets/a827726d-cfd1-411e-80ce-79d87f5f7dbe" />

**Tools:** Interactive trajectory planner in the `trajectory/` directory

### Phase II – Active Diffusion Execution

This represents the core operational layer of the system, responsible for orchestrating the iterative molecular generation, evaluation, and feedback process. 

The architectural elements are organized as follows:
- **Top Layer – Experiment Orchestration Layer** (blue box): Responsible for workflow coordination, configuration management, and resource allocation across HPC environments. It defines the experiment lifecycle. 
- **Bottom Layer – Functional Core Layer** (red box): Contains the domain-specific components that implement the scientific and computational functionality of the diffusion process, including molecular generation and property evaluation.
- **Utilities Subsystem** (green box): Provides cross-cutting services that support both layers by handling data management, task parallelization, and trajectory synchronization.

<img width="280" height="240" alt="active_diffusion_execution" src="https://github.com/user-attachments/assets/bb0c4d90-376a-4cf1-ad2c-f767dece0c78" />

### Phase III – Output Analysis

The final phase focuses on analyzing and visualizing the output. Jupyter notebooks (`accuracy_analysis.ipynb`, `iteration_analysis.ipynb`, `hdf5_analysis.ipynb`) allow users to inspect the active diffusion results, model performance, and generated molecular properties.

<img width="400" height="400" alt="embd_model_eval" src="https://github.com/user-attachments/assets/c23ebf26-e553-4d12-bcee-e8302bc121f0" />

<img width="400" height="400" alt="eval_embd_exp" src="https://github.com/user-attachments/assets/16936e0f-e4e8-4dcc-82b7-91b445ec5684" />

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
