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
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

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
- DFT software (e.g., ORCA, Gaussian, or similar)
- HPC environment (optional, for large-scale experiments)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mikeHayah/QMActiveDiffusion.git
cd QMActiveDiffusion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your environment: 
```bash
cp configs/example_config.yaml configs/my_config.yaml
# Edit my_config.yaml with your settings
```

## Quick Start

1. **Plan a trajectory:**
```bash
python trajectory/trajectory_planner.py --property HLgap --range 1.0 5.0
```

2. **Run active diffusion:**
```bash
python scripts/run_active_diffusion.py --config configs/my_config.yaml --trajectory trajectory/trajectory_001.csv
```

3. **Analyze results:**
```bash
jupyter notebook analysis/accuracy_analysis.ipynb
```

## Usage

### Running Experiments

Execute a complete active learning cycle:

```bash
python actdiff_experiments/run_experiment.py \
  --config configs/experiment_config.yaml \
  --trajectory trajectory/trajectory_001.csv \
  --output results/experiment_001
```

### Configuration

Key configuration parameters in `configs/`:
- **Diffusion model settings:** Architecture, timesteps, noise schedule
- **Active learning parameters:** Selection strategy, batch size, iteration count
- **DFT settings:** Software, basis set, functional, convergence criteria
- **HPC settings:** Resource allocation, parallelization strategy

See `configs/README.md` for detailed configuration options.

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

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please follow these steps: 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qmactivediffusion,
  author = {Your Name},
  title = {QM Active Learning Diffusion for Molecular Design},
  year = {2026},
  url = {https://github.com/mikeHayah/QMActiveDiffusion}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by [funding sources]
- Built using [key frameworks/papers]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
