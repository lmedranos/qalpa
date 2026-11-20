# Installation

## Requirements

- A Linux cluster with SLURM, a GPU partition (forward job) and a CPU partition (backward job)
- Mamba or Conda
- FHI-aims as an HPC module, only if you use it (licensed separately). DFTB+ is installed with the environments, and the EquiDTB model ships in {repo}`src/configs/`.

QALPA uses two environments: a GPU environment for the diffusion model and a CPU environment for QM and screening. A single combined environment is available for development.

## Split environments (recommended)

```bash
git clone https://github.com/lmedranos/qalpa.git
cd qalpa

# GPU: diffusion training and generation
mamba env create -f environment_f.yml -p conda_envs/env-qalpa-forward/

# CPU: geometry optimization, QM, screening
mamba env create -f environment_b.yml -p conda_envs/env-qalpa-backward/

# Patch ASE's DFTB+ calculator in the CPU environment
ASE_DFTB=conda_envs/env-qalpa-backward/lib/python3.10/site-packages/ase/calculators/dftb.py
cp "$ASE_DFTB" "$ASE_DFTB.bak"
cp src/dftb.py "$ASE_DFTB"
```

The job templates activate exactly these two paths, and the backward job points `DFTB_PREFIX` to the bundled Slater–Koster files. For interactive DFTB+ runs, export it yourself:

```bash
export DFTB_PREFIX="$PWD/src/configs/SKfiles/3ob-3-1/"
```

| Environment file | Path used by the job scripts | Used for |
|---|---|---|
| {repo}`environment_f.yml` | `conda_envs/env-qalpa-forward` | Forward job: split, train, plan, generate, dedup (GPU) |
| {repo}`environment_b.yml` | `conda_envs/env-qalpa-backward` | Backward job: screening, relaxation, QM, HDF5 (CPU) |
| {repo}`environment-al.yml` | `conda_envs/env-activeloop` | Both jobs on one machine (development) |

All three use Python 3.10 and include `mlflow` for the optional [training dashboards](../user-guide/mlflow.md).

## Combined environment (development)

:::{dropdown} Single machine, one environment
```bash
mamba env create -f environment-al.yml -p conda_envs/env-activeloop/
```

Apply the same ASE patch inside `conda_envs/env-activeloop/`, then change the `conda activate` line of both job scripts to:

```bash
conda activate $REPO_ROOT/conda_envs/env-activeloop
```
:::

## Next steps

- Adapt the SLURM job templates to your cluster: [Running on SLURM](../user-guide/running-on-slurm.md).
- Run your first experiment: [Quick start](quickstart.md).
- To use the browser app, install its few extra packages: [Navigator setup](../navigator/index.md).
