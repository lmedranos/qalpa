#!/bin/bash -l
#SBATCH -J active-loop-backward
#SBATCH -A p_nanoparticle
#SBATCH --time=05:00:00  
#SBATCH --partition=barnard
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000                    
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=32
#SBATCH -o ../logs/backward_check-log.out
#SBATCH -e ../logs/backward_check-log.err


ulimit -s unlimited
export OMP_NUM_THREADS=1

# Read current iteration value
STATE_FILE="MY_ITER.state"
MY_ITER=$(cat "$STATE_FILE")

echo "Running backward path active loop script ..."
echo "Iteration: $MY_ITER"


# Repo root (Python package location) and experiment config dir (injected by init_experiment.py)
REPO_ROOT="@@REPO_ROOT@@"
EXP_CONFIG_DIR="@@EXP_CONFIG_DIR@@"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
config="$REPO_ROOT/EQMgen/configs"                              # diffusion model configs, MACE model, nctr1.in for FHI-aims


# Load FHI-aims and other necessary modules
module load release/24.04  intel-compilers/2023.2.1  impi/2021.10.0   #imkl/2023.2.0  
module load FHI-aims/240507

# Load Anaconda environment and activate the conda environment
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $REPO_ROOT/conda_envs/env-activeloop-backward

export DFTB_PREFIX="$config/SKfiles/3ob-3-1/" # Set DFTB+ prefix to the location of Slater-Koster files


python -m EQMgen.scripts.backward \
    --config-path "$EXP_CONFIG_DIR" \
    --config-name experiment_config \
    loop.current_iter="$MY_ITER"

if [ $? -ne 0 ]; then
    echo "Backward pipeline failed at iteration $MY_ITER"
    exit 1
else
    # else increment for next run
    echo $((MY_ITER + 1)) > "$STATE_FILE"
    echo "Backward Job completed" 
fi



