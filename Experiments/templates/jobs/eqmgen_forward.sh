#!/bin/bash -l
#SBATCH -J active-loop-forward
#SBATCH --time=20:00:00            
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6                                        
#SBATCH --mem-per-cpu=8000                       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ../logs/forward_check-log.out
#SBATCH -e ../logs/forward_check-log.err

ulimit -s unlimited
export OMP_NUM_THREADS=1

# Read current iteration value
STATE_FILE="MY_ITER.state"
if [ -f "$STATE_FILE" ]; then
    MY_ITER=$(cat "$STATE_FILE")
else
    echo "0" > "$STATE_FILE"
    MY_ITER=0
fi


# Repo root (Python package location) and experiment config dir (injected by init_experiment.py)
REPO_ROOT="@@REPO_ROOT@@"
EXP_CONFIG_DIR="@@EXP_CONFIG_DIR@@"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Load Anaconda environment and activate the conda environment
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $REPO_ROOT/conda_envs/env-activeloop


# Run the forward pipeline. 
python -m EQMgen.scripts.forward \
    --config-path "$EXP_CONFIG_DIR" \
    --config-name experiment_config \
    loop.current_iter="$MY_ITER"


if [ $? -ne 0 ]; then
    echo "Forward pipeline failed at iteration $MY_ITER"
    exit 1
else
    echo "Forward Job completed" 
fi

