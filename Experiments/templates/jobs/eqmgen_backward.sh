#!/bin/bash -l
#SBATCH -J active-loop-backward
#SBATCH -A p_nanoparticle
#SBATCH --time=02:00:00  
#SBATCH --begin=now
#SBATCH --partition=barnard
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000                    
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=32
#SBATCH -o check-log2.out
#SBATCH -e check-log2.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited
export OMP_NUM_THREADS=1


# Read current iteration value
STATE_FILE="MY_ITER.state"
MY_ITER=$(cat "$STATE_FILE")

n_parallel=${SLURM_NNODES} 
echo "Running backward path active loop script ..."
echo "Iteration: $MY_ITER"


config="$root/configs"                                          # A folder containing yaml of diffusion model configs, Mace model for relaxation, and nctr1.in foe FHI-aims


# Load FHI-aims and other necessary modules
module load release/24.04  intel-compilers/2023.2.1  impi/2021.10.0   #imkl/2023.2.0  
module load FHI-aims/240507

# Load Anaconda environment and activate the conda environment
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $root/conda_envs/env-activeloop

export DFTB_PREFIX="$config/SKfiles/3ob-3-1/" # Set DFTB+ prefix to the location of Slater-Koster files



# Run the active loop script with the specified parameters, remember to edit dataset name and number of iterations
python3 backward_p.py 

# Increment for next run
echo $((MY_ITER + 1)) > "$STATE_FILE"

echo "Backward Job completed please run forward-actdiff_c.sh on Capella." 