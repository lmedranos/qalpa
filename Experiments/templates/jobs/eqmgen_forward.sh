#!/bin/bash -l
#SBATCH -J active-loop-forward
#SBATCH -A p_nanoparticle
#SBATCH --time=20:00:00   
#SBATCH --begin=now                
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6                                        
#SBATCH --mem-per-cpu=8000                       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

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

 

config="$root/configs"                                          # A folder containing yaml of diffusion model configs, Mace model for relaxation, and nctr1.in foe FHI-aims
config_exp="$config/config_${exp}"


# Load Anaconda environment and activate the conda environment
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $root/conda_envs/env-activeloop


# Run the active loop script with the specified parameters, remember to edit dataset name and number of iterations
python3 forward_p.py 


echo "Forward Job completed please run backward-actdiff_b.sh on barnard."