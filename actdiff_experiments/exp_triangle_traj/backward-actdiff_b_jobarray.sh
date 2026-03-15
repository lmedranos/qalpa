#!/bin/bash -l
#SBATCH -J triangle-backward-array
#SBATCH -A p_phononics
#SBATCH -o check-log2.out
#SBATCH -e check-log2.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

# define time and resources
BATCH_SIZE=50
exp="triangle_traj"

root="/data/horse/ws/miha493e-research_project/activediffusion" # Parent folder for the project
data="$root/data"                                               # A folder for storing data 


# Read current iteration value
STATE_FILE="MY_ITER.state"
MY_ITER=$(cat "$STATE_FILE")

DATA_DIR="$data/data_${exp}/iter_${MY_ITER}/prop_calc"

# Count files
N_FILES=$(ls -1 $DATA_DIR | wc -l)

# Compute number of batches (ceiling division)
N_JOBS=$(( (N_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Submitting job array with $N_JOBS jobs, each processing up to $BATCH_SIZE files."

sbatch --array=0-$((N_JOBS-1)) backward_batch_task.sh $BATCH_SIZE $N_JOBS