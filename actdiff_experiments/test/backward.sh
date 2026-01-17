#!/bin/bash -l
#SBATCH -J active-loop-backward
#SBATCH -A p_nanoparticle
#SBATCH --time=00:05:00  
#SBATCH --begin=now
#SBATCH --partition=barnard
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000                    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -o check-log2.out
#SBATCH -e check-log2.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

echo "Running backward path active loop script ..."


# Read current iteration value
STATE_FILE="MY_ITER.state"
MY_ITER=$(cat "$STATE_FILE")

echo "Iteration: $MY_ITER"

sleep 40s

# Increment for next run
echo $((MY_ITER + 1)) > "$STATE_FILE"

echo "Backward Job completed."
