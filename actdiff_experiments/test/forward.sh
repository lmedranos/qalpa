#!/bin/bash -l
#SBATCH -J active-loop-forward
#SBATCH -A p_nanoparticle
#SBATCH --time=00:05:00   
#SBATCH --begin=now                
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6                                        
#SBATCH --mem-per-cpu=8000                       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

echo "Starting active learning forward experiment"

# Read current iteration value
STATE_FILE="MY_ITER.state"
MY_ITER=$(cat "$STATE_FILE")

echo "Current iteration is: $MY_ITER"
sleep 40s

echo "Forward Job completed please run backward-actdiff_b.sh on barnard."

