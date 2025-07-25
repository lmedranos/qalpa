#!/bin/bash -l
#SBATCH -J nested_parallel
#SBATCH -A p_nanoparticle
#SBATCH --time=1:00:00                    # time (HH:MM:SS)
#SBATCH --partition=alpha                   # partition
#SBATCH --gres=gpu:2                       # number of GPUs
#SBATCH --mem=2G                          # memory
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

expdir="/data/horse/ws/miha493e-activediffusion_hanna/experiments/exp6"
data="$expdir/data"
model="$expdir/model"


source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /data/horse/ws/miha493e-activediffusion_hanna/conda_envs/env-parallelization


python3 main_loop.py