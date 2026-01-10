#!/bin/bash -l
#SBATCH -J test_dftb
#SBATCH --time=72:05:00
#SBATCH --partition=barnard
#SBATCH -A p_nanoparticle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o dip1.out
#SBATCH -e dip1.err
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=6000
#SBATCH        --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

expdir=/data/horse/ws/miha493e-activediffusion_hanna/experiments/exp1
idir=$expdir/data/
odir=$expdir/results/


source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /data/horse/ws/miha493e-activediffusion_hanna/conda_envs/env-dftb
export OMP_NUM_THREADS=1

export ASE_DFTB_COMMAND='srun -N 1 /data/horse/ws/miha493e-activediffusion_hanna/conda_envs/env-dftb/bin/dftb+'
export DFTB_PREFIX='/data/horse/ws/miha493e-activediffusion_hanna/SKfiles/3ob-3-1/'


python dftb_calc.py $idir/ $odir/

