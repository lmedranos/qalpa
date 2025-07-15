#!/bin/bash -l
#SBATCH -J test_fhi
#SBATCH --time=72:05:00
#SBATCH --partition=barnard
#SBATCH -A p_nanoparticle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=32000
#SBATCH        --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited


expdir=/data/horse/ws/miha493e-activediffusion_hanna/experiments/exp4
idir=$expdir/data/

prop=egap-pol
odir=$expdir/results/$prop/
mkdir $odir

source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /data/horse/ws/miha493e-activediffusion_hanna/conda_envs/env-fhi
module load intel-compilers/2022.1.0 impi/2021.6.0 FHI-aims/240507

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

#mpirun -np $SLURM_NTASKS 
python3 fhi_calc.py $idir $odir
