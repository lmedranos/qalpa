#!/bin/bash -l
#SBATCH -J sp6-1o1
#SBATCH --time=24:00:00
#SBATCH --partition=barnard
#SBATCH -A p_nanoparticle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH -o dip2-A1.out
#SBATCH -e dip2-A1.err
#SBATCH --mail-type=all
#SBATCH        --mail-user=leonardo.medrano@tu-dresden.de
#SBATCH --mem-per-cpu=6000

ulimit -s unlimited

module load intel-compilers/2022.1.0 impi/2021.6.0 FHI-aims/240507

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

wdir=/project/scratch/p200243/Medrano/julian/qm7x-aqm-neq6/

prop=egap-pol
sys=tar1

idir=$wdir/$prop/optimized/eval1/$sys/
odir=$wdir/$prop/props/eval1/$sys/
mkdir $odir

python3 fhi1.py $idir $odir
