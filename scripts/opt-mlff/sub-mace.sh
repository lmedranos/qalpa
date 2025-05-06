#!/bin/bash -l
#SBATCH -J otaq7x6-1o1
#SBATCH --time=48:00:00
#SBATCH -A p200243
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH  --ntasks-per-socket=16
#SBATCH -o dip.out
#SBATCH -e dip.err
#SBATCH --mail-type=all
#SBATCH        --mail-user=leonardo.medrano@tu-dresden.de

ulimit -s unlimited
#export OMP_NUM_THREADS=1

#source /home/users/u100062/miniconda3/envs/mace_env/bin/activate
#conda activate mace_env

module load env/release/2022.1
module load env/staging/2022.1

module load Python/3.10.4-GCCcore-11.3.0
source /project/home/p200243/medrano/mace-venv/bin/activate
module load intel/2022a

#xyz=/project/scratch/p200243/Medrano/dftbnn/data/all
#model=/project/scratch/p200243/Medrano/dftbnn/mace/10k
#
#python3 efor.py $model/MACE_model.model $xyz/dataset_nameALL.extxyz

export OMP_NUM_THREADS=1

#mpirun -n $SLURM_NTASKS /mnt/tier1/project/p200243/Medrano/mambaforge/bin/dftb+
#srun -N 1 /mnt/tier1/project/p200243/Medrano/mambaforge/bin/dftb+

sys=tar1
prop=egap-pol

xyz=/project/scratch/p200243/Medrano/julian/qm7x-aqm-neq6/$prop/evaluation1/$sys/
model=/project/scratch/p200243/Medrano/julian/mlff/qm7x-aqm-neq6/mace/train1/
odir=/project/scratch/p200243/Medrano/julian/qm7x-aqm-neq6/$prop/optimized/eval1/$sys/
mkdir $odir

python3 opt.py $model/MACE_model_swa.model $xyz $odir
