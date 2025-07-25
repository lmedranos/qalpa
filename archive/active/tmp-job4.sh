#!/bin/bash -l
#SBATCH -J ss4-A1
#SBATCH --time=24:00:00
#SBATCH -A p_nanoparticle
#SBATCH --nodes=1              # request 2 nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6000
#SBATCH -o dip1.out
#SBATCH -e dip1.err
#SBATCH --mail-type=all
#SBATCH        --mail-user=leonardo.medrano@tu-dresden.de

ulimit -s unlimited

conda activate /data/horse/ws/leme420g-proj2025/qmdiff/mamba/miniforge3/envs/e3moldiffusion

PYTHONPATH=/data/horse/ws/leme420g-proj2025/qmdiff/e3mol_aqm
#export PYTHONPATH="/data/horse/ws/leme420g-proj2025/qmdiff/e3mol_aqm:$PYTHONPATH"

code=/data/horse/ws/leme420g-proj2025/qmdiff/e3mol_aqm/experiments
work=/data/horse/ws/leme420g-proj2025/qmdiff/code/active
data=/data/horse/ws/leme420g-proj2025/qmdiff/datasets/eq-QM7X.hdf5

echo "$data"

ntar=20
ptar=-0.20

python active_loop.py $code $data $work $ntar $ptar

