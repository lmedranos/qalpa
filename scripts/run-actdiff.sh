#!/bin/bash -l
#SBATCH -J active-loop
#SBATCH -A p_nanoparticle
#SBATCH --time=00:01:00                    # time (HH:MM:SS)
#SBATCH --partition=alpha
#SBATCH --gres=gpu:2                          # number of GPUs
#SBATCH --cpus-per-task=6                      # number of cores                                      
#SBATCH --mem-per-cpu=5000                          # memory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

echo "Running active loop script..."

root="/data/horse/ws/miha493e-research_project/activediffusion"
data="$root/datasets"
model_registry="$root/models_registry"
model_code="$root/diffusion_code"
pipeline="$root/pipeline"
config="$root/configs"

source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $root/conda_envs/env-activeloop

#module load intel-compilers/2022.1.0 impi/2021.6.0 
module load release/24.04  intel-compilers/2023.2.1  impi/2021.10.0      # development/24.04

module load FHI-aims/240507
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# Please remember to spicify the location of base dataset
base_dataset="/data/horse/ws/miha493e-research_project/activediffusion/datasets/qm7x-aqm-gas-noP.hdf5"

python3 main_loop.py --base_dataset $base_dataset  --dataset_name "aqm_qm7x" --root $root --data $data --model_registry $model_registry --model_code $model_code --pipeline $pipeline --config $config --num_iterations 1
echo "Active loop script completed."