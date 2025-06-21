#!/bin/bash -l
#SBATCH -J nested_parallel
#SBATCH -A p_nanoparticle
#SBATCH --time=70:00:00                    # time (HH:MM:SS)
#SBATCH --partition=alpha                  # partition
#SBATCH --gres=gpu:2                       # number of GPUs
#SBATCH --mem=2G                           # memory
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited


root="/data/horse/ws/miha493e-activediffusion_hanna"
data="$root/datasets"
model_registry="$root/models_registry"
model_code="$root/diffusion_code"
pipeline="$root/pipeline"
config="$root/configs"

source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /data/horse/ws/miha493e-activediffusion_hanna/conda_envs/env-parallelization

# Please remember to spicify the location of base dataset
base_dataset = "/data/horse/ws/miha493e-activediffusion_hanna/datasets/qm7x-aqm-gas-noP.hdf5"

python3 main_loop.py --base_dataset "$base_dataset" \
                     --dataset_name "aqm_qm7x" \
                     --root "$root" \
                     --data "$data" \
                     --model_registry "$model_registry" \
                     --model_code "$model_code" \
                     --pipeline "$pipeline" \
                     --config "$config" \
                     --num_iterations 5