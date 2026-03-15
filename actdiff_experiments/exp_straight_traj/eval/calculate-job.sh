#!/bin/bash -l
#SBATCH -J calculate-straight-traj
#SBATCH -A p_nanoparticle
#SBATCH --time=05:00:00  
#SBATCH --begin=now
#SBATCH --partition=barnard
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000                    
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=32
#SBATCH -o check-log2.out
#SBATCH -e check-log2.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

n_parallel=${SLURM_NNODES} 

base_dataset="/data/horse/ws/miha493e-research_project/activediffusion/data/qm7x-aqm-gas-noP.hdf5"

# Please specify the paths to the project folders
root="/data/horse/ws/miha493e-research_project/activediffusion" # Parent folder for the project
data="$root/data"                                               # A folder for storing data 
model_code="$root/diffusion_code"                               # A folder containing e3mol_aqm folder with the diffusion model code
scripts="$root/scripts"                                         # A folder containing Python scripts for running the active loop
blocks="$root/pipeline_blocks"                                  # A folder containing Python files for different steps in the loop
utils="$root/utils"                                            # A folder containing utility functions for data management and parallelization
config="$root/configs"                                          # A folder containing yaml of diffusion model configs, Mace model for relaxation, and nctr1.in foe FHI-aims


# Load FHI-aims and other necessary modules
module load release/24.04  intel-compilers/2023.2.1  impi/2021.10.0   #imkl/2023.2.0  
module load FHI-aims/240507

# Load Anaconda environment and activate the conda environment
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $root/conda_envs/env-activeloop


# define experiment
exp="straight_traj"
data_exp="$data/data_${exp}" 
config_exp="$config/config_${exp}"
config_eval="$config_exp/eval" 

# Run the active loop script with the specified parameters, remember to edit dataset name and number of iterations
python3 calculate.py --base_dataset $base_dataset  --dataset_name "aqm_qm7x" --data $data_exp --model_code $model_code --pipeline_blocks $blocks --actdiff_utils $utils --config $config_eval --props "HLgap" "eMBD" "atNUM" --scores "SAScore" --filter_scheme "score" "accuracy" --max_parallel_tasks $n_parallel --num_models 6
