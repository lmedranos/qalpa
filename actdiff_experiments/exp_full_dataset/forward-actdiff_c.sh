#!/bin/bash -l
#SBATCH -J active-loop-forward
#SBATCH -A p_nanoparticle
#SBATCH --time=20:00:00   
#SBATCH --begin=now                
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6                                        
#SBATCH --mem-per-cpu=8000                       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

# define experiment
exp="full_dataset"

# Read current iteration value
STATE_FILE="MY_ITER.state"
MY_ITER=$(cat "$STATE_FILE")


# Please spicify the location of base dataset that is used for intial training of the model
base_dataset="/data/horse/ws/miha493e-research_project/activediffusion/data/qm7x-aqm-gas-noP.hdf5"

# Please specify the paths to the project folders
root="/data/horse/ws/miha493e-research_project/activediffusion" # Parent folder for the project
data="$root/data"                                               # A folder for storing data 
model_code="$root/diffusion_code"                               # A folder containing e3mol_aqm folder with the diffusion model code
scripts="$root/scripts"                                         # A folder containing Python scripts for running the active loop
blocks="$root/pipeline_blocks"                                  # A folder containing Python files for different steps in the loop
utils="$root/utils"                                            # A folder containing utility functions for data management and parallelization
config="$root/configs"                                          # A folder containing yaml of diffusion model configs, Mace model for relaxation, and nctr1.in foe FHI-aims


# Load Anaconda environment and activate the conda environment
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $root/conda_envs/env-activeloop


data_exp="$data/data_${exp}" 
config_exp="$config/config_${exp}"


# Run the active loop script with the specified parameters, remember to edit dataset name and number of iterations
python3 forward.py --base_dataset $base_dataset  --dataset_name "aqm_qm7x" --data $data_exp --model_code $model_code --pipeline_blocks $blocks --actdiff_utils $utils --config $config_exp --num_iterations 10 --props "HLgap" "eMBD" "atNUM" --scores "SAScore" --filter_scheme "score" "accuracy" --iter $MY_ITER


echo "Forward Job completed please run backward-actdiff_b.sh on barnard."