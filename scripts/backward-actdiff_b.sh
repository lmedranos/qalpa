#!/bin/bash -l
#SBATCH -J active-loop-test-prop
#SBATCH -A p_nanoparticle
#SBATCH --time=20:00:00                    
#SBATCH --partition=barnard
#SBATCH --cpus-per-task=10                    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

MY_ITER=1
echo "Running backward path active loop script ..."
echo "Iteration: $MY_ITER"

# Please spicify the location of base dataset that is used for intial training of the model
# base_dataset="/data/horse/ws/miha493e-research_project/activediffusion/data/qm7x-aqm-gas-noP.hdf5"
base_dataset="/data/horse/ws/miha493e-research_project/activediffusion/data/molecules-fps-40000.hdf5"

# Please specify the paths to the project folders
root="/data/horse/ws/miha493e-research_project/activediffusion" # Parent folder for the project
data="$root/data"                                               # A folder for storing data 
model_code="$root/diffusion_code2"                               # A folder containing e3mol_aqm folder with the diffusion model code
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
exp="embd_jmp"
data_exp="$data/data_${exp}" 
config_exp="$config/config_${exp}"

# Run the active loop script with the specified parameters, remember to edit dataset name and number of iterations
python3 backward.py --base_dataset $base_dataset  --dataset_name "aqm_qm7x" --data $data_exp --model_code $model_code --pipeline_blocks $blocks --actdiff_utils $utils --config $config_exp --num_iterations 10 --props "eMBD" "atNUM" --scores "SAScore" --filter_scheme "score" "accuracy" --max_parallel_tasks 10 --iter $MY_ITER

# if [ "${MY_ITER:-0}" -lt 9 ]; then
#     NEXT_ITER=$((MY_ITER + 1))
#     echo "Submitting next iteration: $NEXT_ITER"
#     ssh miha493e@login2.capella.hpc.tu-dresden.de \
#         "sbatch --export=MY_ITER=$NEXT_ITER \
#         /data/horse/ws/miha493e-research_project/activediffusion/scripts/jobs/job_embd_jumper/forward-actdiff_c.sh"
# else
#     echo "Reached the last iteration: $MY_ITER"
# fi