#!/bin/bash -l
#SBATCH -J active-loop_100
#SBATCH -A p_nanoparticle
#SBATCH --time=165:00:00                    
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4                                               
#SBATCH --mem-per-cpu=8000                       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

echo "Running active loop script..."

# Please spicify the location of base dataset that is used for intial training of the model
# base_dataset="/data/horse/ws/miha493e-research_project/activediffusion/data/qm7x-aqm-gas-noP.hdf5"
base_dataset="/data/horse/ws/miha493e-research_project/activediffusion/data/molecules-fps-40000.hdf5"

# Please specify the paths to the project folders
root="/data/horse/ws/miha493e-research_project/activediffusion" # Parent folder for the project
data="$root/data"                                               # A folder for storing data 
model_code="$root/diffusion_code"                               # A folder containing e3mol_aqm folder with the diffusion model code
scripts="$root/scripts"                                         # A folder containing Python scripts for running the active loop
blocks="$root/pipeline_blocks"                                  # A folder containing Python files for different steps in the loop
utils="$root/utils"                                            # A folder containing utility functions for data management and parallelization
config="$root/configs"                                          # A folder containing yaml of diffusion model configs, Mace model for relaxation, and nctr1.in foe FHI-aims


# # Load FHI-aims and other necessary modules
# module load release/24.04  intel-compilers/2023.2.1  impi/2021.10.0   #imkl/2023.2.0  
# module load FHI-aims/240507

# Load Anaconda environment and activate the conda environment
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate $root/conda_envs/env-activeloop

# # Set environment variables for OpenMP and MKL
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MKL_NUM_THREADS=1 #$SLURM_CPUS_PER_TASK
# export MKL_DYNAMIC=FALSE

# define experimenr
n_atoms=100
data_natoms="$data/data_${n_atoms}" 
config_natoms="$config/config_${n_atoms}"

# Run the active loop script with the specified parameters, remember to edit dataset name and number of iterations
python3 main_loop.py --base_dataset $base_dataset  --dataset_name "aqm_qm7x" --data $data_natoms --model_code $model_code --pipeline_blocks $blocks --actdiff_utils $utils --config $config_natoms --num_iterations 10 --n_nodes $n_atoms

echo "Active loop script completed."