#!/bin/bash -l
#SBATCH -J test_diffusion
#SBATCH -A p_nanoparticle
#SBATCH --time=48:00:00                    # time (HH:MM:SS)
#SBATCH --partition=alpha                   # partition
#SBATCH --gres=gpu:2                       # number of GPUs
#SBATCH --mem=32G                          # memory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --cpus-per-task=1
#SBATCH -o check-log.out
#SBATCH -e check-log.err
#SBATCH --mail-type=all
#SBATCH --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

expdir="/data/horse/ws/miha493e-activediffusion_hanna/experiments/exp2"
data="$expdir/data"
dataset="$expdir/data/dataset"

source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /data/horse/ws/miha493e-activediffusion_hanna/conda_envs/e3moldiffusion

#export OMP_NUM_THREADS=1

PYTHONPATH=$expdir/code/e3mol_aqm

code="$expdir/code/e3mol_aqm/experiments"
#conf=$expdir/code/e3mol_aqm/configs
conf=$expdir/run
model=$expdir/model

echo "Running experiment"
#echo "Splitting data into train, val, test sets"
#python $code/data/aqm/split_data.py --file-path $data/qm7x-aqm-gas-noP.hdf5 --out-path  $dataset/
#echo "Training model"
#python $code/run_train.py --conf $conf/diffusion_aqm.yaml --save-dir $model/ --dataset aqm_qm7x --dataset-root $dataset/ --gpus 2 --batch-size 32
#python $code/run_train.py --conf $conf/diffusion_aqm.yaml --save-dir $model/ --dataset aqm_qm7x --dataset-root $dataset/ --gpus 2 --batch-size 32 --load-ckpt $work/run0/last.ckpt
echo "Generating molecules"
python $code/run_evaluation_qm.py --model-path $model/best_mol_stab.ckpt --save-dir $model/evaluation/tar1 --save-xyz --calculate-relax-change --calculate-props --batch-size 20 --ngraphs 200 --generate-ngraphs-valid --num-nodes 24 --fixed-context   -1.2       6
echo "experiment finished"
