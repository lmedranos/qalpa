#!/bin/bash -l
#SBATCH -J test_mace
#SBATCH --time=72:05:00
#SBATCH --partition=barnard
#SBATCH -A p_nanoparticle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=6000
#SBATCH        --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

expdir=/data/horse/ws/miha493e-activediffusion_hanna/experiments/exp3
idir=$expdir/data/
odir=$expdir/results/
model=$expdir/model/


source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /data/horse/ws/miha493e-activediffusion_hanna/conda_envs/env-mace
export OMP_NUM_THREADS=1

start=$(date +%s)

python3 mace_calc.py $model/MACE_model_swa.model $idir $odir

end=$(date +%s)
runtime=$((end-start))
echo "Job runtime: $runtime seconds"