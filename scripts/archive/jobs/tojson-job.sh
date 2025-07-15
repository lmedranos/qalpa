#!/bin/bash
#SBATCH -J test_json
#SBATCH --time=72:05:00
#SBATCH --partition=barnard
#SBATCH -A p_nanoparticle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o log.out
#SBATCH -e log.err
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=32000
#SBATCH        --mail-user=michael_gamal_nassif.hanna@mailbox.tu-dresden.de

ulimit -s unlimited

#work=/project/scratch/p200243/Medrano/julian/qm7x-aqm
#work=/project/scratch/p200243/Medrano/julian/qm7x-aqm-neq6
expdir=/data/horse/ws/miha493e-activediffusion_hanna/experiments/exp5
data_dir=$expdir/data

ps=egap-pol
#ps=egap-dip
#ps=mbd-pol
#ps=egap-pol-mbd

ev=eval3

ntar=1
for ii in $(seq 1 $ntar); do

	echo "$ps/$ev/tar$ii"
	
	jdir=$data_dir/$ps/json/$ev/tar$ii/
	mkdir -p $jdir

	python3 gen-json.py $data_dir/$ps/done/$ev/tar$ii/ $jdir

done
