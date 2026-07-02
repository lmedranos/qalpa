#!/bin/sh

#mkdir /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_finetune_cutoff8_bonds7_addfeats/evaluation/docking/addfeats_nodes_fix

mkdir /scratch1/e3moldiffusion/logs/cdk2/docking/addfeats_vary_nodes_cutoff5_bonds5

python experiments/generate_ligands.py \
    --model-path /scratch1/e3moldiffusion/logs/crossdocked/x0_snr_addfeats_cutoff5_bonds5/best_valid.ckpt \
    --save-dir /scratch1/e3moldiffusion/logs/cdk2/docking/addfeats_vary_nodes_cutoff5_bonds5 \
    --test-dir /scratch1/cremej01/data/cdk2 \
    --skip-existing \
    --num-ligands-per-pocket 10000 \
    --batch-size 50 \
    --dataset-root /scratch1/cremej01/data/crossdocked_noH_cutoff5 \
    --vary-n-nodes \
    --n-nodes-bias 10
    #--fix-n-nodes
    #--build-obabel-mol \
    #--sanitize 
    #--relax-mol \
    #--max-relax-iter 500 

