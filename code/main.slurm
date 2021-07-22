#!/bin/bash

#SBATCH -J FMGNN
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -w gpu11
#SBATCH --gres=gpu:1
#SBATCH -o out.%j
#SBATCH -e err.%j

export CUDA_VISIBLE_DEVICES=1

cd  $SLURM_SUBMIT_DIR

DATASET=human
#DATASET=celegans
# DATASET=test
#DATASET=stitch900_1v5
INPUTFILE=main.py
OUTFILE=main.log
# radius=1
radius=2
#radius=3

# ngram=2
ngram=3

dim=10
layer_gnn=3
side=5
window=$((2*side+1))
layer_cnn=3
layer_output=3
lr=1e-4
lr_decay=0.01
decay_interval=10
weight_decay=1e-6

iteration=100

setting=$DATASET--radius$radius--ngram$ngram--dim$dim--layer_gnn$layer_gnn--window$window--layer_cnn$layer_cnn--layer_output$layer_output--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration
/public/home/jd_tcy/py38/bin/python $INPUTFILE $DATASET $radius $ngram $dim $layer_gnn $window $layer_cnn $layer_output $lr $lr_decay $decay_interval $weight_decay $iteration $setting > $OUTFILE
