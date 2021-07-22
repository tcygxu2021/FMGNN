#!/bin/bash

#DATASET=human
# DATASET=celegans
#DATASET=stitch900_1v5
DATASET=stitch900
radius=2

ngram=3

python preprocess.py $DATASET $radius $ngram
