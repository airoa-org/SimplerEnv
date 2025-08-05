#!/bin/bash


source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate simpler_env_lerobotpi0

CKPT=$1

python scripts/g3_lerobotpi0/evaluate_fractal.py \
    --ckpt-path $CKPT
