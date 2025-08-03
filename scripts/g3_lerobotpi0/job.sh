#!/bin/bash


source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate simpler_env_lerobotpi0

python scripts/g3_lerobotpi0/evaluate_fractal.py \
    --ckpt-path ./scripts/g3_lerobotpi0/lerobot-pi0-fractal
