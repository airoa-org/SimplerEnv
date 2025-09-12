#!/bin/bash

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate simpler-benchmark-v2

ckpt=$1
action_ensemble=${2:-0}

echo "Evaluating checkpoint: $ckpt"
echo "Action ensemble: $action_ensemble"

args=( --ckpt-path "$ckpt" )
[[ "${action_ensemble,,}" =~ ^(true)$ ]] && args+=( --action-ensemble )

python scripts/g3_lerobotpi/challenge_widowx.py "${args[@]}"
