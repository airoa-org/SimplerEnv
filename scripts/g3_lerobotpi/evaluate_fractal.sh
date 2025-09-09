#!/bin/bash

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate simpler-benchmark-v2

ckpt=$1
action_ensemble=${2:-0}
sticky_action=${3:-0}

echo "Evaluating checkpoint: $ckpt"
echo "Action ensemble: $action_ensemble"
echo "Sticky action: $sticky_action"

args=( --ckpt-path "$ckpt" )
[[ "${action_ensemble,,}" =~ ^(true)$ ]] && args+=( --action-ensemble )
[[ "${sticky_action,,}"  =~ ^(true)$ ]] && args+=( --sticky-action )

python scripts/g3_lerobotpi/evaluate_fractal.py "${args[@]}"
