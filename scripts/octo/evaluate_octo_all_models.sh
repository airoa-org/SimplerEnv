#!/bin/bash

# Octoモデルの配列
declare -a octo_models=(
    "octo-base"
    "octo-small"
)

# GPU IDを設定（必要に応じて変更）
gpu_id=0

# 各Octoモデルに対して包括的評価を実行
for model_type in "${octo_models[@]}"; do
    echo "=========================================="
    echo "Evaluating Octo model: ${model_type}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python /home/group_25b505/group_4/members/kuramitsu/geniac25_team4_codebase/models/SimplerEnv/scripts/octo/evaluate_octo.py \
        --model-type "${model_type}" \
        --policy-setup google_robot \
        --action-scale 1.0 \
        --horizon 2 \
        --pred-action-horizon 4 \
        --exec-horizon 1 \
        --image-size 256 \
        --init-rng 0
    
    echo -e "\n\n"
done

echo "All Octo evaluations completed!"