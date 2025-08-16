#!/bin/bash
declare -a checkpoints=(
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_tf_trained_for_000001120/"
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_tf_trained_for_000058240/"
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_tf_trained_for_000400120/"
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_x_tf_trained_for_002272480_step/"
)

gpu_id=0
echo "RT-1全チェックポイント評価開始（統一版）"

for ckpt_path in "${checkpoints[@]}"; do
    echo "Evaluating: ${ckpt_path}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python /root/workspace/SimplerEnv/scripts/rt1/evaluate_rt1.py \
        --ckpt-path "${ckpt_path}" \
        --policy-setup google_robot \
        --action-scale 1.0 \
        --tf-memory-limit 3072
done

echo "🎉 RT-1評価完了"
