#!/bin/bash
eval_type=pi0-bridge
robot_type=google_robot # widowx
ckpt=010000
task="SimplerEnv"
DATA_PATH=./outputs
echo ${DATA_PATH}/${eval_type}/checkpoints/${ckpt}/pretrained_model

python ./scripts/group5/evaluate_fractal.py \
    --policy.path=${DATA_PATH}/${eval_type}/checkpoints/${ckpt}/pretrained_model \
    --env.type=${robot_type} \
    --env.task "${task}" \
    --device=cuda \
    --output_dir=./outputs/eval/${eval_type}
