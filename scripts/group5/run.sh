#!/bin/bash
eval_type=pi0-bridge
robot_type=widowx
ckpt=010000
task="eval-widowx"
DATA_PATH=./outputs
echo ${DATA_PATH}/${eval_type}/checkpoints/${ckpt}/pretrained_model
# exit

python ./scripts/group5/evaluate_fractal.py \
    --policy.path=${DATA_PATH}/${eval_type}/checkpoints/${ckpt}/pretrained_model \
    --env.type=${robot_type} \
    --env.task "${task}" \
    --device=cuda \
    --output_dir=./outputs/eval/${eval_type}
    # --eval.batch_size=10 \
    # --eval.n_episodes=100 \