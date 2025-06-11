#!/bin/bash

CMD=$1

pwd
echo $CMD

source ~/.bashrc
source /home/group_25b505/group_3/apps/miniconda3/etc/profile.d/conda.sh

conda activate simpler_env_rl

if [ $CMD == "rllib_train" ]; then
    python simpler_env/rls/train/rllib_test.py
elif [ $CMD == "octo_simpler_env_inference" ]; then
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
	    --policy octo-base \
	    --ckpt-path octo-base \
	    --task google_robot_pick_coke_can \
	    --logging-root ./results/results_simple_eval/ \
	    --n-trajs 100
elif [ $CMD == "octo_pt_simpler_env_inference" ]; then
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
	    --policy octo-pt-base \
	    --ckpt-path octo-base \
	    --task google_robot_pick_coke_can \
	    --logging-root ./results/results_simple_eval/ \
	    --n-trajs 100
elif [ $CMD == "finetuned_octo_pt_simpler_env_inference" ]; then
	python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
	    --policy octo-pt-base \
	    --ckpt-path /home/group_25b505/group_3/shared/models/octo/octo_fractal_20250608_134522/octo_finetune_fractal/fractal_finetune/fractal_octo_20250608_134522_20250608_134538/best_model \
	    --task google_robot_pick_coke_can \
	    --logging-root ./results/results_simple_eval/ \
	    --n-trajs 100
else
    echo "Invalid command: $CMD"
    exit 1
fi

