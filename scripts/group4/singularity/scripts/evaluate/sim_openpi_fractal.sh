#!/bin/bash
cd ../
source $(pwd)/SimplerEnv/.venv/bin/activate
cd SimplerEnv

CUDA_VISIBLE_DEVICES=7 uv run ./scripts/group4/evaluate_fractal.py\
 --ckpt-path /home/group_25b505/group_4/members/mimura/geniac25_team4_codebase/models/hsr_openpi/checkpoints/pi0_fractal_low_mem_finetune/my_experiment/15000\