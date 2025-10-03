#!/bin/bash
# GIT_LFS_SKIP_SMUDGE=1 uv sync
# GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../
source $(pwd)/hsr_openpi/.venv/bin/activate
cd SimplerEnv
# uv pip install "numpy<2.0"
# uv pip install -e ./ManiSkill2_real2sim
# uv pip install -e .
# uv pip install "tensorflow-cpu==2.15.*"
# uv pip install mediapy

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python ./scripts/openpi/evaluate_fractal2.py\
 --ckpt-path /home/group_25b505/group_4/members/kuramitsu/geniac25_team4_codebase/models/hsr_openpi/checkpoints/pi0_fractal_low_mem_finetune/my_experiment/39999