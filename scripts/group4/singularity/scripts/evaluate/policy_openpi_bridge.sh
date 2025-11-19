#!/bin/bash
cd ../
source $(pwd)/hsr_openpi/.venv/bin/activate
cd SimplerEnv

CUDA_VISIBLE_DEVICES=6 python3 ./scripts/group4/serve_policy.py --port 8000 --default_prompt "" policy:checkpoint --policy.config pi0_bridge_low_mem_finetune \
--policy.dir /home/group_25b505/group_4/members/mimura/geniac25_team4_codebase/models/hsr_openpi/checkpoints/pi0_bridge_low_mem_finetune/my_experiment/15000