#!/bin/bash

pwd

source ~/.bashrc
source /home/group_25b505/group_3/apps/miniconda3/etc/profile.d/conda.sh

conda activate simpler_env_rl

python simpler_env/rls/train/rllib_test.py
