#!/bin/bash
set -e  # エラーが出たら即終了

# venv が存在しない or python3 が壊れている場合は再作成
if [ ! -x "$(pwd)/.venv/bin/python3" ]; then
    echo "[INFO] .venv not found or broken. Recreating..."
    rm -rf .venv
    uv venv .venv
    cd ..
fi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../
source $(pwd)/hsr_openpi/.venv/bin/activate

cd SimplerEnv

uv pip install "numpy<2.0"
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"
uv pip install mediapy