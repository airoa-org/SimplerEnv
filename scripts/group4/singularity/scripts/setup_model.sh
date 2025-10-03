#!/bin/bash
source $(pwd)/.venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=.venv uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .



# uv pip install -e ./ManiSkill2_real2sim
# uv pip install tensorflow==2.15.0
# uv pip install -r requirements_full_install.txt
# uv pip install -e .
# uv pip install tensorflow[and-cuda]==2.15.1 tensorflow-cpu==2.15.1
# uv pip install -e . ".[torch]"
# uv pip install typing_extensions==4.12.2 pydantic==2.10.6
# uv pip install nvidia-cublas-cu12==12.9.1.4  nvidia-cudnn-cu12==9.13.0.50 nvidia-nccl-cu12==2.28.3
# uv pip install "jax[cuda12_pip]==0.5.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# uv pip install numpy==1.26.4 orbax-checkpoint==0.5.1