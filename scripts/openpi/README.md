## Docker外
```bash
# こっちをやるとmaniskill2のバージョンがずれてしまう
# git clone https://github.com/airoa-org/SimplerEnv.git --recurse-submodules

git clone https://github.com/airoa-org/SimplerEnv.git
git checkout origin/benchmark
git submodule update --init --recursive
cd SimplerEnv
git clone https://github.com/airoa-org/hsr_openpi.git  # サブモジュールの中にもbranchの切り替えとかが必要の可能性がある
# 例
# git branch -a
# git checkout remotes/origin/feature/fractal
docker build -t simpler_env .
```

## Docker内
```bash
# 環境構築
cd hsr_openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..
source $(pwd)/hsr_openpi/.venv/bin/activate
uv pip install "numpy<2.0"
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"
uv pip install mediapy


python simpler_env/policies/openpi/openpi_model.py --robot=widowx --control-freq=5 --sim-freq=500 --scene-name=bridge_table_1_v1 --rgb-overlay-path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png --env-name=StackGreenCubeOnYellowCubeBakedTexInScene-v0
CUDA_VISIBLE_DEVICES=1 python -m scripts.openpi.pick_coke_can_visual_matching --ckpt-paths /data/checkpoints/pi0_fractal_low_mem_finetune2/my_experiment/17000
```

