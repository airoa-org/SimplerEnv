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

# 評価の実行
CUDA_VISIBLE_DEVICES=1 uv run scripts/openpi/evaluate_fractal.py --ckpt-path /data/checkpoints/pi0_fractal_low_mem_finetune2/my_experiment/17000
# about 30%

CUDA_VISIBLE_DEVICES=1 uv run scripts/openpi/evaluate_bridge.py --ckpt-path /data/checkpoints/pi0_bridge_low_mem_finetune2/my_experiment/17000
```

