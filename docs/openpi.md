## Docker外
```bash
git clone https://github.com/airoa-org/SimplerEnv.git --recurse-submodules
cd SimplerEnv
git clone --recurse-submodules https://github.com/airoa-org/hsr_openpi.git  # サブモジュールの中にもbranchの切り替えとかが必要の可能性がある
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
uv pip install numpy==1.24.4
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"
uv pip install mediapy

# テスト
python tes.py

python simpler_env/policies/openpi/openpi_model.py --robot=widowx --control-freq=5 --sim-freq=500 --scene-name=bridge_table_1_v1 --rgb-overlay-path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png --env-name=StackGreenCubeOnYellowCubeBakedTexInScene-v0
```

## previous memo
```bash
git clone --recurse-submodules https://github.com/airoa-org/hsr_openpi.git
docker build -t simpler_env .


# conda install -c conda-forge libgl glib libvulkan-loader vulkan-tools vulkan-headers
# pip install uv

git clone https://github.com/airoa-org/hsr_openpi.git
cd hsr_openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cd ..
source $(pwd)/hsr_openpi/.venv/bin/activate
uv pip install numpy==1.24.4
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"


# for debug
uv venv
source .venv/bin/activate
uv pip install numpy==1.24.4
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
python tes.py
```


