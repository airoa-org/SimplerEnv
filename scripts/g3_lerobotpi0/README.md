## 環境構築

```
g3_shell -c4 --gres gpu:1
git clone https://github.com/airoa-org/SimplerEnv.git
cd SimplerEnv
git checkout origin/bechmark/g3/lerobotpi0
git submodule update --init --recursive

conda create -n simpler_env_lerobotpi0 python=3.11
conda activate simpler_env_lerobotpi0

pip install numpy==1.24.4
cd ManiSkill2_real2sim/
pip install -e .
cd ..
pip install -e .

cd scripts/g3_lerobotpi0/lerobot
conda install -c conda-forge evdev
pip install "av>=12.0.5"
pip install -e ".[pi0]"
pip install numpy==1.25.2 # opencv-pythonとのconflictが起きるが無視
pip install flash-attn==2.8.1
pip install pytest

pip install matplotlib
conda install -c conda-forge ffmpeg
conda install -c conda-forge libvulkan-loader libvulkan-headers
pip install mediapy
```


## 実行
**インタラクティブモード**
```
g3_shell -c4 --gres gpu:1
cd SimplerEnv
conda activate simpler_env_lerobotpi0
python scripts/g3_lerobotpi0/evaluate_fractal.py \
    --ckpt-path /home/group_25b505/group_3/members/user_00029_25b505/lerobot-pi0-fractal
```
* `--ckpt-path`で学習済みモデルのパスを指定

**Jobを投げる**
```
cd SimplerEnv
g3_sbatch \
	--gpus-per-node=1 \
	--output=output/%j.out \
	--time=24:00:00 \
	scripts/g3_lerobotpi0/job.sh \
	/home/group_25b505/group_3/members/user_00029_25b505/lerobot-pi0-fractal
```
* $1で学習済みモデルのパスを指定
