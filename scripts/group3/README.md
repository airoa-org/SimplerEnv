# Group3 SimplerEnv Evaluation


## Setup
```
git clone https://github.com/airoa-org/SimplerEnv.git
cd SimplerEnv
git checkout benchmark-v2-g3-submission
git submodule update --init --recursive

conda create -n simpler-benchmark-v2-g3-submission python=3.11
conda activate simpler-benchmark-v2-g3-submission

pip install numpy==1.24.4
cd ManiSkill2_real2sim/
pip install -e .
cd ..
pip install -e .

conda install -c conda-forge ffmpeg libvulkan-loader libvulkan-headers
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 以下を実行するとconflictのエラーが出るが無視 (Error内容：simpler-env 0.0.1 requires numpy<2.0, but you have numpy 2.2.6 which is incompatible.)
pip install git+ssh://git@github.com/huggingface/lerobot.git@67196c9d5344cd932612cef79229f9d04134c91e#egg=lerobot[pi0]

# 以下を実行するとconflictのエラーが出るが無視 (Error内容：opencv-python-headless 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.25.2 which is incompatible.)
pip install numpy==1.25.2

pip install pytest
pip install -e "git+ssh://git@github.com/airoa-org/geniac25_team3_haptics.git@user_00047_25b505#egg=g3_haptics"
```


## Evaluation

**Google Robot**
```
cd SimplerEnv
conda activate simpler-benchmark-v2-g3-submission
python scripts/group3/evaluate_fractal.py --ckpt-path /path/to/ckpt
```


**WidowX**
```
cd SimplerEnv
conda activate simpler-benchmark-v2-g3-submission
python scripts/group3/evaluate_bridge.py --ckpt-path /path/to/ckpt
```
