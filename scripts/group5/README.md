# **Installation & Evaluation Guide: SimplerEnv + ManiSkill2\_real2sim + geniac25\_team5\_codebase**

This guide provides step-by-step instructions to set up the environment, install dependencies, and run robot evaluation scripts.

## 1. Create and Activate Conda Environment

Create a dedicated environment to avoid conflicts with system packages:

```bash
conda create -n simpler_env python=3.10 -y
conda activate simpler_env
```

## 2. Clone the Repository

Clone the SimplerEnv repository including all submodules:

```bash
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules --depth 1
cd SimplerEnv
```

## 3. Install Core Dependencies

Install essential Python packages:

```bash
pip install numpy==1.24.4
```

## 4. Install ManiSkill2\_real2sim

Install ManiSkill2\_real2sim in editable mode:

```bash
cd ManiSkill2_real2sim
pip install -e .
cd ..
```

## 5. Install SimplerEnv

Install SimplerEnv in editable mode:

```bash
pip install -e .
```

## 6. Install geniac25\_team5\_codebase

Install the team’s codebase in editable mode:

```bash
cd geniac25_team5_codebase
pip install -e .
cd ..
```

## 7. Install Additional Packages

Some extra dependencies required for evaluation and experiments:

```bash
pip install rerun-sdk==0.23.1 bitsandbytes transformers==4.48.1 pytest
```

## 8. Reinstall draccus with Specific Version

Ensure compatibility with the project:

```bash
pip uninstall -y draccus
pip install draccus==0.10.0
```

## 9. Install FFmpeg

Required for video processing during evaluation:

```bash
conda install -c conda-forge ffmpeg -y
```

## 10. Running Evaluation

### Evaluate on WidowX Robot

```bash
bash scripts/group5/eval_widowx.sh
```

### Evaluate on Google Robot

```bash
bash scripts/group5/eval_google_robot.sh
```

> **Note:** Evaluation results, including logs and videos, will be saved under `{this_repo}/results`.




```bash
uv venv -p 3.10 scripts/group5/.venv

source $(pwd)/scripts/group5/.venv/bin/activate


uv pip install numpy==1.24.4
cd ManiSkill2_real2sim
uv pip install -e .
cd ..
uv pip install -e .



cd geniac25_team5_codebase
uv pip install -e .
uv pip install rerun-sdk==0.23.1 bitsandbytes transformers==4.48.1 pytest
uv pip install draccus==0.10.0

# cd ..
# uv pip install -e . ".[torch]"



curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip
./aws/install

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=

aws s3 ls s3://airoa-fm-development-competition/group5/ --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
aws s3 cp s3://airoa-fm-development-competition/group5/pi0-bridge_fractal ./ --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
aws s3 cp s3://airoa-fm-development-competition/group5/pi0-bridge_fractal/ ./ --recursive --endpoint-url=https://s3.ap-northeast-1.wasabisys.com

outputs/pi0-bridge/checkpoints/010000に移動

bash scripts/group5/eval_fractal.sh
```

pi0-bridge_fractalに含まれているconfig.jsonから以下のキーを削除する必要があります。
削除するキー：load_vlm_weights、vlm_model_name