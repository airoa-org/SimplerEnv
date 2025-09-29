# Group3 SimplerEnv Evaluation


## Setup
```
git clone https://github.com/airoa-org/SimplerEnv.git
cd SimplerEnv
git checkout benchmark-v2-g3-submission
git submodule update --init --recursive

# Create a conda environment
export REPO_ROOT="$(pwd -P)"
conda env create -f scripts/group3/environment.yml
conda activate simpler-benchmark-v2-g3-submission

# Downlaod the group3 simpler_env model from wasabi
aws s3 cp s3://airoa-fm-development-competition/group3/submitted_202509291552_simpler ./g3_simpler_model/ --recursive --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
```


## Evaluation

**Google Robot**
```
conda activate simpler-benchmark-v2-g3-submission
python scripts/group3/evaluate_fractal.py --ckpt-path ./g3_simpler_model
```


**WidowX**
```
conda activate simpler-benchmark-v2-g3-submission
python scripts/group3/evaluate_bridge.py --ckpt-path ./g3_simpler_model
```
