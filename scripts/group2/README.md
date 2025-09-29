# Setup

## Prerequisite

Install both conda and uv. Conda is used to install system-level packages, whereas UV is used to install packages required to run a model. 
Note that conda is not mandatory, if system-level packages required to run SimplerEnv benchmarks is already installed, e.g. via apt-get.

- uv (https://www.google.com/search?q=uv+install&oq=uv+install&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg7MgYIAhBFGDzSAQgxMjA4ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- (Optional) conda (https://www.anaconda.com/docs/getting-started/miniconda/install)

## Setup Group2 VLA codebase
```
git clone git@github.com:airoa-org/geniac25_team2_codebase.git
cd geniac25_team2_codebase
make setup # install packages
source .venv/bin/activate
```

## Download weights
```
uv pip install awscli
aws s3 cp s3://airoa-fm-development-competition/group2/postrain.simpler.step282500.min_max.B64.quat.mask.attn.interleave-hierarchical.tar.gz ./ \
  --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
tar -xvf postrain.simpler.step282500.min_max.B64.quat.mask.attn.interleave-hierarchical.tar.gz
```

## Run Simpler evaluation
```
python scripts/group2/evaluate_fractal.py --ckpt-path /PATH_TO_YOUR_WEIGHT/postrain.simpler.step282500.min_max.B64.quat.mask.attn.interleave-hierarchical/epochxx-global_stepxxx 

python scripts/group2/evaluate_bridge.py --ckpt-path /PATH_TO_YOUR_WEIGHT/postrain.simpler.step282500.min_max.B64.quat.mask.attn.interleave-hierarchical/epochxx-global_stepxxx 
```
