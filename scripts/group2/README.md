# Setup

## Prerequisite

Install both conda and uv. Conda is used to install system-level packages, whereas UV is used to install packages required to run a model. 
Note that conda is not mandatory, if system-level packages required to run SimplerEnv benchmarks is already installed, e.g. via apt-get.

- uv (https://www.google.com/search?q=uv+install&oq=uv+install&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg7MgYIAhBFGDzSAQgxMjA4ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- 9Optional) conda (https://www.anaconda.com/docs/getting-started/miniconda/install)

## Setup Group2 VLA codebase
```
git clone git@github.com:airoa-org/geniac25_team2_codebase.git
cd geniac25_team2_codebase
uv sync # install packages
conda ...
```

## Download weights
```
uv pip install awscli
aws cp xxx xxxx
```

## Run Simpler evaluation
```
python scripts/group2/evaluate_fractal.py --ckpt-path xxx
python scripts/group2/evaluate_bridge.py --ckpt-path xxx
```
