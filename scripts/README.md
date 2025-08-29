## Introduction

### WidowX
**タスクセッティング**

|  Task  |  Challenge  | Task Definition               |
|--------|-------------|--------------------------------|
| 1      | 1           | Pick `<object>`               |
| 2      | 2           | Stacking `<cube>` on `<cube>` |
| 3      | 2           | Put `<object>` on `<top>`     |
| 4      | 2           | Put `<object>` in basket      |

**ランダマイザープール**

| Pool Name | Element |
|-----------|---------|
|`<object>` |`green cube`, `yellow cube`, `eggplant`, `spoon`, `carrot`|
|  `<top>`  |`plate`, `towel`|
| `<cube>`  |`green cube`, `yellow cube`|


## 環境構築

### Docker外
```bash
git clone https://github.com/airoa-org/SimplerEnv.git
git checkout origin/benchmark-v2
git submodule update --init --recursive
cd SimplerEnv
git clone your_openpi
# サブモジュールの中にもbranchの切り替えとかが必要の可能性がある
# 例
# git branch -a
# git checkout remotes/origin/feature/fractal
cp docker/Dockerfile .
docker build -t simpler_env .
```

### Docker内
```bash
cd your_openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
cd ..

uv pip install "numpy<2.0"
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"
uv pip install mediapy
```


## Get Started

### openpiモデルを起動する
```bash
# current directory: /app/SimplerEnV
cd openpi
export SERVER_ARGS="policy:checkpoint --policy.config=pi0_bridge_low_mem_finetune --policy.dir=/path/to/ckpt"
uv run scripts/serve_policy.py $SERVER_ARGS
```

### WidowX Challengeを行う
```bash
# current directory: /app/SimplerEnV
python scripts/openpi/challenge_widowx.py --ckpt /path/to/ckpt --control-freq 5
```
