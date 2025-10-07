
# SimplerEnv Group4

システムは推論サーバー(モデル)とシミュレーターの2つのプロセスで動作し、両者はソケット通信によってやり取りを行います。そのため、tmux などを使い、2つのプロセスを同時に実行することを想定しています。

# 環境構築
まずは対象リポジトリをクローンします。
---
    git clone https://github.com/airoa-org/SimplerEnv -b group4/benchmark-v2
---

## 1. SimplerEnv設定
0. Group4フォルダへ移動

---
    cd SimplerEnv/scripts/group4
---

1. 推論モデルのクローン

推論に必要なモデルファイルを事前にcloneします。

---
    bash singularity/scripts/clone_models.sh
---

## 2. Singularity
Singularity環境起動手順

### 2.0. 使用パスの設定
        - singularity/scripts/env.shのパスを設定
            - repositoryの場所、学習データの場所等
### 2.1. sifのビルド

Singularity の .sif イメージを作成します。

    ---
        bash singularity/scripts/build_image.sh
    ---

### 2.2. コンテナの起動

作成したイメージからコンテナを起動します。

    ---
        bash singularity/scripts/start_container.sh
    ---


## 3. プロセス2つ
socket通信方式のため、tmux等を用いて複数プロセスを同時に動かすことを想定しています。

計算ノード上でtmuxを実行し、プロセスを分割してから、コンテナを起動します。

### 3.1 推論プロセス
0. 推論側の依存関係のインストール

    Server

    ---
        bash scripts_launch/setup_model.sh
    ---

1. サーバー側の実行

    推論用のサーバープロセスを立ち上げます。

    ---
        bash scripts_launch/evaluate/policy_openpi_oxe.sh
    ---

### 3.2 シミュレータープロセス
0. シミュレーター + clientの依存関係をインストール

    ---
        bash scripts_launch/setup_simulator.sh
    ---

1. シミュレーター実行

    利用する環境に応じて以下を選択します。

    1.1 fractal用

    ---
        bash scripts_launch/evaluate/sim_openpi_fractal.sh
    ---

    1.2 bridge用

    ---
        bash scripts_launch/evaluate/sim_openpi_bridge.sh
    ---

### model path
    - s3://airoa-fm-development-competition/group4/simplerenv_checkpoints/





# HPC意外での構築方法

```bash

git clone https://github.com/airoa-org/hsr_openpi.git -b feature/group4

cd hsr_openpi
git checkout feature/group4-

uv venv -p 3.10 scripts/group4/.venv
source $(pwd)/scripts/group4/.venv/bin/activate

cd hsr_openpi
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=../scripts/group4/.venv uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cd ..
uv pip install -e . ".[torch]"



aws configure
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=


mkdir checkpoints/group4
cd checkpoints/group4
aws s3 ls s3://airoa-fm-development-competition/group4/ --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
aws s3 cp s3://airoa-fm-development-competition/group4/simplerenv_checkpoints/ ./ --recursive --endpoint-url=https://s3.ap-northeast-1.wasabisys.com



CUDA_VISIBLE_DEVICES=0 python ./scripts/group4/serve_policy.py --port 8000 --default_prompt "" policy:checkpoint --policy.config pi0_fractal_low_mem_finetune --policy.dir checkpoints/group4


source $(pwd)/scripts/group4/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python ./scripts/openpi/evaluate_fractal2.py --ckpt-path checkpoints/group4
```