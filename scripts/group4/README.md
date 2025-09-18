
# SimplerEnv Group4
# 環境構築
---
    git clone https://github.com/airoa-org/SimplerEnv -b group4/benchmark-v2
---

## 1. SimplerEnv設定
0. Group4フォルダへ移動

---
    cd SimplerEnv/scripts/group4
---

1. 推論モデルのクローン
---
    bash singularity/scripts/clone_models.sh
---

## 2. Singularity
Singularity環境起動手順

### 2.0. 使用パスの設定
        - singularity/scripts/env.shのパスを設定
            - repositoryの場所、学習データの場所等
### 2.1. sifのビルド
    ---
        bash singularity/scripts/build_image.sh
    ---
### 2.2. コンテナの起動
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

    oxe weight model

    ---
        bash scripts_launch/evaluate/policy_openpi_oxe.sh
    ---

### 3.2 シミュレータープロセス
0. シミュレーター+clientの依存関係をインストール

    ---
        bash scripts_launch/setup_simulator.sh
    ---

1. シミュレーター実行

    1.1 fractal用

    ---
        bash scripts_launch/evaluate/sim_openpi_fractal.sh
    ---

    1.2 bridge用

    ---
        bash scripts_launch/evaluate/sim_openpi_bridge.sh
    ---