## 評価手順（MILE / SimplerEnv）

この手順では、`hsr_openpi` と `geniac25_team4_mile` で別々の `.venv` を用いて SimplerEnv 上で評価を実施します。

### 1) コンテナ起動
```bash
bash singularity/scripts/start_container.sh
```

### 2) 各モデルの .venv 準備
- OpenPI:
```bash
bash scripts_launch/setup_openpi_venv.sh
```

- MILE:
```bash
# 既定は python3.10。必要に応じて `PYTHON_BIN=python3.X` を指定
PYTHON_BIN=python3.11 bash scripts_launch/setup_mile_venv.sh
```

### 3) 評価の実行
- OpenPI（Fractal）
```bash
OPENPI_CKPT_PATH=/home/group_25b505/group_4/members/mimura/geniac25_team4_codebase/models/hsr_openpi/checkpoints/pi0_fractal_low_mem_finetune/my_experiment/1000 \
OPENPI_CONFIG_NAME=pi0_fractal_low_mem_finetune \
bash scripts_launch/evaluate/evaluate_openpi_fractal.sh
```

- MILE（Fractal）
```bash
MILE_CKPT_PATH=/home/group_25b505/group_4/members/koen/geniac25_team4_mile/checkpoints/oxe_fractal_robot_mile/best.pt \
MILE_CONFIG_NAME=mile_default \
bash scripts_launch/evaluate/evaluate_mile_fractal.sh
```

環境変数を指定しない場合はデフォルト値が使われます。

### 備考
- `.venv` はそれぞれ `models/hsr_openpi/.venv` と `models/geniac25_team4_mile/.venv`（コンテナ内では `/root/workspace/.../.venv`）に作成されます。
- MILE は CARLA 依存を持つため、必要に応じて各自で CARLA のセットアップを行ってください（本リポジトリでは詳細手順は割愛）。


