#!/bin/bash

# RT-1とOcto評価環境の統一セットアップスクリプト（最終版）

set -e

echo "=================================================================="
echo "RT-1とOcto評価環境のセットアップを開始します（統一版）"
echo "=================================================================="

BASE_DIR="/root/workspace/SimplerEnv"
CHECKPOINT_DIR="/home/group_25b505/group_4/datasets/checkpoints"

echo "統一ベースディレクトリ: $BASE_DIR"

# GPU確認
echo "GPU確認..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
fi

cd "$BASE_DIR"

echo ""
echo "=== 基本環境確認・セットアップ ==="

VENV_PATH="./hsr_openpi/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "基本セットアップを実行..."
    cd hsr_openpi
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    cd ..
    source $(pwd)/hsr_openpi/.venv/bin/activate
    uv pip install "numpy<2.0"
    uv pip install -e ./ManiSkill2_real2sim
    uv pip install -e .
    uv pip install "tensorflow-cpu==2.15.*"
    uv pip install mediapy
else
    source $(pwd)/hsr_openpi/.venv/bin/activate
fi

echo "=== RT-1/Octo依存関係確認 ==="

# RT-1依存関係
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null || pip install tensorflow==2.15.0
python -c "import tensorflow_hub" 2>/dev/null || pip install tensorflow_hub==0.16.0
python -c "import tf_agents" 2>/dev/null || pip install tf-agents==0.19.0

# Octo依存関係
python -c "import jax" 2>/dev/null || pip install "jax>=0.4.20"
python -c "import flax" 2>/dev/null || pip install "flax>=0.7.5"
python -c "import optax" 2>/dev/null || pip install "optax>=0.1.5"
python -c "import distrax" 2>/dev/null || pip install "distrax>=0.1.5"

# Octoパッケージ
python -c "from octo.model.octo_model import OctoModel" 2>/dev/null || {
    if [ ! -d "octo" ]; then
        git clone https://github.com/octo-models/octo/
        cd octo && git checkout 653c54acde686fde619855f2eac0dd6edad7116b && cd ..
    fi
    cd octo && pip install -e . && cd ..
}

echo "=== 評価スクリプト設定 ==="

mkdir -p "$BASE_DIR/scripts/rt1"

cat > "$BASE_DIR/scripts/rt1/evaluate_rt1_all.sh" << 'EOF'
#!/bin/bash
declare -a checkpoints=(
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_tf_trained_for_000001120/"
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_tf_trained_for_000058240/"
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_tf_trained_for_000400120/"
    "/home/group_25b505/group_4/datasets/checkpoints/rt_1_x_tf_trained_for_002272480_step/"
)

gpu_id=0
echo "RT-1全チェックポイント評価開始（統一版）"

for ckpt_path in "${checkpoints[@]}"; do
    echo "Evaluating: ${ckpt_path}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python /root/workspace/SimplerEnv/scripts/rt1/evaluate_rt1.py \
        --ckpt-path "${ckpt_path}" \
        --policy-setup google_robot \
        --action-scale 1.0 \
        --tf-memory-limit 3072
done

echo "🎉 RT-1評価完了"
EOF

chmod +x "$BASE_DIR/scripts/rt1/evaluate_rt1_all.sh"

echo "=== 最終確認 ==="

python -c "
try:
    import numpy, tensorflow, tensorflow_hub, tf_agents
    import jax, flax, optax, distrax
    from simpler_env.policies.rt1.rt1_model import RT1Inference
    from octo.model.octo_model import OctoModel
    from simpler_env.policies.octo.octo_model import OctoInference
    from simpler_env.evaluation.scores import run_comprehensive_evaluation
    print('🎉 すべての依存関係OK')
except Exception as e:
    print(f'❌ エラー: {e}')
"

declare -a checkpoints=("rt_1_tf_trained_for_000001120/" "rt_1_tf_trained_for_000058240/" "rt_1_tf_trained_for_000400120/" "rt_1_x_tf_trained_for_002272480_step/")
for ckpt in "${checkpoints[@]}"; do
    [ -d "$CHECKPOINT_DIR/$ckpt" ] && echo "✓ $ckpt" || echo "✗ $ckpt"
done

echo ""
echo "=================================================================="
echo "🎉 統一RT-1とOcto評価環境セットアップ完了！"
echo "=================================================================="
echo ""
echo "評価実行方法:"
echo "bash /root/workspace/SimplerEnv/run_evaluation.sh"