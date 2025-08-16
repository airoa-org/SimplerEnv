#!/bin/bash

# WidowX Octo評価統一実行スクリプト

set -e

echo "=================================================================="
echo "🤖 WidowX Octo評価スクリプト開始"
echo "=================================================================="

BASE_DIR="/root/workspace/SimplerEnv"
cd "$BASE_DIR"

# 環境変数設定
export PYTHONPATH="/root/.local/lib/python3.10/site-packages:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# GPU確認
echo "GPU確認..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
fi

# 引数のデフォルト値
POLICY_MODEL=${1:-"octo-base"}
NUM_EPISODES=${2:-"4"}
SCENE_NAME=${3:-"bridge_table_1_v2"}
INIT_RNG=${4:-"0"}

echo ""
echo "実行設定:"
echo "  ポリシーモデル: $POLICY_MODEL"
echo "  エピソード数: $NUM_EPISODES"
echo "  シーン名: $SCENE_NAME"
echo "  初期RNG: $INIT_RNG"
echo ""

# WidowXタスク実行関数
run_widowx_task() {
    local task_name=$1
    local env_name=$2
    local max_steps=$3
    local robot_setup=$4
    local rgb_overlay=$5
    local robot_x=$6
    local robot_y=$7
    
    echo "🚀 実行中: $task_name"
    echo "   環境: $env_name"
    echo "   最大ステップ: $max_steps"
    echo ""
    
    python3.10 simpler_env/main_inference.py \
        --policy-model "$POLICY_MODEL" \
        --ckpt-path None \
        --robot "$robot_setup" \
        --policy-setup widowx_bridge \
        --octo-init-rng "$INIT_RNG" \
        --control-freq 5 \
        --sim-freq 500 \
        --max-episode-steps "$max_steps" \
        --env-name "$env_name" \
        --scene-name "$SCENE_NAME" \
        --rgb-overlay-path "$rgb_overlay" \
        --robot-init-x "$robot_x" "$robot_x" 1 \
        --robot-init-y "$robot_y" "$robot_y" 1 \
        --obj-variation-mode episode \
        --obj-episode-range 0 "$NUM_EPISODES" \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-save-tags visual_matching \
        --logging-dir "$BASE_DIR/scripts/octo/results"
    
    echo "✅ $task_name 完了"
    echo ""
}

# メニュー表示
echo "利用可能なWidowXタスク:"
echo "1) put_eggplant_in_basket - なすをバスケットに入れる"
echo "2) put_carrot_on_plate - にんじんを皿に乗せる" 
echo "3) put_spoon_on_tablecloth - スプーンをテーブルクロスに置く"
echo "4) stack_green_cube_on_yellow - 緑のキューブを黄色いキューブに積む"
echo "5) all - 全タスクを実行"
echo ""

# タスク選択
TASK_CHOICE=${5:-"1"}

case $TASK_CHOICE in
    1|"put_eggplant_in_basket")
        echo "🥬 なすをバスケットに入れるタスクを実行"
        run_widowx_task \
            "put_eggplant_in_basket" \
            "PutEggplantInBasketScene-v0" \
            120 \
            "widowx_sink_camera_setup" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png" \
            0.127 \
            0.06
        ;;
    2|"put_carrot_on_plate")
        echo "🥕 にんじんを皿に乗せるタスクを実行"
        run_widowx_task \
            "put_carrot_on_plate" \
            "PutCarrotOnPlateInScene-v0" \
            60 \
            "widowx" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
            0.147 \
            0.028
        ;;
    3|"put_spoon_on_tablecloth")
        echo "🥄 スプーンをテーブルクロスに置くタスクを実行"
        run_widowx_task \
            "put_spoon_on_tablecloth" \
            "PutSpoonOnTableClothInScene-v0" \
            60 \
            "widowx" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
            0.147 \
            0.028
        ;;
    4|"stack_cube")
        echo "🟩 緑のキューブを黄色いキューブに積むタスクを実行"
        run_widowx_task \
            "stack_green_cube_on_yellow" \
            "StackGreenCubeOnYellowCubeBakedTexInScene-v0" \
            60 \
            "widowx" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
            0.147 \
            0.028
        ;;
    5|"all")
        echo "🔄 全WidowXタスクを実行"
        run_widowx_task \
            "put_eggplant_in_basket" \
            "PutEggplantInBasketScene-v0" \
            120 \
            "widowx_sink_camera_setup" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png" \
            0.127 \
            0.06
        
        run_widowx_task \
            "put_carrot_on_plate" \
            "PutCarrotOnPlateInScene-v0" \
            60 \
            "widowx" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
            0.147 \
            0.028
        
        run_widowx_task \
            "put_spoon_on_tablecloth" \
            "PutSpoonOnTableClothInScene-v0" \
            60 \
            "widowx" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
            0.147 \
            0.028
        
        run_widowx_task \
            "stack_green_cube_on_yellow" \
            "StackGreenCubeOnYellowCubeBakedTexInScene-v0" \
            60 \
            "widowx" \
            "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
            0.147 \
            0.028
        ;;
    *)
        echo "❌ 無効な選択: $TASK_CHOICE"
        echo "使用法: $0 [ポリシーモデル] [エピソード数] [シーン名] [初期RNG] [タスク番号]"
        echo "例: $0 octo-base 4 bridge_table_1_v2 0 1"
        exit 1
        ;;
esac

# 結果の場所を表示
echo "=================================================================="
echo "🎉 WidowX評価完了！"
echo "=================================================================="
echo ""
echo "📁 結果の保存場所:"
echo "   $BASE_DIR/scripts/octo/results/$POLICY_MODEL/"
echo ""
echo "📊 成功率や詳細な結果は各タスクディレクトリ内のファイルで確認できます。"
echo "🎬 実行動画(.mp4)も同じディレクトリに保存されています。"