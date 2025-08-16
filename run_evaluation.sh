#!/bin/bash

# RT-1とOcto評価実行スクリプト（統一最終版）

set -e

echo "=========================================="
echo "統一RT-1とOcto評価の実行"
echo "=========================================="

BASE_DIR="/root/workspace/SimplerEnv"
VENV_PATH="$BASE_DIR/hsr_openpi/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "エラー: 仮想環境が見つかりません"
    echo "セットアップを実行してください: bash $BASE_DIR/setup_evaluation.sh"
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "GPU確認..."
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

echo ""
echo "評価対象を選択してください:"
echo "1) RT-1のみ"
echo "2) Octoのみ" 
echo "3) RT-1とOcto両方"
echo "4) 動作確認テストのみ"

if [ "$1" = "--rt1" ]; then
    choice=1
elif [ "$1" = "--octo" ]; then
    choice=2
elif [ "$1" = "--both" ]; then
    choice=3
elif [ "$1" = "--test" ]; then
    choice=4
else
    read -p "選択 (1-4): " choice
fi

case $choice in
    1)
        echo "RT-1評価を実行..."
        bash "$BASE_DIR/scripts/rt1/evaluate_rt1_all.sh"
        ;;
    2)
        echo "Octo評価を実行..."
        bash "$BASE_DIR/scripts/octo/evaluate_octo_all_models.sh"
        ;;
    3)
        echo "RT-1評価を実行..."
        bash "$BASE_DIR/scripts/rt1/evaluate_rt1_all.sh"
        echo "Octo評価を実行..."
        bash "$BASE_DIR/scripts/octo/evaluate_octo_all_models.sh"
        ;;
    4)
        echo "動作確認テストを実行..."
        python "$BASE_DIR/test_evaluation.py"
        ;;
    *)
        echo "無効な選択です"
        exit 1
        ;;
esac

echo "🎉 評価完了！"