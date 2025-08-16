#!/usr/bin/env python3
"""RT-1とOcto評価環境の統一テストスクリプト（最終版）"""

import sys
import os

def test_all():
    """全環境テスト"""
    print("RT-1とOcto評価環境の統一テスト開始...\n")
    
    print(f"Python: {sys.version}")
    print(f"実行環境: {sys.executable}\n")
    
    success = True
    
    # 基本環境
    try:
        import numpy as np
        import tensorflow as tf
        import simpler_env
        print(f"✓ 基本環境: numpy {np.__version__}, tensorflow {tf.__version__}")
    except ImportError as e:
        print(f"✗ 基本環境エラー: {e}")
        success = False
    
    # RT-1環境
    try:
        import tensorflow_hub
        import tf_agents
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        print(f"✓ RT-1環境: tf_agents {tf_agents.__version__}")
    except ImportError as e:
        print(f"✗ RT-1環境エラー: {e}")
        success = False
    
    # Octo環境
    try:
        import jax
        import flax
        import optax
        import distrax
        from octo.model.octo_model import OctoModel
        from simpler_env.policies.octo.octo_model import OctoInference
        print(f"✓ Octo環境: jax {jax.__version__}, flax {flax.__version__}")
    except ImportError as e:
        print(f"✗ Octo環境エラー: {e}")
        success = False
    
    # 評価フレームワーク
    try:
        from simpler_env.evaluation.scores import run_comprehensive_evaluation
        print("✓ 評価フレームワーク")
    except ImportError as e:
        print(f"✗ 評価フレームワークエラー: {e}")
        success = False
    
    # チェックポイント確認
    checkpoint_dir = "/home/group_25b505/group_4/datasets/checkpoints"
    checkpoints = ["rt_1_tf_trained_for_000001120/", "rt_1_tf_trained_for_000058240/", 
                  "rt_1_tf_trained_for_000400120/", "rt_1_x_tf_trained_for_002272480_step/"]
    
    print("\nチェックポイント確認:")
    for ckpt in checkpoints:
        if os.path.exists(os.path.join(checkpoint_dir, ckpt)):
            print(f"✓ {ckpt}")
        else:
            print(f"✗ {ckpt}")
            success = False
    
    # 評価スクリプト確認
    scripts = ["/root/workspace/SimplerEnv/scripts/rt1/evaluate_rt1_all.sh",
               "/root/workspace/SimplerEnv/scripts/octo/evaluate_octo_all_models.sh"]
    
    print("\n評価スクリプト確認:")
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {os.path.basename(script)}")
        else:
            print(f"✗ {script}")
            success = False
    
    # 環境作成テスト
    print("\n環境作成テスト:")
    try:
        import simpler_env
        env = simpler_env.make('google_robot_pick_coke_can')
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        print(f"✓ 環境作成・リセット成功: '{instruction}'")
        env.close()
    except Exception as e:
        print(f"✗ 環境作成エラー: {e}")
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 すべてのテスト成功！統一環境準備完了")
        print("\n評価実行: bash /root/workspace/SimplerEnv/run_evaluation.sh")
    else:
        print("❌ テスト失敗。セットアップ再実行してください")
        print("bash /root/workspace/SimplerEnv/setup_evaluation.sh")
    print("="*60)

if __name__ == "__main__":
    test_all()