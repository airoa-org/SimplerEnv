import argparse
from typing import Dict
import numpy as np
import tensorflow as tf
import os

from simpler_env.policies.rt1.rt1_model import RT1Inference
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy


class RT1ToAiroaPolicy(AiroaBasePolicy):
    """RT1モデルをAiroaBasePolicy互換にラップするアダプター"""
    def __init__(self, rt1_model):
        self.model = rt1_model
        self.action_scale = rt1_model.action_scale
        self.task_description = None

    def step(self, image: np.ndarray, eef_pos: np.ndarray, task_description: str) -> tuple:
        # RT1モデルのstepメソッドを呼び出し
        raw_action, action = self.model.step(image, eef_pos, task_description)
        
        # maniskill2_evaluatorの期待する形式で返す
        return raw_action, action

    def reset(self, task_description: str = None) -> None:
        # RT1モデルのresetを呼び出す
        if task_description is not None:
            self.task_description = task_description
            self.model.reset(task_description)
    
    def visualize_epoch(self, predicted_raw_actions, images, save_path):
        # RT1モデルのvisualize_epochメソッドに委譲
        return self.model.visualize_epoch(predicted_raw_actions, images, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive RT-1 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the RT-1 checkpoint to evaluate.")
    parser.add_argument("--policy-setup", type=str, default="google_robot", 
                       choices=["google_robot", "widowx_bridge"], 
                       help="Policy setup for the robot.")
    parser.add_argument("--action-scale", type=float, default=1.0, 
                       help="Scaling factor for actions.")
    parser.add_argument("--tf-memory-limit", type=int, default=3072,
                       help="GPU memory limit for TensorFlow in MB")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # TensorFlow GPU設定
    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    
    # RT1モデルの初期化
    print(f"Loading RT-1 model from: {args.ckpt_path}")
    rt1_model = RT1Inference(
        saved_model_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        action_scale=args.action_scale,
    )
    
    # AiroaPolicy互換のラッパーを作成
    policy = RT1ToAiroaPolicy(rt1_model)
    
    print("RT-1 model initialized. Starting comprehensive evaluation...")
    
    # 包括的評価の実行
    final_scores = run_comprehensive_evaluation(env_policy=policy, ckpt_path=args.ckpt_path)
    
    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")