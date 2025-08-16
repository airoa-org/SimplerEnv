import argparse
from typing import Dict
import numpy as np
import os

from simpler_env.policies.octo.octo_model import OctoInference
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy


class OctoToAiroaPolicy(AiroaBasePolicy):
    """OctoモデルをAiroaBasePolicy互換にラップするアダプター"""
    def __init__(self, octo_model):
        self.model = octo_model
        self.action_scale = octo_model.action_scale
        self.task_description = None

    def step(self, image: np.ndarray, eef_pos: np.ndarray, task_description: str) -> tuple:
        # Octoモデルのstepメソッドを呼び出し
        raw_action, action = self.model.step(image, eef_pos, task_description)
        
        # maniskill2_evaluatorの期待する形式で返す
        return raw_action, action

    def reset(self, task_description: str = None) -> None:
        # Octoモデルのresetを呼び出す
        if task_description is not None:
            self.task_description = task_description
            self.model.reset(task_description)
    
    def visualize_epoch(self, predicted_raw_actions, images, save_path):
        # Octoモデルのvisualize_epochメソッドに委譲
        return self.model.visualize_epoch(predicted_raw_actions, images, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive Octo Evaluation")
    parser.add_argument("--model-type", type=str, default="octo-base", 
                       choices=["octo-base", "octo-small"], 
                       help="Octo model type to evaluate.")
    parser.add_argument("--policy-setup", type=str, default="google_robot", 
                       choices=["google_robot", "widowx_bridge"], 
                       help="Policy setup for the robot.")
    parser.add_argument("--action-scale", type=float, default=1.0, 
                       help="Scaling factor for actions.")
    parser.add_argument("--horizon", type=int, default=2,
                       help="Observation history horizon.")
    parser.add_argument("--pred-action-horizon", type=int, default=4,
                       help="Action prediction horizon.")
    parser.add_argument("--exec-horizon", type=int, default=1,
                       help="Action execution horizon.")
    parser.add_argument("--image-size", type=int, default=256,
                       help="Input image size.")
    parser.add_argument("--init-rng", type=int, default=0,
                       help="Initial random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # JAX GPU設定
    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Octoモデルの初期化
    print(f"Loading Octo model: {args.model_type}")
    octo_model = OctoInference(
        model_type=args.model_type,
        policy_setup=args.policy_setup,
        horizon=args.horizon,
        pred_action_horizon=args.pred_action_horizon,
        exec_horizon=args.exec_horizon,
        image_size=args.image_size,
        action_scale=args.action_scale,
        init_rng=args.init_rng,
    )
    
    # AiroaPolicy互換のラッパーを作成
    policy = OctoToAiroaPolicy(octo_model)
    
    print("Octo model initialized. Starting comprehensive evaluation...")
    
    # 包括的評価の実行
    final_scores = run_comprehensive_evaluation(env_policy=policy, ckpt_path=args.model_type)
    
    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")