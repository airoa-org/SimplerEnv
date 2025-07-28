import argparse
from typing import Dict
import time
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import action_chunk_broker

import sys
import os
sys.path.append(os.path.dirname(__file__))
from droid_adapter import DroidAdapter
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy


class OpenpiToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy

    def step(self, obs: Dict) -> Dict:
        outputs = self.policy.infer(obs)
        outputs["terminate_episode"] = np.zeros(outputs["actions"].shape[0])
        return outputs

    def reset(self):
        self.policy.reset()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pi0 Droid policy")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--action-horizon", type=int, default=10, help="Action horizon for chunk broker")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Output directory for results")
    
    args = parser.parse_args()
    ckpt_path = args.ckpt_path

    print("Initializing Pi0 Droid policy...")
    start_time = time.time()
    
    # Create Pi0 policy with action_dim=32 (default), action_horizon=10
    policy = OpenpiToAiroaPolicy(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=_policy_config.create_trained_policy(
                _config.get_config("pi0_droid_low_mem_finetune"),
                ckpt_path,
            ),
            action_horizon=args.action_horizon,
        ),
    )

    # Use custom DROID adapter
    env_policy = DroidAdapter(policy=policy)
    
    init_time = time.time() - start_time
    print(f"Policy initialized in {init_time:.2f}s. Starting evaluation...")

    # Run evaluation with the correct function signature
    results = run_comprehensive_evaluation(
        env_policy, 
        ckpt_path
    )
    
    print("\n=== Pi0 Droid Evaluation Results ===")
    print(f"Model: Pi0 (Standard)")
    print(f"Action Dim: 32 (default)")
    print(f"Action Horizon: {args.action_horizon}")
    print(f"Initialization Time: {init_time:.2f}s")
    print(f"Total Episodes: {args.num_episodes}")
    
    if results:
        for task, score in results.items():
            print(f"{task}: {score:.3f}")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()