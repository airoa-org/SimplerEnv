import argparse
from typing import Dict

import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import action_chunk_broker, websocket_client_policy

from simpler_env.evaluation.adapter import AiroaToSimplerFractalAdapter
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy


class OpenpiToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy

    def step(self, obs: Dict) -> Dict:
        outputs = self.policy.infer(obs)
        outputs["terminate_episode"] = np.zeros(outputs["actions"].shape[0])
        return outputs

    def reset(self) -> None:
        self.policy.reset()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_bridge_low_mem_finetune",
        help="Name of the config to load from openpi.training.config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = OpenpiToAiroaPolicy(
        policy=action_chunk_broker.ActionChunkBroker(
            # policy = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000),
            policy=_policy_config.create_trained_policy(
                _config.get_config(args.config_name),
                ckpt_path,
            ),
            action_horizon=10,
        ),
    )

    env_policy = AiroaToSimplerFractalAdapter(policy=policy)

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=env_policy, ckpt_path=args.ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
