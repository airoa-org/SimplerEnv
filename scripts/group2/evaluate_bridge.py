import os
import argparse
import logging

from omegaconf import DictConfig, OmegaConf
from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket
)
from airoa_g2.model.policy.bridge_policy import BridgePolicy


def initialize_policy(cfg: DictConfig) -> BridgePolicy:
    """Create SimplerPolicy instance from Hydra configuration."""
    return BridgePolicy.from_hydra_config(cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    logging.basicConfig(level=getattr(logging, 'INFO'))

    logger = logging.getLogger(__name__)
    
    policy_config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    policy_cfg = OmegaConf.load(policy_config_path)
    policy_cfg.device = 'auto'
    policy_cfg.checkpoint_path = ckpt_path
    policy_cfg.dataset_dir = "/home/group_25b505/group_2/datasets"
    policy_cfg.asset_dir = "/home/group_25b505/group_2/assets"


    policy = initialize_policy(policy_cfg)
    logger.info("Policy loaded successfully!")

    print("Policy initialized. Starting evaluation...")
    tasks = [
        widowx_task1_pick_object, 
        widowx_task2_stack_cube, 
        widowx_task3_put_object_on_top, 
        widowx_task4_put_object_in_basket
    ]

    final_scores = []
    for task in tasks:
        cur_scores = task(
            env_policy=policy, ckpt_path=args.ckpt_path, control_freq=args.control_freq
        )
        final_scores += cur_scores

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
