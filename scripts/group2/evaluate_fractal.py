"""
Group2 Policy Evaluation Script for SimplerEnv Benchmark

This script evaluates Group2's model on the SimplerEnv benchmark tasks.
It implements the required AiroaBasePolicy interface and runs comprehensive evaluation.
"""

import argparse
import logging
import os
from typing import Dict
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from simpler_env.evaluation.fractal_tasks import run_comprehensive_evaluation

from airoa_g2.model.policy.fractal_policy import FractalPolicy


def initialize_policy(cfg: DictConfig) -> FractalPolicy:
    """Create SimplerPolicy instance from Hydra configuration."""
    return FractalPolicy.from_hydra_config(cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--asset_dir", type=str, required=True, help="Path to the checkpoint to evaluate.")
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
    policy_cfg.dataset_dir = ""
    policy_cfg.asset_dir = args.asset_dir
    
    policy = initialize_policy(policy_cfg)
    logger.info("Policy loaded successfully!")    
    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=policy, ckpt_path=args.ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")