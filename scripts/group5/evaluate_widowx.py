#!/usr/bin/env python3
# Lerobot-specific modules for policy construction and logging
from lerobot.common.logger import log_output_dir
from lerobot.common.utils.utils import (
    init_logging,          # Initialize logging configuration
    set_global_seed,       # Set global random seed for reproducibility
)
# Configuration parser and schema definition for evaluation pipeline
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig

# Standard libraries and third-party packages
import torch               # PyTorch for deep learning and tensor operations
import logging             # Logging utility
from pprint import pformat # Pretty-printing nested data structures
from dataclasses import asdict # Convert dataclass instances to dictionaries

from simpler_env.evaluation.evaluate_bridge import run_comprehensive_evaluation_bridge
from simpler_env.policies.group5.pi0 import Group5PiInference

@parser.wrap()
def main(cfg: EvalPipelineConfig) -> None:
    """
    Main entry point for evaluating a pretrained policy using the provided configuration.

    This function performs the following steps:
    - Logs the full configuration.
    - Selects a safe PyTorch device (CPU/GPU).
    - Configures PyTorch backend options for performance.
    - Sets a global random seed for reproducibility.
    - Logs the output directory path.
    - Asserts the presence of a pretrained model path.
    - Constructs and prepares the policy model for evaluation.

    Args:
        cfg (EvalPipelineConfig): Configuration object containing all parameters
                                  for evaluation, including device, policy, environment,
                                  seed, and output paths.
    """
    # Log the entire configuration in a pretty format
    logging.info(pformat(asdict(cfg)))

    # Enable cuDNN benchmark for improved performance
    torch.backends.cudnn.benchmark = True
    # Allow TensorFloat-32 (TF32) for matrix multiplications (CUDA)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Set global random seed for reproducibility
    set_global_seed(cfg.seed)

    # Log the output directory for saving results and logs
    log_output_dir(cfg.output_dir)

    # Ensure a pretrained model path is provided for evaluation
    assert cfg.policy.pretrained_path is not None, "Pretrained path must be specified for evaluation"
    logging.info(f"Using pretrained model for eval: {cfg.policy.pretrained_path}")
    
    # Construct the policy (model)
    logging.info("Making policy...")
    policy = Group5PiInference(cfg)

    logging.info("Policy initialized. Starting evaluation...")
    scores = run_comprehensive_evaluation_bridge(env_policy=policy, ckpt_path=cfg.policy.pretrained_path)

    logging.info("Evaluation finished.")
    logging.info(f"Final calculated scores: {scores}")

if __name__ == '__main__':
    init_logging()
    main()