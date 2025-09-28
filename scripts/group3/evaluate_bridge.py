import argparse
import time

import numpy as np

from scripts.group3.g3_configuration_pi0 import G3PI0Config
from scripts.group3.g3_pi0_or_fast import G3LerobotPiFastInference
from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--control-freq", type=int, default=5, help="Set control frequency (default->5)")
    return parser.parse_args()


if __name__ == "__main__":
    N_ACTION_STEPS = 1
    ACTION_ENSEMBLE_TEMP = 0.8
    ACTION_ENSEMBLE = False
    STICKY_ACTION = False

    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = G3LerobotPiFastInference(
        saved_model_path=ckpt_path,
        policy_setup="widowx_bridge",
        action_scale=1.0,
        action_ensemble_temp=ACTION_ENSEMBLE_TEMP,
        action_ensemble=ACTION_ENSEMBLE,
        sticky_action=STICKY_ACTION,
        n_action_steps=N_ACTION_STEPS,
    )

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
