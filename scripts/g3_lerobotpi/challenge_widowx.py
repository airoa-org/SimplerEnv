import argparse
import time

import numpy as np

from scripts.g3_lerobotpi.g3_configuration_pi0 import G3PI0Config
from scripts.g3_lerobotpi.g3_pi0_or_fast import G3LerobotPiFastInference
from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket
)
from simpler_env.evaluation.evaluate import calculate_robust_score

def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--control-freq", type=int, default=5, help="Set control frequency (default->5)")
    parser.add_argument("--action-ensemble", action="store_true", help="Use action ensemble if set.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = G3LerobotPiFastInference(
        saved_model_path=ckpt_path,
        policy_setup="widowx_bridge",
        action_scale=1.0,
        action_ensemble=args.action_ensemble,
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
        s = time.time()

        # Evaluate a task
        cur_scores = task(
            env_policy=policy, ckpt_path=args.ckpt_path, control_freq=args.control_freq
        )
        final_scores += cur_scores

        # Log the results
        print(f"Task: {task.__name__}")
        print(f"Time: {(time.time() - s)/60:.2f} min")
        print(f"Success Rate: {np.mean(cur_scores)*100:.2f}% ({np.sum(cur_scores)}/{np.prod(np.shape(cur_scores))})")

    print("\nEvaluation finished.")
    print(f"Final Success Rate: {np.mean(final_scores)*100:.2f}%")
    print(f"Final calculated scores: {calculate_robust_score(final_scores)}")
