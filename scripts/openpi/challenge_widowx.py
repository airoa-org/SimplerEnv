import argparse

from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket
)
from simpler_env.policies.openpi.pi0_or_fast import OpenPiFastInference


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = OpenPiFastInference(saved_model_path=ckpt_path, policy_setup="widowx_bridge", action_scale=1.0)

    print("Policy initialized. Starting evaluation...")

    tasks = [
        widowx_task1_pick_object, 
        widowx_task2_stack_cube, 
        widowx_task3_put_object_on_top, 
        widowx_task4_put_object_in_basket
    ]

    final_scores = []
    for task in tasks:
        cur_scores = task(env_policy=policy, ckpt_path=args.ckpt_path)
        final_scores += cur_scores

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
