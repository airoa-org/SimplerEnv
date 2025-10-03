import argparse

from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket,
)
from simpler_env.policies.rt1.rt1_model import RT1Inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--control-freq", type=int, default=5, help="Set control frequency (default->5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = RT1Inference(saved_model_path=ckpt_path, policy_setup="widowx_bridge")

    print("Policy initialized. Starting evaluation...")

    tasks = [widowx_task1_pick_object, widowx_task2_stack_cube, widowx_task3_put_object_on_top, widowx_task4_put_object_in_basket]

    final_scores = []
    for task in tasks:
        cur_scores = task(env_policy=policy, ckpt_path=args.ckpt_path, control_freq=args.control_freq)
        final_scores += cur_scores

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
