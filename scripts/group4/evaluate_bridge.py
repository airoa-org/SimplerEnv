import argparse

from openpi_client import action_chunk_broker

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket,
)
from simpler_env.policies.adapter import AiroaToSimplerBridgeAdapter
from simpler_env.policies.hsr_openpi.pi0_or_fast import OpenpiToAiroaPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--control-freq", type=int, default=5, help="Set control frequency (default->5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = OpenpiToAiroaPolicy(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=_policy_config.create_trained_policy(
                _config.get_config("pi0_fractal_low_mem_finetune"),
                ckpt_path,
            ),
            action_horizon=10,
        ),
    )
    env_policy = AiroaToSimplerBridgeAdapter(policy=policy)

    print("Policy initialized. Starting evaluation...")

    tasks = [widowx_task1_pick_object, widowx_task2_stack_cube, widowx_task3_put_object_on_top, widowx_task4_put_object_in_basket]

    final_scores = []
    for task in tasks:
        cur_scores = task(env_policy=env_policy, ckpt_path=args.ckpt_path, control_freq=args.control_freq)
        final_scores += cur_scores

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
