import argparse
from types import SimpleNamespace
import os
import json
import tempfile
import draccus
import torch
import time
import numpy as np

from simpler_env.policies.g3lerobotpi.adapter import AiroaToG3Pi0FractalBridgeAdapter
from simpler_env.policies.g3lerobotpi.policy import G3Pi0multiLerobotToAiroaPolicy

from g3_haptics.policies.g3factory import get_policy_class, get_policy_config_class

from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket,
)
from simpler_env.evaluation.evaluate import calculate_robust_score
from simpler_env.evaluation.evaluate import (
    run_comprehensive_evaluation,
    run_partial_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Comprehensive ManiSkill2 Evaluation"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--action-ensemble",
        type=bool,
        default=True,
        help="Whether to use action ensemble.",
    )
    parser.add_argument(
        "--action-ensemble-temp",
        type=float,
        default=-0.8,
        help="Temperature for action ensemble.",
    )
    parser.add_argument(
        "--rot6d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="applied state rot 6d.",
    )
    parser.add_argument(
        "--sticky-action", action="store_true", help="Use sticky action if set."
    )
    parser.add_argument(
        "--eval-task", type=str, required=True, help="Evaluation task name."
    )
    parser.add_argument(
        "--save-path-suffix",
        type=str,
        default="",
        help="Suffix to add to the save path.",
    )
    parser.add_argument(
        "--control-freq", type=int, default=5, help="Set control frequency (default->5)"
    )
    parser.add_argument(
        "--policy-setup",
        type=str,
        choices=["google_robot", "widowx_bridge"],  # 許可する値だけ
        default="google_robot",  # 既定値
        help="Choose the policy setup: 'google-robot' or 'widowx'.",
    )
    parser.add_argument(
        "--add-taks-prefix", action="store_true", help="add robot type in prompt."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.rot6d:
        max_state_dim = 10
    else:
        max_state_dim = 8

    dataset_cfg = SimpleNamespace(
        embodiments=[
            SimpleNamespace(  # Embodiment A
                state_key="observation.state",
                image_key=["observation.images.image"],
                action_key="action",
                ft_key=None,
            ),
            SimpleNamespace(  # Embodiment B
                state_key="observation.state",
                image_key=["observation.images.image"],
                action_key="action",
                ft_key=None,
            ),
        ],
        # ↓ いずれも「全エンボディメントの中での最大値」を入れるのがコツ
        max_state_dim=max_state_dim,  # 例：max(embA_state_dim, embB_state_dim)
        max_image_num=1,
        max_image_shape=[3, 256, 320],
        max_action_dim=7,  # 例：max(embA_action_dim, embB_action_dim)
    )

    ### make dummy dataset status
    E = 2
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(E, max_state_dim),
            "std": torch.ones(E, max_state_dim),
        },
        "action": {
            "mean": torch.zeros(E, 7),
            "std": torch.ones(E, 7),
        },
    }

    with open(os.path.join(args.ckpt_path, "config.json")) as f:
        raw_cfg = json.load(f)

    PolicyCfgClass = get_policy_config_class(
        raw_cfg.pop("type")
    )  # "g3pi0" -> G3PI0Config
    with tempfile.NamedTemporaryFile("w+") as tmp:
        json.dump(raw_cfg, tmp)
        tmp.flush()
        with draccus.config_type("json"):
            cfg = draccus.parse(PolicyCfgClass, tmp.name, args=[])

    # 3) クラスを取得して from_pretrained
    cfg.multi_embodiment = True
    print(f"cfg {cfg}")
    PolicyCls = get_policy_class("g3pi0")  # -> G3PI0Policy
    policy = PolicyCls.from_pretrained(
        pretrained_name_or_path=args.ckpt_path,
        config=cfg,
        dataset_stats=dataset_stats,
    )

    print(f"policy {policy}")

    policy = G3Pi0multiLerobotToAiroaPolicy(
        policy=policy, dataset_cfg=dataset_cfg, policy_setup=args.policy_setup
    )

    env_policy = AiroaToG3Pi0FractalBridgeAdapter(
        policy=policy,
        rot6d=args.rot6d,
        policy_setup=args.policy_setup,
        action_ensemble_temp=args.action_ensemble_temp,
        action_ensemble=args.action_ensemble,
        sticky_action=args.sticky_action,
        add_taks_prefix=args.add_taks_prefix        
    )
    if args.policy_setup == "google_robot":
        print("Policy initialized. Starting evaluation for google robot...")

        if args.eval_task == "all":
            final_scores = run_comprehensive_evaluation(
                env_policy=env_policy,
                ckpt_path=args.ckpt_path + f"_{args.save_path_suffix}",
            )
        else:
            final_scores = run_partial_evaluation(
                env_policy=env_policy,
                ckpt_path=args.ckpt_path + f"_{args.save_path_suffix}",
                task=args.eval_task,
            )

        print("\nEvaluation finished.")
        print(f"Final calculated scores: {final_scores}")
    elif args.policy_setup == "widowx_bridge":
        print("Policy initialized. Starting evaluation for widowx...")
        print("evaluation bridge")

        tasks = [
            widowx_task1_pick_object,
            widowx_task2_stack_cube,
            widowx_task3_put_object_on_top,
            widowx_task4_put_object_in_basket,
        ]

        final_scores = []
        for task in tasks:
            s = time.time()

            # Evaluate a task
            cur_scores = task(
                env_policy=env_policy,
                ckpt_path=args.ckpt_path + f"_{args.save_path_suffix}",
                control_freq=args.control_freq,
            )
            final_scores += cur_scores

            # Log the results
            print(f"Task: {task.__name__}")
            print(f"Time: {(time.time() - s) / 60:.2f} min")
            print(
                f"Success Rate: {np.mean(cur_scores) * 100:.2f}% ({np.sum(cur_scores)}/{np.prod(np.shape(cur_scores))})"
            )

        print("\nEvaluation finished.")
        print(f"Final Success Rate: {np.mean(final_scores) * 100:.2f}%")
        print(f"Final calculated scores: {calculate_robust_score(final_scores)}")
