import argparse
from types import SimpleNamespace
import os
import time
import tempfile
import json

import draccus
import torch
import numpy as np

from g3_haptics.policies.g3factory import get_policy_class, get_policy_config_class
from simpler_env.policies.g3lerobotpi.adapter import AiroaToG3Pi0FractalBridgeAdapter
from simpler_env.policies.g3lerobotpi.policy import G3Pi0multiLerobotToAiroaPolicy
from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket,
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

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    max_state_dim = 10
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
    PolicyCls = get_policy_class("g3pi0")  # -> G3PI0Policy
    cfg.n_action_steps = N_ACTION_STEPS
    policy = PolicyCls.from_pretrained(
        pretrained_name_or_path=args.ckpt_path,
        config=cfg,
        dataset_stats=dataset_stats,
    )

    policy = G3Pi0multiLerobotToAiroaPolicy(
        policy=policy, dataset_cfg=dataset_cfg, policy_setup="widowx_bridge"
    )

    env_policy = AiroaToG3Pi0FractalBridgeAdapter(
        policy=policy,
        rot6d=True,
        policy_setup="widowx_bridge",
        action_ensemble_temp=ACTION_ENSEMBLE_TEMP,
        action_ensemble=ACTION_ENSEMBLE,
        sticky_action=STICKY_ACTION,
        add_taks_prefix=True,        
    )

    print("Policy initialized. Starting evaluation...")

    tasks = [
        widowx_task1_pick_object,
        widowx_task2_stack_cube,
        widowx_task3_put_object_on_top,
        widowx_task4_put_object_in_basket,
    ]

    final_scores = []
    for task in tasks:
        cur_scores = task(
            env_policy=env_policy, ckpt_path=args.ckpt_path, control_freq=args.control_freq,
        )
        final_scores += cur_scores

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
