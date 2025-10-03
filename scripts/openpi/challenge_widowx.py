import argparse
from datetime import datetime
from types import SimpleNamespace
import json
import os
import tempfile
import time
from typing import Dict, List

import draccus
import numpy as np
import torch

from scripts.g3_lerobotpi.g3_pi0_or_fast import G3LerobotPiFastInference
from scripts.g3_lerobotpi.submission_utils import (
    copy_run_summaries,
    dump_submission_payload,
    prepare_submission_dir,
)
from simpler_env.evaluation.bridge_tasks import (
    widowx_task1_pick_object,
    widowx_task2_stack_cube,
    widowx_task3_put_object_on_top,
    widowx_task4_put_object_in_basket,
)
from simpler_env.evaluation.evaluate import calculate_robust_score
from simpler_env.policies.g3lerobotpi.adapter import AiroaToG3Pi0FractalBridgeAdapter
from simpler_env.policies.g3lerobotpi.policy import G3Pi0multiLerobotToAiroaPolicy
from g3_haptics.policies.g3factory import get_policy_class, get_policy_config_class


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument(
        "--policy-setup",
        type=str,
        default="widowx_bridge",
        choices=["widowx_bridge", "google_robot"],
        help="Policy setup to use when loading the G3 Pi0 checkpoint.",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=5,
        help="Control frequency (Hz) used for evaluation (default 5)",
    )
    parser.add_argument(
        "--save-path-suffix",
        type=str,
        default="",
        help="Suffix appended to the checkpoint name when saving results.",
    )
    parser.add_argument(
        "--multi-embodiment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the multi-embodiment Pi0 policy pipeline.",
    )
    parser.add_argument(
        "--rot6d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 6D rotation state when running the multi-embodiment adapter.",
    )
    parser.add_argument(
        "--action-ensemble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle action ensemble (applies to both single and multi pipelines).",
    )
    parser.add_argument(
        "--action-ensemble-temp",
        type=float,
        default=-0.8,
        help="Temperature for the action ensemble module.",
    )
    parser.add_argument(
        "--sticky-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sticky gripper control when policy setup is google_robot.",
    )
    parser.add_argument(
        "--exec-horizon",
        type=int,
        default=4,
        help="Number of actions executed per inference step in the fast single-embodiment pipeline.",
    )
    parser.add_argument(
        "--add-task-prefix",
        action="store_true",
        help="Prepend the policy setup to the language prompt (multi-embodiment only).",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="submissions",
        help="Directory to write submission artifacts (score.json, summaries).",
    )
    parser.add_argument(
        "--submission-name",
        type=str,
        default=None,
        help="Optional fixed name for the submission folder (defaults to timestamped checkpoint id).",
    )
    parser.add_argument(
        "--no-save-submission",
        action="store_true",
        help="Skip writing submission artifacts; only print results.",
    )
    return parser.parse_args()


def _build_multi_policy(args: argparse.Namespace) -> AiroaToG3Pi0FractalBridgeAdapter:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    max_state_dim = 10 if args.rot6d else 8

    dataset_cfg = SimpleNamespace(
        embodiments=[
            SimpleNamespace(
                state_key="observation.state",
                image_key=["observation.images.image"],
                action_key="action",
                ft_key=None,
            ),
            SimpleNamespace(
                state_key="observation.state",
                image_key=["observation.images.image"],
                action_key="action",
                ft_key=None,
            ),
        ],
        max_state_dim=max_state_dim,
        max_image_num=1,
        max_image_shape=[3, 256, 320],
        max_action_dim=7,
    )

    emb_count = 2
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(emb_count, max_state_dim),
            "std": torch.ones(emb_count, max_state_dim),
        },
        "action": {
            "mean": torch.zeros(emb_count, 7),
            "std": torch.ones(emb_count, 7),
        },
    }

    with open(os.path.join(args.ckpt_path, "config.json"), encoding="utf-8") as f:
        raw_cfg = json.load(f)

    PolicyCfgClass = get_policy_config_class(raw_cfg.pop("type"))
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as tmp:
        json.dump(raw_cfg, tmp)
        tmp.flush()
        with draccus.config_type("json"):
            cfg = draccus.parse(PolicyCfgClass, tmp.name, args=[])

    cfg.multi_embodiment = True

    PolicyCls = get_policy_class("g3pi0")
    base_policy = PolicyCls.from_pretrained(
        pretrained_name_or_path=args.ckpt_path,
        config=cfg,
        dataset_stats=dataset_stats,
    )

    wrapped_policy = G3Pi0multiLerobotToAiroaPolicy(
        policy=base_policy,
        dataset_cfg=dataset_cfg,
        policy_setup=args.policy_setup,
    )

    return AiroaToG3Pi0FractalBridgeAdapter(
        policy=wrapped_policy,
        rot6d=args.rot6d,
        policy_setup=args.policy_setup,
        action_ensemble=args.action_ensemble,
        action_ensemble_temp=args.action_ensemble_temp,
        sticky_action=args.sticky_action,
        add_taks_prefix=args.add_task_prefix,
    )


def _build_single_policy(args: argparse.Namespace) -> G3LerobotPiFastInference:
    return G3LerobotPiFastInference(
        saved_model_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        action_scale=1.0,
        action_ensemble=args.action_ensemble,
        action_ensemble_temp=args.action_ensemble_temp,
        sticky_action=args.sticky_action,
        exec_horizon=args.exec_horizon,
    )


def _flatten_runs(runs: List[List[bool]]) -> List[bool]:
    return [bool(item) for episode in runs for item in episode]


if __name__ == "__main__":
    args = parse_args()

    ckpt_identifier = (
        args.ckpt_path
        if not args.save_path_suffix
        else args.ckpt_path + f"_{args.save_path_suffix}"
    )
    should_save, submission_path = prepare_submission_dir(
        args.submission_dir,
        args.submission_name,
        ckpt_identifier,
        default_tokens=[args.policy_setup, "widowx"],
        skip=args.no_save_submission,
    )

    if args.multi_embodiment:
        policy = _build_multi_policy(args)
    else:
        policy = _build_single_policy(args)

    print("Policy initialized. Starting evaluation...")

    tasks = [
        widowx_task1_pick_object,
        widowx_task2_stack_cube,
        widowx_task3_put_object_on_top,
        widowx_task4_put_object_in_basket,
    ]

    final_scores: List[List[bool]] = []
    task_breakdown: Dict[str, Dict[str, object]] = {}

    for task in tasks:
        start_time = time.time()

        cur_scores = task(
            env_policy=policy,
            ckpt_path=ckpt_identifier,
            control_freq=args.control_freq,
        )
        final_scores += cur_scores

        flat_scores = _flatten_runs(cur_scores)
        success_rate = float(np.mean(flat_scores)) if flat_scores else 0.0
        num_success = int(np.sum(flat_scores))
        num_trials = int(len(flat_scores))

        print(f"Task: {task.__name__}")
        print(f"Time: {(time.time() - start_time) / 60:.2f} min")
        print(f"Success Rate: {success_rate * 100:.2f}% ({num_success}/{num_trials})")

        task_breakdown[task.__name__] = {
            "success_rate": success_rate,
            "num_success": num_success,
            "num_trials": num_trials,
            "per_trial_success": [bool(x) for x in flat_scores],
        }

    flattened = _flatten_runs(final_scores)
    robust_score = float(calculate_robust_score(final_scores)) if final_scores else 0.0
    overall_success = float(np.mean(flattened)) if flattened else 0.0

    print("\nEvaluation finished.")
    print(f"Final Success Rate: {overall_success * 100:.2f}%")
    print(f"Final calculated scores: {robust_score}")

    if should_save and submission_path is not None:
        payload: Dict[str, object] = {
            "policy_setup": args.policy_setup,
            "checkpoint": args.ckpt_path,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "control_freq": args.control_freq,
                "multi_embodiment": args.multi_embodiment,
                "rot6d": args.rot6d,
                "action_ensemble": args.action_ensemble,
                "action_ensemble_temp": args.action_ensemble_temp,
                "sticky_action": args.sticky_action,
                "exec_horizon": args.exec_horizon,
                "add_task_prefix": args.add_task_prefix,
                "save_path_suffix": args.save_path_suffix,
            },
            "results": {
                "robust_score": robust_score,
                "overall_success_rate": overall_success,
                "num_success": int(np.sum(flattened)),
                "num_trials": int(len(flattened)),
                "task_breakdown": task_breakdown,
            },
        }

        dump_submission_payload(submission_path, payload)
        copy_run_summaries(submission_path, ckpt_identifier)
