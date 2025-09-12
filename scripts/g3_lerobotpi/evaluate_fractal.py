import argparse

from scripts.g3_lerobotpi.g3_configuration_pi0 import G3PI0Config
from scripts.g3_lerobotpi.g3_pi0_or_fast import G3LerobotPiFastInference
from simpler_env.evaluation.evaluate import run_comprehensive_evaluation, run_partial_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--action-ensemble", action="store_true", help="Use action ensemble if set.")
    parser.add_argument("--sticky-action", action="store_true", help="Use sticky action if set.")
    parser.add_argument("--eval-task", type=str, required=True, help="Evaluation task name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = G3LerobotPiFastInference(
        saved_model_path=ckpt_path,
        policy_setup="google_robot",
        action_scale=1.0,
        action_ensemble=args.action_ensemble,
        sticky_action=args.sticky_action,
    )

    print("Policy initialized. Starting evaluation...")

    if args.eval_task == "all":
        final_scores = run_comprehensive_evaluation(env_policy=policy, ckpt_path=args.ckpt_path)
    else:
        final_scores = run_partial_evaluation(
            env_policy=policy, ckpt_path=args.ckpt_path, task=args.eval_task
        )

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
