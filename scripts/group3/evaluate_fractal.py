import argparse
import datetime

from scripts.group3.g3_configuration_pi0 import G3PI0Config
from scripts.group3.g3_pi0_or_fast import G3LerobotPiFastInference
from simpler_env.evaluation.fractal_tasks import run_comprehensive_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    N_ACTION_STEPS = 4
    ACTION_ENSEMBLE_TEMP = 0.8
    ACTION_ENSEMBLE = False
    STICKY_ACTION = False

    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = G3LerobotPiFastInference(
        saved_model_path=ckpt_path,
        policy_setup="google_robot",
        action_scale=1.0,
        action_ensemble_temp=ACTION_ENSEMBLE_TEMP,
        action_ensemble=ACTION_ENSEMBLE,
        sticky_action=STICKY_ACTION,
        n_action_steps=N_ACTION_STEPS,
    )

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=policy, ckpt_path=args.ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")
