import argparse
from collections import OrderedDict, defaultdict
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute episode scores from results.json")
    parser.add_argument("--result-path", required=True, help="Directory that contains results.json and where scores.csv will be written")
    parser.add_argument("--policy-setup", required=True, help="Policy setup string; Options: ['widowx', 'google_robot', 'both']")
    return parser.parse_args()


# You can modify the following to customize your scoring config
# 1) The "rubrics_score" dict MUST preserve order (use OrderedDict or rely on Python 3.7+ insertion order).
# 2) Weights MUST sum to 1.0.
widowx_rubrics_score = OrderedDict(
    [
        ("is_src_obj_grasped", 0.3),
        ("consecutive_grasp", 0.3),
        ("src_on_target", 0.4),
    ]
)
google_robot_move_near_rubrics_score = OrderedDict(
    [
        ("moved_correct_obj", 0.3),
        ("near_tgt_obj", 0.3),
        ("is_closest_to_tg", 0.4),
    ]
)
google_robot_put_in_rubrics_score = OrderedDict(
    [
        ("is_drawer_open", 0.3),
        ("grasped", 0.3),
        ("has_contact", 0.4),
    ]
)

WIDOWX_TASKS = {
    "widowx": {
        "challenge_1": [
            "widowx_task1_pick_object",
        ],
        "challenge_2": {
            "widowx_task2_stack_cube": widowx_rubrics_score,
            "widowx_task3_put_object_on_top": widowx_rubrics_score,
            "widowx_task4_put_object_in_basket": widowx_rubrics_score,
        },
    }
}

GOOGLE_ROBOT_TASKS = {
    "google_robot": {
        "challenge_1": [
            "fractal_pick_object_visual_matching",
            "fractal_pick_object_variant_agg",
            "fractal_pick_object_among_visual_matching",
            "fractal_pick_object_among_variant_agg",
            "fractal_drawer_visual_matching",
            "fractal_drawer_variant_agg",
        ],
        "challenge_2": {
            "fractal_move_near_visual_matching": google_robot_move_near_rubrics_score,
            "fractal_move_near_variant_agg": google_robot_move_near_rubrics_score,
            "fractal_put_in_drawer_visual_matching": google_robot_put_in_rubrics_score,
            "fractal_put_in_drawer_variant_agg": google_robot_put_in_rubrics_score,
        }
    }
}

# Build scoring config
def build_scoring_config():
    cfg = {}
    cfg.update(WIDOWX_TASKS)
    cfg.update(GOOGLE_ROBOT_TASKS)

    wid = WIDOWX_TASKS["widowx"]
    gr = GOOGLE_ROBOT_TASKS["google_robot"]
    cfg["both"] = {
        "challenge_1": list(wid["challenge_1"]) + list(gr["challenge_1"]),
        "challenge_2": {**wid["challenge_2"], **gr["challenge_2"]},
    }
    return cfg

SCORING_CONFIG = build_scoring_config()


# Build reverse scoring config
def build_reverse_maps():
    robot_challenge_task = {
        "widowx": {
            "challenge_1": set(WIDOWX_TASKS["widowx"]["challenge_1"]),
            "challenge_2": set(WIDOWX_TASKS["widowx"]["challenge_2"].keys()),
        },
        "google_robot": {
            "challenge_1": set(GOOGLE_ROBOT_TASKS["google_robot"]["challenge_1"]),
            "challenge_2": set(GOOGLE_ROBOT_TASKS["google_robot"]["challenge_2"].keys()),
        },
    }
    task_to_robot_challenge = {}
    for robot, ch_map in robot_challenge_task.items():
        for ch, tasks in ch_map.items():
            for t in tasks:
                task_to_robot_challenge[t] = (robot, ch)
    return task_to_robot_challenge, robot_challenge_task

TASK_TO_ROBOT_CHALLENGE, ROBOT_CHALLENGE_TASK = build_reverse_maps()


# Utils functions
def ensure_weights_sum_to_one(weights: OrderedDict):
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


def pick_policy(policy_setup: str) -> str:
    s = (policy_setup or "").strip().lower()
    if s in ("widowx", "google_robot", "both"):
        return s
    raise ValueError("`--policy-setup` must be one of 'widowx', 'google_robot', or 'both'.")


def load_results(json_file: str):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []
    return data


def main():
    args = parse_args()

    base_path = os.path.abspath(args.result_path)
    json_file = os.path.join(base_path, "results.json")
    scores_csv = os.path.join(base_path, "scores.csv")
    metrics_json = os.path.join(base_path, "metrics.json")

    # Set scoring config
    policy = pick_policy(args.policy_setup)
    scoring_config = SCORING_CONFIG[policy]
    c1_tasks = set(scoring_config.get("challenge_1", []))
    c2_task2rubric = scoring_config.get("challenge_2", {})

    for rubric in c2_task2rubric.values():
        ensure_weights_sum_to_one(rubric)

    # Load json file
    data = load_results(json_file)

    # Accumulators
    score_list = []
    task_sum, task_cnt = defaultdict(float), defaultdict(int)
    challenge_sum, challenge_cnt = defaultdict(float), defaultdict(int)
    robot_sum, robot_cnt = defaultdict(float), defaultdict(int)

    # Start scoring
    allowed_tasks = c1_tasks.union(set(c2_task2rubric.keys()))
    for task in data:
        task_name = task.get("task_name", "")
        if task_name not in allowed_tasks:
            continue

        # Get robot type and challenge type
        robot, ch_type = TASK_TO_ROBOT_CHALLENGE.get(task_name, (None, None))
        if robot is None or ch_type is None:
            continue

        episodes = task.get("episodes", [])

        # Challenge 1
        if task_name in c1_tasks:
            for trial in episodes:
                trial_id = trial.get("trial_id")
                final_status = str(trial.get("final_status", "")).lower()
                success = (final_status == "success")
                score = 1.0 if success else 0.0

                # Details
                score_list.append(
                    {
                        "task_name": task_name,
                        "trial_id": trial_id,
                        "score": score,
                    }
                )

                # Accumulate
                task_sum[task_name] += score
                task_cnt[task_name] += 1
                challenge_sum[ch_type] += score
                challenge_cnt[ch_type] += 1
                robot_sum[robot] += score
                robot_cnt[robot] += 1

            continue

        # Challenge 2
        if task_name in c2_task2rubric:
            rubrics_score = c2_task2rubric[task_name]
            for trial in episodes:
                trial_id = trial.get("trial_id")
                episode_stats = trial.get("episode_stats", {})
                score = 0.0
                for subtask, weight in rubrics_score.items():
                    status = episode_stats.get(subtask, {}).get("status", False)
                    if not status:
                        break
                    score += weight

                # Details
                score_list.append(
                    {
                        "task_name": task_name,
                        "trial_id": trial_id,
                        "score": score,
                    }
                )

                # Accumulate
                task_sum[task_name] += score
                task_cnt[task_name] += 1
                challenge_sum[ch_type] += score
                challenge_cnt[ch_type] += 1
                robot_sum[robot] += score
                robot_cnt[robot] += 1

            continue

    # Save scores.csv
    cols = ["task_name", "trial_id", "score"]
    os.makedirs(os.path.dirname(scores_csv), exist_ok=True)
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(score_list)

    # Compute & Save metrics.json
    metrics = {
        "tasks": {},
        "challenges": {},
        "robots": {},
    }

    for t in sorted(task_sum.keys()):
        mean = task_sum[t] / task_cnt[t] if task_cnt[t] else 0.0
        metrics["tasks"][t] = {"mean_score": round(mean, 6), "episodes": task_cnt[t]}

    for ch in ["challenge_1", "challenge_2"]:
        mean = challenge_sum[ch] / challenge_cnt[ch] if challenge_cnt[ch] else 0.0
        metrics["challenges"][ch] = {"mean_score": round(mean, 6), "episodes": challenge_cnt[ch]}

    robots = ["widowx", "google_robot"] if policy == "both" else [policy]
    for r in robots:
        mean = robot_sum[r] / robot_cnt[r] if robot_cnt[r] else 0.0
        metrics["robots"][r] = {"mean_score": round(mean, 6), "episodes": robot_cnt[r]}

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Scores saved to {scores_csv}, total {len(score_list)} rows.")
    print(f"Metrics saved to {metrics_json}.")


if __name__ == "__main__":
    main()