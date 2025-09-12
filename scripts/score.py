import json
import csv

json_file = "/app/SimplerEnv/results/bridge_lora/results.json"
csv_file = "/app/SimplerEnv/results/bridge_lora/score.csv"

scoring_config = {
    "is_src_obj_grasped": 0.3,
    "consecutive_grasp": 0.3,
    "src_on_target": 0.4
}

# Load json file
try:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = []
except Exception:
    data = []


score_list = []
for task_id, task in enumerate(data):
    if "task1" in task["task_name"]:
        continue
    task_name = task["task_name"]
    trials = task["episodes"]
    for trial in trials:
        trial_id = trial["trial_id"]
        episode_stats = trial["episode_stats"]
        score = 0
        for subtask in scoring_config.keys():
            status = episode_stats.get(subtask, {}).get("status", False)
            if not status:
                break
            score += scoring_config[subtask]
        cur_score_data = {
            "task_id": task_id+1,
            "trial_id": trial_id,
            "score": score
        }
        score_list.append(cur_score_data)
            

# Export as csv file
cols = ["task_id", "trial_id", "score"]
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(score_list)

print(f"Scores saved to {csv_file}, total {len(score_list)} rows.")