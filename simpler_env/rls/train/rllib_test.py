import time
from datetime import datetime

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import  PPOConfig

import simpler_env
from simpler_env.rls.envs import make_env
from simpler_env.rls.utils import make_logger


def main():
    NUM_GPUS = 1
    LOG_DIR = "/home/user_00029_25b505/SimplerEnv/results/rllib_test"

    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")

    ray.init(
        num_gpus=NUM_GPUS,
        include_dashboard = False,
        ignore_reinit_error = True,
    )

    register_env(
        "SimplerEnv",
        make_env,
    )

    config = (
        PPOConfig()
        .environment(
            env="SimplerEnv", 
            env_config={"env_id": "google_robot_pick_coke_can", "simpler_env_rgb_observation_wrapper": True},
        )
        .framework("torch")
        .resources(
            num_learner_workers=NUM_GPUS,
            num_gpus_per_learner_worker=1,
        )
        .env_runners(num_env_runners=0)
        .rl_module(
            model_config={
                "conv_filters": [
                    [32, [8, 8], 4],
                    [64, [4, 4], 2],
                    [64, [3, 3], 2],
                ],
                "conv_activation": "relu",
            }
        )
    )

    algo = config.build_algo(
        logger_creator=make_logger(f"{LOG_DIR}/{timestr}"),
    )
    for i in range(100+1):
        s = time.time()
        print(f"iter {i:04d}: Training...")
        result = algo.train()
        # dict_keys(['timers', 'env_runners', 'learners', 'num_training_step_calls_per_iteration', 'num_env_steps_sampled_lifetime', 'fault_tolerance', 'env_runner_group', 'num_env_steps_sampled_lifetime_throughput', 'done', 'training_iteration', 'trial_id', 'date', 'timestamp', 'time_this_iter_s', 'time_total_s', 'pid', 'hostname', 'node_ip', 'config', 'time_since_restore', 'iterations_since_restore', 'perf'])
        print(f"TIME: {(time.time() - s)/60:.2f} [min]")

        if (i+1) % 50 == 0:
            checkpoint = algo.save(f"{LOG_DIR}/{timestr}")
            print("saved to", checkpoint)

    algo.stop()
    ray.shutdown()
    print(f"fin!")


if __name__ == "__main__":
    main()
