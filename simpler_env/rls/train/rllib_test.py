import time
from datetime import datetime

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import  PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import simpler_env
from simpler_env.rls.envs import make_env
from simpler_env.rls.utils import make_logger
from simpler_env.rls.callbacks import WandbCallback
from simpler_env.rls.modules import Pi0PPOTorchRLModule


def main():
    NUM_GPUS = 1
    LOG_DIR = "/home/user_00029_25b505/SimplerEnv/results/rllib_test"
    ENV_ID = "google_robot_pick_coke_can"
    POLICY_SETUP = "google_robot"
    IMAGE_SIZE = [224, 224]
    ACTION_SCALE = 1.0

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
            env_config={
                "env_id": ENV_ID,
                "simpler_env_rgb_observation_wrapper": False,
                "lerobot_pi0_wrapper": True,
                "policy_setup": POLICY_SETUP,
            },
        )
        .framework("torch")
        .resources(
            num_gpus=NUM_GPUS,
            num_learner_workers=NUM_GPUS,
            num_gpus_per_learner_worker=1,
            num_gpus_per_worker=1,
        )
        .learners(num_learners=0, num_gpus_per_learner=1)
        .training(
            train_batch_size=4000, # default: 4000
            num_epochs=30, # default 30
            minibatch_size=128, # default: 128 

        )
        .env_runners(
            num_env_runners=0,
            num_gpus_per_env_runner=1,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=Pi0PPOTorchRLModule,
                model_config={
                    "policy_setup": POLICY_SETUP,
                    "image_size": IMAGE_SIZE,
                    "action_scale": ACTION_SCALE,
                },
            ),
        )
        .callbacks(WandbCallback)
    )

    print(f"config: {config.to_dict()}")

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
            checkpoint = algo.save(f"{LOG_DIR}/{timestr}/checkpoints")
            print("saved to", checkpoint)

    algo.stop()
    ray.shutdown()
    print(f"fin!")


if __name__ == "__main__":
    main()
