import wandb, datetime
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.tune.utils import flatten_dict


PROJECT_NAME = "simpler-env-rllib"
ENTITY_NAME = "<your_wandb_entity>"


class WandbCallback(RLlibCallback):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._writer = None

    def on_algorithm_init(self, *, algorithm, **kw):
        cfg = algorithm.config.to_dict() if hasattr(algorithm.config, "to_dict") else algorithm.config
        self._writer = wandb.init(
            project=PROJECT_NAME,
            entity=ENTITY_NAME,
            name=algorithm.logdir.split("/")[-2],
            sync_tensorboard=True,
            config=cfg,
            save_code=True,
        )

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        training_iteration = result["training_iteration"]
        log_dict = {
            "metrics/episode_return_max" : result["env_runners"]["episode_return_max"],
            "metrics/episode_return_min" : result["env_runners"]["episode_return_min"],
            "metrics/episode_return_mean" : result["env_runners"]["episode_return_mean"],
            "metrics/episode_len_max": result["env_runners"]["episode_len_max"],
            "metrics/episode_len_min": result["env_runners"]["episode_len_min"],
            "metrics/episode_len_mean": result["env_runners"]["episode_len_mean"],

            "steps/num_episodes": result["env_runners"]["num_episodes"],
            "steps/num_env_steps_sampled_lifetime": result["env_runners"]["num_env_steps_sampled_lifetime"],

            "loss/mean_kl_loss": result["learners"]["default_policy"]["mean_kl_loss"],
            "loss/vf_loss": result["learners"]["default_policy"]["vf_loss"],
            "loss/vf_loss_unclipped": result["learners"]["default_policy"]["vf_loss_unclipped"],
            "loss/policy_loss": result["learners"]["default_policy"]["policy_loss"],
            "loss/entropy": result["learners"]["default_policy"]["entropy"],
            "loss/total_loss": result["learners"]["default_policy"]["total_loss"],
            "loss/vf_explained_var": result["learners"]["default_policy"]["vf_explained_var"],
            "loss/curr_entropy_coeff": result["learners"]["default_policy"]["curr_entropy_coeff"],
            "loss/curr_kl_coeff": result["learners"]["default_policy"]["curr_kl_coeff"],
            "loss/gradients_default_optimizer_global_norm": result["learners"]["default_policy"]["gradients_default_optimizer_global_norm"],
            "loss/diff_num_grad_updates_vs_sampler_policy": result["learners"]["default_policy"]["diff_num_grad_updates_vs_sampler_policy"],
            "loss/module_train_batch_size_mean": result["learners"]["default_policy"]["module_train_batch_size_mean"],
        }
        wandb.log(log_dict, step=training_iteration)
