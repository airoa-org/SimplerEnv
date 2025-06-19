import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.common import envs, policies  # noqa: F401
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from g3_haptics.configs.train import TrainPipelineConfigG3Haptics


@dataclass
class G3EvalPipelineConfig(EvalPipelineConfig):
    env: envs.EnvConfig = None
    cfg_all: TrainPipelineConfigG3Haptics | None = None

    def __post_init__(self):
        super().__post_init__()
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.cfg_all = TrainPipelineConfigG3Haptics.from_pretrained(policy_path, cli_overrides=cli_overrides)


@parser.wrap()
def eval_main(cfg: G3EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making policy.")

    # Calculate train and val episodes
    ds_meta = LeRobotDatasetMetadata(
        cfg.cfg_all.dataset.repo_id, root=cfg.cfg_all.dataset.root, revision=cfg.cfg_all.dataset.revision
    )

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )
    policy.eval()


if __name__ == "__main__":
    init_logging()
    eval_main()
