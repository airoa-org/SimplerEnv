# airoa-SimplerEnv-benchmark

HSR以外にもシングルアームロボットに対してどのくらいモデルが有効かを評価するためのベンチマーク
## スコアの計算方法

**ロバストスコア (Robust Score)**
このスコアは、モデルの**平均性能**と**安定性**の両方を評価する。

$$\text{Robust Score} = \max(0, \mu - k \cdot \sigma)$$

ここで、

  * $\mu$ (ミュー) は、評価項目全体の**平均成功率**。
  * $\sigma$ (シグマ) は、各評価項目の成功率の**標準偏差**。これは性能のばらつき（不安定さ）を示す。
  * $k$ は**ペナルティ係数**。標準偏差がスコアに与える影響の大きさを調整する。今回は `0.5` に設定し、ばらつきに対して中程度のペナルティを課す。

## スコア提出、算出、モデルのアップロード方法

### 0. 環境構築
```bash
# こっちをやるとmaniskill2のバージョンがずれてしまう
# git clone https://github.com/airoa-org/SimplerEnv.git --recurse-submodules

git clone https://github.com/airoa-org/SimplerEnv.git
git checkout origin/benchmark
git submodule update --init --recursive
cd SimplerEnv
docker build -t simpler_env .
```

このdocker内にあなたのモデルの推論環境を構築してください。
uvを用いた例
```bash
git clone https://github.com/airoa-org/<your_group_name>
cd <your_group_name>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..
source $(pwd)/<your_group_name>/.venv/bin/activate
# install simpler_env to your model environment
uv pip install "numpy<2.0"
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"
uv pip install mediapy
```

### 1. ポリシーの実装
scripts/[your_group_name]/evaluate_fractal.pyに実装
scripts/[your_group_name]/README.mdにあなたのポリシーの説明を記載
を行う
scripts/[your_group_name]/*以外のファイルは変更しないでください


実装方法
`AiroaBasePolicy`のインターフェースとあなたのモデルを合わせる。
そのために`AiroaBasePolicy`を継承したクラスを作成し、`step`と`reset`、`terminate_episode`を実装する。
```python
import argparse
from typing import Dict

import numpy as np

from simpler_env.evaluation.adapter import AiroaToSimplerFractalAdapter
from simpler_env.evaluation.scores import run_comprehensive_evaluation
from simpler_env.policies.base import AiroaBasePolicy


class OpenpiToAiroaPolicy(AiroaBasePolicy):
    def __init__(self, policy):
        self.policy = policy

    def step(self, obs: Dict) -> Dict:
        outputs = self.policy.infer(obs)
        outputs["terminate_episode"] = np.zeros(outputs["actions"].shape[0])
        return outputs

    def reset(self) -> None:
        self.policy.reset()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive ManiSkill2 Evaluation")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path

    policy = OpenpiToAiroaPolicy(
        policy=...
    )

    env_policy = AiroaToSimplerFractalAdapter(policy=policy)

    print("Policy initialized. Starting evaluation...")

    final_scores = run_comprehensive_evaluation(env_policy=env_policy, ckpt_path=args.ckpt_path)

    print("\nEvaluation finished.")
    print(f"Final calculated scores: {final_scores}")

```

なお、それぞれの引数は以下のような値を取る
```python
outputs
{'actions': array([ 0.07858976,  0.01038913, -0.01314185, -0.12083581, -0.04212694,
        0.01047405, -0.47187362]), 'terminate_episode': array([0., 0., 0., 0., 0., 0., 0.])}
special variables
function variables
'actions' =
array([ 0.07858976,  0.01038913, -0.01314185, -0.12083581, -0.04212694,
        0.01047405, -0.47187362])
'terminate_episode' =
array([0., 0., 0., 0., 0., 0., 0.])
len() =
2
outputs["actions"]
array([ 0.07858976,  0.01038913, -0.01314185, -0.12083581, -0.04212694,
        0.01047405, -0.47187362])
outputs["actions"].shape
(7,)
obs
{'image': array([[[ 55,  35,  24],
        [ 55,  35,  24],
        [ 55,  35,  24],
        .....],
        [170, 133, 100]]], dtype=uint8), 'prompt': 'pick coke can', 'state': array([ 0.6511007 , -0.1227724 ,  1.05364919, -0.39264132,  0.79223414,
        0.22855263,  0.40738379,  0.82691604])}
obs["image"].shape
(512, 640, 3)
```

### 2. スコアの計算

```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/<your_group_name>/evaluate_fractal.py --ckpt-path <your_checkpoint_path>
```
この処理にはかなり時間がかかる。


### 3. モデルのアップロード

モデルをwasabiにアップロードする
Naming rule: `<checkpoint_date>_<checkpoint_name>`
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Get general summary
aws s3 ls --summarize --recursive s3://airoa-fm-development-competition/<your_group_folder>/ --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
# List objects
aws s3 ls s3://airoa-fm-development-competition/<your_group_folder>/ --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
# Copy object local to wasabi
aws s3 cp testfile s3://airoa-fm-development-competition/<your_group_folder>/ --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
# Copy object wasabi to local
aws s3 cp s3://airoa-fm-development-competition/<your_group_folder>/testfile ./ --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
```



# SimplerEnv: Simulated Manipulation Policy Evaluation Environments for Real Robot Setups

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simpler-env/SimplerEnv/blob/main/example.ipynb)

![](./images/teaser.png)

Significant progress has been made in building generalist robot manipulation policies, yet their scalable and reproducible evaluation remains challenging, as real-world evaluation is operationally expensive and inefficient. We propose employing physical simulators as efficient, scalable, and informative complements to real-world evaluations. These simulation evaluations offer valuable quantitative metrics for checkpoint selection, insights into potential real-world policy behaviors or failure modes, and standardized setups to enhance reproducibility.

This repository's code is based in the [SAPIEN](https://sapien.ucsd.edu/) simulator and the CPU based [ManiSkill2](https://maniskill2.github.io/) benchmark. We have also integrated the Bridge dataset environments into ManiSkill3, which offers GPU parallelization and can run 10-15x faster than the ManiSkill2 version. For instructions on how to use the GPU parallelized environments and evaluate policies on them, see: https://github.com/simpler-env/SimplerEnv/tree/maniskill3

This repository encompasses 2 real-to-sim evaluation setups:
- `Visual Matching` evaluation: Matching real & sim visual appearances for policy evaluation by overlaying real-world images onto simulation backgrounds and adjusting foreground object and robot textures in simulation.
- `Variant Aggregation` evaluation: creating different sim environment variants (e.g., different backgrounds, lightings, distractors, table textures, etc) and averaging their results.

We hope that our work guides and inspires future real-to-sim evaluation efforts.

- [SimplerEnv: Simulated Manipulation Policy Evaluation Environments for Real Robot Setups](#simplerenv-simulated-manipulation-policy-evaluation-environments-for-real-robot-setups)
  - [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Current Environments](#current-environments)
  - [Compare Your Policy Evaluation Approach to SIMPLER](#compare-your-policy-evaluation-approach-to-simpler)
  - [Code Structure](#code-structure)
  - [Adding New Policies](#adding-new-policies)
  - [Adding New Real-to-Sim Evaluation Environments and Robots](#adding-new-real-to-sim-evaluation-environments-and-robots)
  - [Full Installation (RT-1 and Octo Inference, Env Building)](#full-installation-rt-1-and-octo-inference-env-building)
    - [RT-1 Inference Setup](#rt-1-inference-setup)
    - [Octo Inference Setup](#octo-inference-setup)
  - [Troubleshooting](#troubleshooting)
  - [Citation](#citation)

## Getting Started

Follow the [Installation](#installation) section to install the minimal requirements for our environments. Then you can run the following minimal inference script with interactive python. The scripts creates prepackaged environments for our `visual matching` evaluation setup.

```python
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

env = simpler_env.make('google_robot_pick_coke_can')
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

done, truncated = False, False
while not (done or truncated):
   # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
   # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
   image = get_image_from_maniskill2_obs_dict(env, obs)
   action = env.action_space.sample() # replace this with your policy inference
   obs, reward, done, truncated, info = env.step(action) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
   new_instruction = env.get_language_instruction()
   if new_instruction != instruction:
      # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
      instruction = new_instruction
      print("New Instruction", instruction)

episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
```

Additionally, you can play with our environments in an interactive manner through [`ManiSkill2_real2sim/mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py`](https://github.com/simpler-env/ManiSkill2_real2sim/blob/main/mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py). See the script for more details and commands.

## Installation

Prerequisites:
- CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
- An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Create an anaconda environment:
```
conda create -n simpler_env python=3.10 (3.10 or 3.11)
conda activate simpler_env
```

Clone this repo:
```
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install -e .
```

**If you'd like to perform evaluations on our provided agents (e.g., RT-1, Octo), or add new robots and environments, please additionally follow the full installation instructions [here](#full-installation-rt-1-and-octo-inference-env-building).**


## Examples

- Simple RT-1 and Octo evaluation script on prepackaged environments with visual matching evaluation setup: see [`simpler_env/simple_inference_visual_matching_prepackaged_envs.py`](https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/simple_inference_visual_matching_prepackaged_envs.py).
- Colab notebook for RT-1 and Octo inference: see [this link](https://colab.research.google.com/github/simpler-env/SimplerEnv/blob/main/example.ipynb).
- Environment interactive visualization and manual control: see [`ManiSkill2_real2sim/mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py`](https://github.com/simpler-env/ManiSkill2_real2sim/blob/main/mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py)
- Policy inference scripts to reproduce our Google Robot and WidowX real-to-sim evaluation results with sweeps over object / robot poses and advanced loggings. These contain both visual matching and variant aggregation evaluation setups along with RT-1, RT-1-X, and Octo policies. See [`scripts/`](https://github.com/simpler-env/SimplerEnv/tree/main/scripts).
- Real-to-sim evaluation videos from running `scripts/*.sh`: see [this link](https://huggingface.co/datasets/xuanlinli17/simpler-env-eval-example-videos/tree/main).

## Current Environments

To get a list of all available environments, run:
```
import simpler_env
print(simpler_env.ENVIRONMENTS)
```

| Task Name | ManiSkill2 Env Name | Image (Visual Matching) |
| ----------- | ----- | ----- |
| google_robot_pick_coke_can | GraspSingleOpenedCokeCanInScene-v0 | <img src="./images/example_visualization/google_robot_coke_can_visual_matching.png" width="128" height="128" > |
| google_robot_pick_object | GraspSingleRandomObjectInScene-v0 | <img src="./images/example_visualization/google_robot_pick_random_object.png" width="128" height="128" > |
| google_robot_move_near | MoveNearGoogleBakedTexInScene-v1 | <img src="./images/example_visualization/google_robot_move_near_visual_matching.png" width="128" height="128" > |
| google_robot_open_drawer | OpenDrawerCustomInScene-v0 | <img src="./images/example_visualization/google_robot_open_drawer_visual_matching.png" width="128" height="128" > |
| google_robot_close_drawer | CloseDrawerCustomInScene-v0 | <img src="./images/example_visualization/google_robot_close_drawer_visual_matching.png" width="128" height="128" > |
| google_robot_place_in_closed_drawer | PlaceIntoClosedDrawerCustomInScene-v0 | <img src="./images/example_visualization/google_robot_put_apple_in_closed_top_drawer.png" width="128" height="128" > |
| widowx_spoon_on_towel    | PutSpoonOnTableClothInScene-v0                | <img src="./images/example_visualization/widowx_spoon_on_towel_visual_matching.png" width="128" height="128" > |
| widowx_carrot_on_plate   | PutCarrotOnPlateInScene-v0                    | <img src="./images/example_visualization/widowx_carrot_on_plate_visual_matching.png" width="128" height="128" > |
| widowx_stack_cube        | StackGreenCubeOnYellowCubeBakedTexInScene-v0  | <img src="./images/example_visualization/widowx_stack_cube_visual_matching.png" width="128" height="128" > |
| widowx_put_eggplant_in_basket        | PutEggplantInBasketScene-v0  | <img src="./images/example_visualization/widowx_put_eggplant_in_basket_visual_matching.png" width="128" height="128" > |

We also support creating sub-tasks variations such as `google_robot_pick_{horizontal/vertical/standing}_coke_can`, `google_robot_open_{top/middle/bottom}_drawer`, and `google_robot_close_{top/middle/bottom}_drawer`. For the `google_robot_place_in_closed_drawer` task, we use the `google_robot_place_apple_in_closed_top_drawer` subtask for paper evaluations.

By default, Google Robot environments use a control frequency of 3hz, and Bridge environments use a control frequency of 5hz. Simulation frequency is ~500hz.


## Compare Your Policy Evaluation Approach to SIMPLER

We make it easy to compare your offline robot policy evaluation approach to SIMPLER. In [our paper](https://simpler-env.github.io/), we use two metrics to measure the quality of simulated evaluation pipelines: Mean Maximum Rank Violation (MMRV) and the Pearson Correlation Coefficient. Both capture how well the offline evaluations reflect the policy's real-world performance and behaviors during robot rollouts.

To make comparisons easy, we provide our real and SIMPLER evaluation performance for all policies on all tasks. We also provide the corresponding functions for computing the aforementioned metrics we report in the paper.

To compute the corresponding metrics for *your* offline policy evaluation approach `your_sim_eval(task, policy)`, you can use the following snippet:
```
from simpler_env.utils.metrics import mean_maximum_rank_violation, pearson_correlation, REAL_PERF

sim_eval_perf = [
    your_sim_eval(task="google_robot_move_near", policy=p) 
    for p in ["rt-1-x", "octo", ...]
]
real_eval_perf = [
    REAL_PERF["google_robot_move_near"][p] for p in ["rt-1-x", "octo", ...]
]
mmrv = mean_maximum_rank_violation(real_eval_perf, sim_eval_perf)
pearson = pearson_correlation(real_eval_perf, sim_eval_perf)
```

To reproduce the key numbers from our paper for SIMPLER, you can run the [`tools/calc_metrics.py`](tools/calc_metrics.py) script:
```
python3 tools/calc_metrics.py
```

## Code Structure

```
ManiSkill2_real2sim/: the ManiSkill2 real-to-sim environment codebase, which contains the environments, robots, and objects for real-to-sim evaluation.
   data/
      custom/: custom object assets (e.g., coke can, cabinet) and their infos
      hab2_bench_assets/: custom scene assets
      real_inpainting/: real-world inpainting images for visual matching evaluation
      debug/: debugging assets
   mani_skill2_real2sim/
      agents/: robot agents, configs, and controller implementations
      assets/: robot assets such as URDF and meshes
      envs/: environments
      examples/demo_manual_control_custom_envs.py: interactive script for environment visualization and manual
      utils/
   ...
simpler_env/
   evaluation/: real-to-sim evaluator with advanced environment building and logging
      argparse.py: argument parser supporting custom policy and environment building
      maniskill2_evaluator.py: evaluator that supports environment parameter sweeps and advanced logging
   policies/: policy implementations
      rt1/: RT-1 policy implementation
      octo/: Octo policy implementation
   utils/:
      env/: environment building and observation utilities
      debug/: debugging tools for policies and robots
      ...
   main_inference.py: main inference script, taking in args from evaluation.argparse and calling evaluation.maniskill2_evaluator
   simple_inference_visual_matching_prepackaged_envs.py: an independent simple inference script on prepackaged environments, doesn't depend on evaluation/*
tools/
   robot_object_visualization/: tools for visualizing robots and objects when creating new environments
   sysid/: tools for system identification when adding new robots
   calc_metrics.py: tools for summarizing eval results and calculating metrics, such as Mean Maximum Rank Violation (MMRV) and Pearson Correlation
   coacd_process_mesh.py: tools for generating convex collision meshes through CoACD when adding new assets
   merge_videos.py: tools for merging videos into one
   ...
scripts/: example bash scripts for policy inference under our variant aggregation / visual matching evaluation setup,
          with custom environment building and advanced logging; also useful for reproducing our evaluation results
...
```

## Adding New Policies

If you want to use existing environments for evaluating new policies, you can keep `./ManiSkill2_real2sim` as is.

1. Implement new policy inference scripts in `simpler_env/policies/{your_new_policy}`, following the examples for RT-1 (`simpler_env/policies/rt1`) and Octo (`simpler_env/policies/octo`) policies.
2. You can now use `simpler_env/simple_inference_visual_matching_prepackaged_envs.py` to perform policy evaluations in simulation.
   - If the policy behaviors deviate a lot from those in the real-world, you can write similar scripts as in `simpler_env/utils/debug/{policy_name}_inference_real_video.py` to debug the policy behaviors. The debugging script performs policy inference by feeding real eval video frames into the policy. If the policy behavior still deviates significantly from real, this may suggest that policy actions are processed incorrectly into the simulation environments. Please double check action orderings and action spaces.
3. If you'd like to perform customized evaluations,
   - Modify a few lines in `simpler_env/main_inference.py` to support your new policies.
   - Add policy inference scripts in `scripts/` with customized configs.
   - Optionally, modify the scripts in `tools/calc_metrics.py` to calculate the real-to-sim evaluation metrics for your new policies.


## Adding New Real-to-Sim Evaluation Environments and Robots

We provide a step-by-step guide to add new real-to-sim evaluation environments and robots in [this README](ADDING_NEW_ENVS_ROBOTS.md)


## Full Installation (RT-1 and Octo Inference, Env Building)

If you'd like to perform evaluations on our provided agents (e.g., RT-1, Octo), or add new robots and environments, please follow the full installation instructions below.

```
sudo apt install ffmpeg
```

```
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support
```

Install simulated annealing utils for system identification:
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

### RT-1 Inference Setup

Download RT-1 Checkpoint:
```
# First, install gsutil following https://cloud.google.com/storage/docs/gsutil_install

# Make a checkpoint dir:
mkdir {this_repo}/checkpoints

# RT-1-X
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
unzip rt_1_x_tf_trained_for_002272480_step.zip
mv rt_1_x_tf_trained_for_002272480_step checkpoints
rm rt_1_x_tf_trained_for_002272480_step.zip

# RT-1-Converged
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000400120 .
mv rt_1_tf_trained_for_000400120 checkpoints

# RT-1-15%
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000058240 .
mv rt_1_tf_trained_for_000058240 checkpoints

# RT-1-Begin
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000001120 .
mv rt_1_tf_trained_for_000001120 checkpoints      
```

### Octo Inference Setup

Install Octo:
```
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # or jax[cuda12_pip] if you have CUDA 12

cd {this_repo}
git clone https://github.com/octo-models/octo/
cd octo
git checkout 653c54acde686fde619855f2eac0dd6edad7116b  # we use octo-1.0
pip install -e .
# You don't need to run "pip install -r requirements.txt" inside the octo repo; the package dependencies are already handled in the simpler_env repo
# Octo checkpoints are managed by huggingface, so you don't need to download them manually.
```

If you are using CUDA 12, then to use GPU for Octo inference, you need CUDA version >= 12.2 to satisfy the requirement of Jax; in this case, you can perform a runfile install of the corresponding CUDA (e.g., version 12.3), then set the environment variables whenever you run Octo inference scripts:

`PATH=/usr/local/cuda-12.3/bin:$PATH   LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH   bash scripts/octo_xxx_script.sh`

## Troubleshooting

1. If you encounter issues such as

```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.
Segmentation fault (core dumped)
```

Follow [this link](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to troubleshoot the issue. (Even though the doc points to SAPIEN 3 and ManiSkill3, the troubleshooting section still applies to the current environments that use SAPIEN 2.2 and ManiSkill2).

2. You can ignore the following error if it is caused by tensorflow's internal code. Sometimes this error will occur when running the inference or debugging scripts.

```
TypeError: 'NoneType' object is not subscriptable
```


## Citation

If you find our ideas / environments helpful, please cite our work at
```
@article{li24simpler,
         title={Evaluating Real-World Robot Manipulation Policies in Simulation},
         author={Xuanlin Li and Kyle Hsu and Jiayuan Gu and Karl Pertsch and Oier Mees and Homer Rich Walke and Chuyuan Fu and Ishikaa Lunawat and Isabel Sieh and Sean Kirmani and Sergey Levine and Jiajun Wu and Chelsea Finn and Hao Su and Quan Vuong and Ted Xiao},
         journal = {arXiv preprint arXiv:2405.05941},
         year={2024}
}
```
