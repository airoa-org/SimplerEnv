
```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=../scripts/openpi/.venv uv sync
source ../scripts/openpi/.venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..

source $(pwd)/scripts/openpi/.venv/bin/activate
uv pip install -e .


huggingface-cli download --resume-download --repo-type model HaomingSong/openpi0-fractal-lora --local-dir /data/checkpoints/openpi0-fractal-lora

python scripts/openpi/evaluate_fractal2.py --ckpt-path /data/checkpoints/openpi0-fractal-lora
CUDA_VISIBLE_DEVICES=1 python scripts/openpi/evaluate_fractal.py --ckpt-path HaomingSong/openpi0-bridge-lora
```


https://huggingface.co/HaomingSong/openpi0-fast-fractal-fft
```python
@dataclasses.dataclass(frozen=True)
class LeRobotFractalDataConfig(DataConfigFactory):
    use_quantile_norm: bool = True

    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    prompt_from_task: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/primary_image": "observation.images.image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                fractal_policy.FractalInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[fractal_policy.FractalOutputs()],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=self.prompt_from_task,
        )

```


https://github.com/DelinQu/SimplerEnv-OpenVLA/issues/24
```python
@dataclasses.dataclass(frozen=True)
class LeRobotBridgeDataConfig(DataConfigFactory):
    use_quantile_norm: bool = True

    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    prompt_from_task: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/primary_image": "observation.images.image_0",
                        # "observation/left_yellow_image": "observation.images.image_1",
                        # "observation/right_blue_image": "observation.images.image_2",
                        # "observation/wirst_image": "observation.images.image_3",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                ### THE FOLLOWING bridge_policy MODULE IS MISSING ###
                bridge_policy.BridgeInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[bridge_policy.BridgeOutputs()],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=self.prompt_from_task,
        )
```


https://github.com/HaomingSong/openpi/blob/bridge/src/openpi/policies/bridge_policy.py
```python
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import torch


def make_bridge_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/primary_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/left_yellow_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/right_blue_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # "observation/wirst_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BridgeInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # NOTE: for bridge dataset at IPEC-COMMUNITY/bridge_orig_lerobot, the state is 8-dim.
        # Get the state. We are padding from 8 to the model action dim.
        # state = data["observation/state"][:8]
        state = torch.zeros(data["observation/state"].shape)
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        primary_image = _parse_image(data["observation/primary_image"])
        # left_yellow_image = _parse_image(data["observation/left_yellow_image"])
        # right_blue_image = _parse_image(data["observation/right_blue_image"])
        # wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "primary_image": primary_image,
                # "left_yellow_image": left_yellow_image,
                # "left_yellow_image": np.zeros_like(primary_image),
                # "right_blue_image": right_blue_image,
                # "right_blue_image": np.zeros_like(primary_image),
                # "wrist_image": wrist_image,
            },
            "image_mask": {
                "primary_image": np.True_,
                # "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                # "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
```