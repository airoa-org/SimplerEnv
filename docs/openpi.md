

```bash
git clone --recurse-submodules https://github.com/airoa-org/hsr_openpi.git
docker build -t simpler_env .


# conda install -c conda-forge libgl glib libvulkan-loader vulkan-tools vulkan-headers
# pip install uv

git clone https://github.com/airoa-org/hsr_openpi.git
cd hsr_openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cd ..
source $(pwd)/hsr_openpi/.venv/bin/activate
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"


# ImportError: libvulkan.so.1: cannot open shared object file: No such file or directory

python policies/openpi/openpi_model.py

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs ${coke_can_option};
```


