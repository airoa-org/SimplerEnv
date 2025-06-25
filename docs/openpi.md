

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
uv pip install numpy==1.24.4
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
uv pip install "tensorflow-cpu==2.15.*"


# for debug
uv venv
source .venv/bin/activate
uv pip install numpy==1.24.4
uv pip install -e ./ManiSkill2_real2sim
uv pip install -e .
python tes.py
```


