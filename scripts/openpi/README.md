
```bash
uv venv -p 3.11 scripts/openpi/.venv

git clone https://github.com/airoa-org/hsr_openpi.git
cd hsr_openpi
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=../scripts/openpi/.venv uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..

source $(pwd)/scripts/openpi/.venv/bin/activate
uv pip install -e .


python scripts/openpi/evaluate_fractal.py --ckpt-path /data/checkpoints/pi0_fractal_low_mem_finetune2/my_experiment/17000
```