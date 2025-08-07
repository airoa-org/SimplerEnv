
```bash
uv venv -p 3.10 policies/lerobotpi/.venv


# git clone https://github.com/huggingface/lerobot.git

# uv pip install -e .
source $(pwd)/policies/lerobotpi/.venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install lerobot[pi0]==0.3.2 
uv pip install -e .
uv pip install pytest

huggingface-cli login
bash scripts/run_lerobotpifast.sh 

```