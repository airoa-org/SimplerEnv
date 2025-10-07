# Load shared environment variables
source "$(dirname "$0")/env.sh"

# add symbolic link
mkdir -p ~/.cache/huggingface/lerobot/IPEC-COMMUNITY
ln -s /home/group_25b505/dataset/oxe/raw/datasets--IPEC-COMMUNITY--bridge_orig_lerobot/snapshots/0e9d76d07e9df3ea3eba257b2520d4913833fad2 ~/.cache/huggingface/lerobot/IPEC-COMMUNITY/bridge_orig_lerobot

# Sync the repository while skipping Git LFS smudge
GIT_LFS_SKIP_SMUDGE=1 uv sync || { echo "Failed to sync repository"; exit 1; }

# Compute normalization statistics
# Use GPU 0 for this default operation
# ${CUDA_VISIBLE_DEVICES}
cd "$OPENPI_DIR_SIF" || { echo "Failed to change directory to $OPENPI_DIR_SIF"; exit 1; }
CUDA_VISIBLE_DEVICES=2,3 uv run "$OPENPI_DIR_SIF/scripts/compute_norm_stats.py" --config-name pi0_fast_bridge_low_mem_finetune || { echo "Failed to compute normalization statistics"; exit 1; }

# Train the model
# Use GPU 0 and limit memory usage to 90% to avoid OOM errors
CUDA_VISIBLE_DEVICES=2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run "$OPENPI_DIR_SIF/scripts/train.py" pi0_fast_bridge_low_mem_finetune --exp-name=my_experiment --overwrite || { echo "Failed to train the model"; exit 1; }