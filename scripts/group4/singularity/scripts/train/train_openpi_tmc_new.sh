# Load shared environment variables
source "$(dirname "$0")/env.sh"

# Create the Hugging Face cache directory if it does not exist
cache_dir="$HOME/.cache/huggingface/lerobot/data"
mkdir -p "$cache_dir" || { echo "Failed to create cache directory: $cache_dir"; exit 1; }

# Copy processed data to the cache directory
cp -r /opt/processed/* "$cache_dir" || { echo "Failed to copy processed data to $cache_dir"; exit 1; }

# Sync the repository while skipping Git LFS smudge
GIT_LFS_SKIP_SMUDGE=1 uv sync || { echo "Failed to sync repository"; exit 1; }

# Compute normalization statistics
# Use GPU 0 for this default operation
cd "$OPENPI_DIR_SIF" || { echo "Failed to change directory to $OPENPI_DIR_SIF"; exit 1; }
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run "$OPENPI_DIR_SIF/scripts/compute_norm_stats.py" --config-name pi0_hsr_low_mem_finetune || { echo "Failed to compute normalization statistics"; exit 1; }

# Train the model
# Use GPU 0 and limit memory usage to 90% to avoid OOM errors
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run "$OPENPI_DIR_SIF/scripts/train.py" pi0_hsr_low_mem_finetune --exp-name=my_experiment --overwrite || { echo "Failed to train the model"; exit 1; }