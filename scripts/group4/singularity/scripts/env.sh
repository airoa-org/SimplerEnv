# Base repositorie
export CODEBASE_DIR="/home/group_25b505/group_4/members/kuramitsu/geniac25_team4_codebase/models/SimplerEnv/scripts/group4"

# hsr_openpi
export OPENPI_DIR="$CODEBASE_DIR/../../../hsr_openpi"
export OPENPI_DIR_SIF="/root/workspace/hsr_openpi"

# SimplerEnv
export SIMPLER_ENV_DIR="$CODEBASE_DIR/../../../SimplerEnv"
export SIMPLER_ENV_DIR_SIF="/root/workspace/SimplerEnv"
export SINGULARITY_DIR="$CODEBASE_DIR/singularity"
export SIF_DIR="$SINGULARITY_DIR/sif"
export DEF_DIR="$SINGULARITY_DIR/def"
export SIF_FILE="$SIF_DIR/hsr_openpi_simplerenv.sif"
export DEF_FILE="$DEF_DIR/hsr_openpi.def"

# Datasets
export GROUP_DATA_DIR="/home/group_25b505/group_4/datasets"

# Scratch/caches
export HF_HOME=/scratch/$USER/huggingface
export PIP_CACHE_DIR=/scratch/$USER/pip_cache
export UV_CACHE_DIR=/scratch/$USER/uv_cache
export SINGULARITY_TMPDIR=/scratch/$USER/singularity_tmp
export SINGULARITY_CACHEDIR=/scratch/$USER/singularity_cache
export CHECKPOINT_DIR=/scratch/$USER/checkpoints
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/openpi

# Ensure dirs exist
mkdir -p $HF_HOME $PIP_CACHE_DIR $UV_CACHE_DIR $SINGULARITY_TMPDIR \
         $SINGULARITY_CACHEDIR $CHECKPOINT_DIR $UV_PROJECT_ENVIRONMENT