# Load shared environment variables
source "$(dirname "$0")/env.sh"

# Define directories for using the server memory
export SINGULARITY_TMPDIR="$CODEBASE_DIR/singularity/tmp_build"
export SINGULARITY_CACHEDIR="$CODEBASE_DIR/singularity/singularity_cache"
# Create temporary and cache directories if it does not exist
mkdir -p "$SINGULARITY_TMPDIR" || { echo "Failed to create SINGULARITY_TMPDIR"; exit 1; }
mkdir -p "$SINGULARITY_CACHEDIR" || { echo "Failed to create SINGULARITY_CACHEDIR"; exit 1; }

# Load necessary modules
module purge
module load singularitypro || {
    echo "Failed to load singularitypro module"
    exit 1
}
module load cuda || {
    echo "Failed to load cuda module"
    exit 1
}

# Create the destinated model directories if it does not exist
mkdir -p $OPENPI_DIR || {
    echo "Failed to create $OPENPI_DIR directory"
    exit 1
}
mkdir -p $SIMPLER_ENV_DIR || {
    echo "Failed to create $SIMPLER_ENV_DIR directory"
    exit 1
}

singularity exec --fakeroot \
    --bind "$GROUP_DATA_DIR:/opt/processed" \
    --bind /usr/share/vulkan:/usr/share/vulkan \
    --bind /usr/share/glvnd:/usr/share/glvnd \
    --bind /usr/share/nvidia:/usr/share/nvidia\
    --bind "/home/group_25b505:/home/group_25b505" \
    --bind "$OPENPI_DIR:$OPENPI_DIR_SIF" \
    --bind "$SIMPLER_ENV_DIR:$SIMPLER_ENV_DIR_SIF" \
    --bind "$CODEBASE_DIR/singularity/scripts/.bash_aliases:/root/.bash_aliases" \
    --bind "$CODEBASE_DIR/singularity/scripts:$OPENPI_DIR_SIF/scripts_launch" \
    --pwd $OPENPI_DIR_SIF \
    --nv \
    "$CODEBASE_DIR/singularity/sif/hsr_openpi_simplerenv.sif" \
    bash || {
    echo "Failed to start Singularity exec"
    exit 1
}
