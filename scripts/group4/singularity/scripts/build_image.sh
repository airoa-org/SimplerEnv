# Load shared environment variables
source "$(dirname "$0")/env.sh"

# Define directories for using the server memory
export SINGULARITY_TMPDIR="$CODEBASE_DIR/singularity/tmp_build"
export SINGULARITY_CACHEDIR="$CODEBASE_DIR/singularity/singularity_cache"
# Create temporary and cache directories if it does not exist
mkdir -p "$SINGULARITY_TMPDIR" || { echo "Failed to create SINGULARITY_TMPDIR"; exit 1; }
mkdir -p "$SINGULARITY_CACHEDIR" || { echo "Failed to create SINGULARITY_CACHEDIR"; exit 1; }

# Load necessary modules
module load singularitypro || { echo "Failed to load singularitypro module"; exit 1; }
module load cuda || { echo "Failed to load cuda module"; exit 1; }

# Check if the directory exists
if [ ! -d "$CODEBASE_DIR/singularity" ]; then
    echo "Error: Directory $CODEBASE_DIR/singularity does not exist."
    exit 1
fi

# Navigate to the singularity directory
cd "$CODEBASE_DIR/singularity" || { echo "Failed to change directory to $CODEBASE_DIR/singularity"; exit 1; }

# Create the SIF directory if it does not exist
mkdir -p sif || { echo "Failed to create sif directory"; exit 1; }

# Check if the SIF file already exists
if [ -f "sif/hsr_openpi_simplerenv.sif" ]; then
    echo "SIF file already exists. Skipping build."
else
    # Build the Singularity image
    singularity build --fakeroot --force sif/hsr_openpi_simplerenv.sif def/hsr_openpi.def || { echo "Error: Singularity build failed"; exit 1; }
fi
