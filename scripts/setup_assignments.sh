#!/bin/bash
# scripts/setup_assignments.sh
# Downloads and extracts CS224n assignments for self-study.

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# Configurations
BASE_URL="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/assignments"
ASSIGNMENTS=("a1" "a2" "a3" "a4" "a5")
TARGET_DIRS=(
    "a1_word_vectors"
    "a2_word2vec"
    "a3_dependency_parser"
    "a4_nmt_attention"
    "a5_pretrain_bert"
)

# Ensure script is run from project root
if [ ! -d "assignments" ]; then
    echo "Error: Please run this script from the project root directory."
    echo "Usage: bash scripts/setup_assignments.sh"
    exit 1
fi

echo "=================================================="
echo "         Stanford CS224n Assignment Downloader    "
echo "=================================================="

# Function to download and extract a single assignment
setup_assignment() {
    local idx=$1
    local name=${ASSIGNMENTS[$idx]}
    local target_dir="assignments/${TARGET_DIRS[$idx]}"
    local zip_file="assignments/${name}.zip"
    local url="${BASE_URL}/${name}.zip"

    echo "--------------------------------------------------"
    echo "Processing Assignment $(($idx + 1)): ${name}..."

    # Check if target directory already has content
    if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
        echo "--> Directory $target_dir is already set up. Skipping."
        return 0
    fi

    # Create target directory
    mkdir -p "$target_dir"

    # Download assignment
    echo "--> Downloading ${url}..."
    if curl -s -f -o "$zip_file" "$url"; then
        echo "--> Download successful."
    else
        echo "--> WARNING: Failed to download assignment ${name}.zip from ${url}."
        echo "    It might not have been published yet or the URL changed."
        # Remove target directory if empty
        rmdir "$target_dir" 2>/dev/null || true
        return 0
    fi

    # Extract assignment
    echo "--> Extracting ${zip_file}..."
    
    # Create a temp directory for extraction to inspect the structure
    local temp_dir="assignments/temp_${name}"
    mkdir -p "$temp_dir"
    unzip -q "$zip_file" -d "$temp_dir"

    # Determine internal folder structure of the zip
    # Some zips extract into a subfolder matching the assignment name (e.g. a1/), some do not.
    local extracted_subdirs=("$temp_dir"/*)
    if [ ${#extracted_subdirs[@]} -eq 1 ] && [ -d "${extracted_subdirs[0]}" ]; then
        # Moving contents of the single inner directory to our target directory
        local inner_dir="${extracted_subdirs[0]}"
        mv "$inner_dir"/* "$target_dir"/ 2>/dev/null || mv "$inner_dir"/.??* "$target_dir"/ 2>/dev/null || true
        # Also move hidden files if any
        find "$inner_dir" -mindepth 1 -maxdepth 1 -exec mv -t "$target_dir"/ {} + 2>/dev/null || true
    else
        # Move everything from temp to target
        mv "$temp_dir"/* "$target_dir"/ 2>/dev/null || true
        find "$temp_dir" -mindepth 1 -maxdepth 1 -exec mv -t "$target_dir"/ {} + 2>/dev/null || true
    fi

    # Cleanup temporary structures
    rm -rf "$temp_dir"
    rm -f "$zip_file"

    echo "--> Successfully set up $target_dir"
}

# Run setup for all assignments
for i in "${!ASSIGNMENTS[@]}"; do
    setup_assignment "$i"
done

echo "--------------------------------------------------"
echo "Setup complete! All assignments processed."
echo "=================================================="
