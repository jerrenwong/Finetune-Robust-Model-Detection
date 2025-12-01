#!/bin/bash

# Script to LLsub all submission scripts matching the pattern scripts/submission_scripts/submit_shard_*.sh

SUBMISSION_DIR="scripts/submission_scripts"

# Check if directory exists
if [ ! -d "$SUBMISSION_DIR" ]; then
    echo "Error: Directory $SUBMISSION_DIR not found."
    exit 1
fi

# Find and LLsub the scripts
for script in "$SUBMISSION_DIR"/submit_shard_*.sh; do
    if [ -f "$script" ]; then
        echo "Submitting $script..."
        LLsub "$script"
    else
        echo "No scripts found matching $SUBMISSION_DIR/submit_shard_*.sh"
        exit 1
    fi
done

echo "All submissions complete."
