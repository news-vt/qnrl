#!/bin/bash

####
# Run training on a given configuration multiple times using a different random seed for each run.
####

NUM_RUNS=10
RANDOM=0

# CONFIG_FILE="./configs/c51_CartPole-v1_0.py"
CONFIG_FILE="$1"

SCRIPT="uv run python -m qnrl.cli train --config $CONFIG_FILE"
for i in $(seq 1 $NUM_RUNS); do
    set -x # Enable echoing commands
    $SCRIPT --seed "$RANDOM"
    set +x # Disable echoing commands
done