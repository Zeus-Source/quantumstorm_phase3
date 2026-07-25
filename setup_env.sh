#!/usr/bin/env bash
# setup_env.sh - Setup persistent qBraid environment for QuantumStorm Phase 3.
#
# This script creates a persistent local qBraid environment that survives JLab/pod restarts
# and agent resets, so you never have to reinstall dependencies again.

set -e

ENV_NAME="quantumstorm"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

echo "================================================================================"
echo "         QUANTUMSTORM PERSISTENT QBRAID ENVIRONMENT SETUP"
echo "================================================================================"

# 1. Check if the environment already exists in qBraid registry
echo "Checking if qBraid environment '$ENV_NAME' is registered..."
if qbraid envs list | grep -q "$ENV_NAME"; then
    echo "✓ qBraid environment '$ENV_NAME' already exists."
else
    echo "Creating persistent qBraid environment '$ENV_NAME'..."
    # Create the environment and automatically install requirements
    qbraid envs create --name "$ENV_NAME" --requirements "$REQ_FILE" --yes
fi

echo "================================================================================"
echo "✓ Persistent qBraid Environment '$ENV_NAME' is completely setup and ready!"
echo "================================================================================"
echo "How to use this environment:"
echo "1. JupyterLab UI: Select the 'quantumstorm' kernel from the launcher / kernel picker."
echo "2. Terminal / CLI: Run your scripts directly using the environment's python:"
echo "   /home/jovyan/.qbraid/environments/$ENV_NAME/bin/python scripts/run_benchmark.py"
echo "================================================================================"
