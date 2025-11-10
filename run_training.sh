#!/bin/bash
# Training script runner for Astra Guardian
# This script ensures the virtual environment is activated and PYTHONPATH is set

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please create it first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to include the project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Set library path for M2 MacBook Air (XGBoost)
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Run the training script
echo "Starting training with M2 MacBook Air optimizations..."
echo "PYTHONPATH: $PYTHONPATH"
echo ""

python scripts/train.py

