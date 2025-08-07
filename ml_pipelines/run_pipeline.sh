#!/bin/bash
set -e

# Go to project root (directory containing this script)
cd "$(dirname "$0")"/..

echo "Step 1: Running preprocessing..."
python src/preprocessing.py

echo "Step 2: Training and deploying model..."
python src/train_and_deploy.py

echo "All steps completed successfully."