#!/bin/bash
# GRSI installation script
# This script sets up a virtual environment, installs dependencies,
# initializes submodules, and downloads pretrained models.

set -e  # exit immediately if any command fails

echo "=== Setting up environment for GRSI reproducibility ==="

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize submodules
git submodule update --init --recursive

# Download pretrained models
bash ./submodules/pifuhd/scripts/download_trained_model.sh

echo "=== Installation complete. ==="
