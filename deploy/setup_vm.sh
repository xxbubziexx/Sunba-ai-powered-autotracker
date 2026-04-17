#!/usr/bin/env bash
# Sunba Autotrack — Ubuntu 24.04 VM setup
# Run from the project root: bash deploy/setup_vm.sh
set -euo pipefail

VENV="venv39"

echo "[1/3] Installing Python 3.9 and system deps..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -qq
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev \
    ffmpeg libgl1 libglib2.0-0

echo "[2/3] Creating venv and installing Python packages..."
python3.9 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install \
    "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl" \
    "https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_x86_64.whl"
"$VENV/bin/pip" install "numpy<2.0" opencv-python-headless ffpyplayer

echo "[3/3] Installing Coral Edge TPU runtime..."
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] \
https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update -qq
sudo apt-get install -y libedgetpu1-std

echo ""
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Run with:             python main.py --headless"
