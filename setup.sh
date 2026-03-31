#!/bin/bash

set -e

echo "[INFO] Creating dataset directory structure..."
mkdir -p datasetCTU/b42

echo "[INFO] Downloading CTU-13 B42 dataset files..."
cd datasetCTU/b42

curl -L -C - -O "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/capture20110810.truncated.pcap.bz2"
curl -L -C - -O "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/capture20110810.binetflow.2format"
curl -L -C - -O "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/README.md"

cd ../../

echo "[INFO] Setting up Python virtual environment..."
python -m venv .venv

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[SUCCESS] Setup complete! The dataset is downloaded and dependencies are installed."
echo "-> To activate the environment in your current shell session, run: source .venv/bin/activate"
