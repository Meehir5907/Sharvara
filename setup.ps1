$ErrorActionPreference = "Stop"

Write-Host "[INFO] Creating dataset directory structure..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "datasetCTU\b42" | Out-Null

Write-Host "[INFO] Downloading CTU-13 B42 dataset files..." -ForegroundColor Cyan
Set-Location -Path "datasetCTU\b42"

curl.exe -L -C - -O "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/capture20110810.truncated.pcap.bz2"
curl.exe -L -C - -O "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/capture20110810.binetflow.2format"
curl.exe -L -C - -O "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/README.md"

Set-Location -Path "..\.."

Write-Host "[INFO] Setting up Python virtual environment..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "[INFO] Installing dependencies..." -ForegroundColor Cyan

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[SUCCESS] Setup complete! The dataset is downloaded and dependencies are installed." -ForegroundColor Green
Write-Host "-> To activate the environment in your current shell session, run:" -ForegroundColor Yellow
Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
