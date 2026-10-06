# FedSure Mutual Enterprise Platform Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  FedSure Mutual - Federated Insurance Enterprise System" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching web application on http://localhost:8501..." -ForegroundColor Yellow

$pythonCmd = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $pythonCmd) {
    Write-Error "Python was not found on your system PATH."
    exit 1
}

python -m streamlit run app.py --server.port 8501 --server.headless $false
