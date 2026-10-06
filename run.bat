@echo off
title FedSure Mutual Enterprise Platform
echo ========================================================
echo   FedSure Mutual - Federated Insurance Enterprise System
echo ========================================================
echo.
echo Starting FedSure Mutual Platform on http://localhost:8501 ...
echo.
python -m streamlit run app.py --server.port 8501 --server.headless false
pause
