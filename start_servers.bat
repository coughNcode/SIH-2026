@echo off
REM DepthWizard Quick Start — runs both servers
REM Usage: double-click this file from the repo root

cd /d "%~dp0"

echo [1/2] Starting Python inference API on port 8000...
start "DepthWizard Python API" cmd /k "src\dsm_model\.venv\Scripts\python.exe api\server.py"

timeout /t 3 /nobreak > nul

echo [2/2] Starting C++ backend on port 8080...
start "DepthWizard C++ Server" cmd /k "cd backend && depthwizard_server.exe 8080"

timeout /t 2 /nobreak > nul

echo.
echo DepthWizard is running!
echo   Frontend:   http://localhost:8080
echo   Python API: http://localhost:8000
echo   Health:     http://localhost:8080/api/health
echo   Metrics:    http://localhost:8000/metrics
echo.
start http://localhost:8080
