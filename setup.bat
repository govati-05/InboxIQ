@echo off
REM InboxIQ - one-time setup on Windows: creates a virtual environment,
REM installs dependencies and trains the ML models for your installed scikit-learn.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python was not found. Install Python 3.10+ from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv || (pause & exit /b 1)
)
echo Installing dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt || (pause & exit /b 1)
echo Training ML models...
venv\Scripts\python.exe src\train_models.py || (pause & exit /b 1)
echo.
echo Setup complete. Double-click run.bat to start InboxIQ.
pause
