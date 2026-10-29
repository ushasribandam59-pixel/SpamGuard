@echo off
REM One-time setup for Windows: creates a virtual environment and installs dependencies.
python -m venv venv || goto :err
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Start the app with run.bat
exit /b 0
:err
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
exit /b 1
