:: spaudible.bat
@echo off
setlocal enabledelayedexpansion

:: Already set up? Run directly (fast path)
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe main.py
    if errorlevel 1 pause
    exit /b 0
)

:: First-time setup
echo ==========================================
echo Spaudible First-Time Setup (Windows)
echo ==========================================

:: Check for UV in PATH
where uv >nul 2>&1
if %errorlevel% == 0 (
    set UV_CMD=uv
) else (
    echo Downloading UV (Python project manager)...
    curl -L -o uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip 2>nul
    if not exist uv.zip (
        powershell -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile uv.zip"
    )
    powershell -Command "Expand-Archive -Path uv.zip -DestinationPath . -Force"
    del uv.zip
    set UV_CMD=uv.exe
)

echo Installing Python 3.12...
%UV_CMD% python install 3.12 --quiet

echo Creating environment...
%UV_CMD% venv --python 3.12

echo Installing dependencies (5-10 minutes)...
%UV_CMD% pip install

echo Launching...
%UV_CMD% run main.py
if errorlevel 1 pause
