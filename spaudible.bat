:: spaudible.bat
@echo off
setlocal enabledelayedexpansion

:: Run directly if the environment has already been set up
if exist ".venv\Scripts\python.exe" (
    echo Launching Spaudible...
    .venv\Scripts\python.exe main.py
    if errorlevel 1 pause
    exit /b 0
)

echo ==========================================
echo Spaudible - First-Time Setup (Windows)
echo ==========================================
echo.

:: Check for UV in PATH
where uv >nul 2>&1
if %errorlevel% == 0 (
    set UV_CMD=uv
) else (
    if not exist "uv.exe" (
        echo Downloading UV (Python project manager)...
        curl -L -o uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip 2>nul
        
        if not exist uv.zip (
            powershell -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile uv.zip"
        )
        
        if not exist uv.zip (
            echo Error: Failed to download UV. Check internet connection.
            pause
            exit /b 1
        )
        
        powershell -Command "Expand-Archive -Path uv.zip -DestinationPath . -Force"
        del uv.zip
        
        :: Handle subdirectory extraction (eg. uv-x86_64-pc-windows-msvc\uv.exe)
        if not exist "uv.exe" (
            for /d %%D in (uv-*) do (
                if exist "%%D\uv.exe" (
                    move "%%D\uv.exe" . >nul 2>&1
                    move "%%D\uvx.exe" . >nul 2>&1
                    rd /s /q "%%D" 2>nul
                )
            )
        )
        
        if not exist "uv.exe" (
            echo Error: UV installation failed. uv.exe not found after extraction.
            pause
            exit /b 1
        )
    )
    set UV_CMD=uv.exe
)

echo Installing Python 3.12 (this may take a moment)...
%UV_CMD% python install 3.12 --quiet
if errorlevel 1 (
    echo Error: Failed to install Python 3.12.
    pause
    exit /b 1
)

echo Creating virtual environment...
%UV_CMD% venv --python 3.12
if errorlevel 1 (
    echo Error: Failed to create virtual environment.
    pause
    exit /b 1
)

echo Installing dependencies (this may take several minutes)...
%UV_CMD% pip install -e .
if errorlevel 1 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Launching Spaudible...
echo ==========================================
%UV_CMD% run main.py
if errorlevel 1 pause
