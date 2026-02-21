:: spaudible.bat
@echo off
setlocal enabledelayedexpansion

:: Ensure we run from the batch file's directory
cd /d "%~dp0"

:: Download Open Sans font for the GUI if not yet present
if not exist "data\fonts\OpenSans-Regular.ttf" (
    echo Downloading Open Sans font...
    mkdir "data\fonts" 2>nul

    :: Primary: .ttf from GitHub
    curl -L -o "data\fonts\OpenSans-Regular.ttf" "https://github.com/googlefonts/opensans/raw/refs/heads/main/fonts/ttf/OpenSans-Regular.ttf" --silent --fail 2>nul
    :: Also grab SemiBold style for headers
    curl -L -o "data\fonts\OpenSans-SemiBold.ttf" "https://github.com/googlefonts/opensans/raw/refs/heads/main/fonts/ttf/OpenSans-SemiBold.ttf" --silent --fail 2>nul

    if not exist "data\fonts\OpenSans-Regular.ttf" (
        echo Font download failed, trying fallback...
        
        :: Create temporary PowerShell script to handle extraction
        (
            echo try {
            echo     $zipPath = 'data\fonts\opensans_temp.zip'
            echo     $extractPath = 'data\fonts\temp'
            echo     $finalPath = 'data\fonts\OpenSans-Regular.ttf'
            echo     Invoke-WebRequest -Uri 'https://fonts.google.com/download?family=Open+Sans' -OutFile $zipPath -ErrorAction Stop
            echo     Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
            echo     Copy-Item "$extractPath\static\OpenSans-Regular.ttf" $finalPath -Force -ErrorAction Stop
            echo     Copy-Item "$extractPath\static\OpenSans-SemiBold.ttf" 'data\fonts\OpenSans-SemiBold.ttf' -Force -ErrorAction SilentlyContinue
            echo     Remove-Item $zipPath -Force
            echo     Remove-Item $extractPath -Recurse -Force
            echo } catch {
            echo     Write-Host "Fallback font download failed:" $_.Exception.Message
            echo }
        ) > "%TEMP%\spaudible_font_installer.ps1"
        
        powershell -ExecutionPolicy Bypass -File "%TEMP%\spaudible_font_installer.ps1"
        del "%TEMP%\spaudible_font_installer.ps1" 2>nul
    )

    if exist "data\fonts\OpenSans-Regular.ttf" (
        echo Font downloaded.
    ) else (
        echo [Warning] Could not download font; will use system default.
    )
)

:: Run directly if the environment has already been set up
if exist ".venv\Scripts\python.exe" (
    echo Launching Spaudible...
    call ".venv\Scripts\python.exe" main.py
    if errorlevel 1 (
        echo.
        echo [Error] Spaudible crashed or encountered an error (exit code %errorlevel%^).
        echo If the error above mentions "ModuleNotFoundError", try deleting the .venv folder and running again.
        pause
        exit /b 1
    )
    echo.
    echo Spaudible has finished running.
    pause
    exit /b 0
)

echo ==========================================
echo Spaudible - First-Time Setup (Windows)
echo ==========================================
echo.

:: Check for UV in PATH
where uv >nul 2>&1
if %errorlevel% equ 0 (
    set UV_CMD=uv
    echo Found UV in system PATH.
) else (
    echo UV not found in PATH, checking for local copy...
    if not exist "uv.exe" (
        echo Downloading UV ^(Python project manager^)...
        
        :: Try curl first (Windows 10/11), fallback to PowerShell
        curl -L -o uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip 2>nul
        if not exist uv.zip (
            echo curl not available or download failed, trying PowerShell...
            powershell -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile uv.zip"
        )
        
        if not exist uv.zip (
            echo [Error] Failed to download UV.
            echo Please check your internet connection or download UV manually from:
            echo https://github.com/astral-sh/uv/releases
            echo Then extract and place uv.exe into this directory, and rerun spaudible.bat.
            pause
            exit /b 1
        )
        
        echo Extracting UV...
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
            echo [Error] UV installation failed. uv.exe not found after extraction.
            pause
            exit /b 1
        )
        echo UV downloaded successfully.
    )
    set UV_CMD=uv.exe
)

echo Installing Python 3.12 (this may take a moment)...
%UV_CMD% python install 3.12 --quiet
if errorlevel 1 (
    echo [Error] Failed to install Python 3.12.
    echo This might be due to insufficient disk space or network issues.
    pause
    exit /b 1
)

echo Creating virtual environment...
%UV_CMD% venv --python 3.12
if errorlevel 1 (
    echo [Error] Failed to create virtual environment.
    pause
    exit /b 1
)

echo Installing dependencies (this may take several minutes)...

:: Check for NVIDIA GPU before installing dependencies to download correct PyTorch version
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo NVIDIA GPU detected; installing CUDA-enabled PyTorch...
    %UV_CMD% pip install torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cu128
) else (
    echo.
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch...
    %UV_CMD% pip install torch==2.9.1
)

:: Install remaining dependencies (torch already satisfied; will skip)
%UV_CMD% pip install -e .
if errorlevel 1 (
    echo [Error] Failed to install dependencies.
    echo Check that you have an internet connection and that pyproject.toml exists.
    pause
    exit /b 1
)

:: Verify CUDA installation if NVIDIA was detected
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo Verifying CUDA installation...
    .venv\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    if errorlevel 1 (
        echo [Warning] CUDA verification failed, but continuing with CPU mode...
    )
)

echo.
echo ==========================================
echo Launching Spaudible...
echo ==========================================

:: Direct launch command for Spaudible (prevents uv from reverting to the torch CPU version)
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [Error] Spaudible encountered an error during launch.
    pause
    exit /b 1
)

echo.
echo Setup complete!
pause
exit /b 0
