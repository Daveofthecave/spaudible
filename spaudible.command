#!/bin/bash
# spaudible.command

# If not running in a terminal (eg. Linux double-click), reopen in one
if [ ! -t 0 ] && [ "$(uname -s)" = "Linux" ]; then
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash "$0" "$@"
        exit
    elif command -v konsole &> /dev/null; then
        konsole -e bash "$0" "$@"
        exit
    elif command -v xfce4-terminal &> /dev/null; then
        xfce4-terminal -e "bash '$0' '$@'"
        exit
    elif command -v xterm &> /dev/null; then
        xterm -e bash "$0" "$@"
        exit
    elif command -v qterminal &> /dev/null; then
        qterminal -e bash "$0" "$@"
        exit
    fi
fi

# Keep terminal open on error so user can see what failed
trap 'echo ""; echo "========================================"; echo "Error: Setup failed. See message above."; echo "========================================"; read -p "Press Enter to close..."' ERR
set -e

cd "$(dirname "$0")"

# Download Open Sans font for the GUI if not yet present
if [ -f "data/fonts/OpenSans-Regular.ttf" ]; then
    : # Font exists, skip
else
    echo "Downloading Open Sans font..."
    mkdir -p data/fonts 2>/dev/null

    # Primary: .ttf from GitHub
    curl -L -o "data/fonts/OpenSans-Regular.ttf" "https://github.com/googlefonts/opensans/raw/refs/heads/main/fonts/ttf/OpenSans-Regular.ttf" --silent --fail 2>/dev/null
    # Also grab SemiBold style for headers
    curl -L -o "data/fonts/OpenSans-SemiBold.ttf" "https://github.com/googlefonts/opensans/raw/refs/heads/main/fonts/ttf/OpenSans-SemiBold.ttf" --silent --fail 2>/dev/null

    # Fallback: Query Google Fonts API for direct .ttf URLs
    if [ ! -f "data/fonts/OpenSans-Regular.ttf" ]; then
        echo "Downloading font from fallback URL..."
        
        download_weight() {
            local weight=$1
            local name=$2
            local css_url="https://fonts.googleapis.com/css2?family=Open+Sans:wght@${weight}"
            
            # Extract .ttf URL from CSS response
            url=$(curl -s "$css_url" | grep -o 'https://fonts\.gstatic\.com/s/opensans/[^)]*\.ttf' | head -1)
            if [ -n "$url" ]; then
                curl -L -o "data/fonts/${name}.ttf" "$url" --silent --fail 2>/dev/null
            fi
        }
        
        # Download Regular (400) and SemiBold (600)
        download_weight 400 "OpenSans-Regular"
        download_weight 600 "OpenSans-SemiBold"
    fi

    if [ -f "data/fonts/OpenSans-Regular.ttf" ]; then
        echo "Font downloaded."
    else
        echo "[Warning] Could not download font; will use system default."
    fi
fi

# Fast path: already set up?
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    echo "Launching Spaudible..."
    .venv/bin/python main.py
    read -p "Press Enter to close..."
    exit 0
fi

echo "=========================================="
echo "Spaudible - First-Time Setup (Mac/Linux)"
echo "=========================================="
echo ""

# Download UV if needed
if ! command -v uv &> /dev/null; then
    if [ ! -f "./uv" ]; then
        echo "Downloading UV (Python project manager)..."
        
        # Detect OS and architecture
        OS=$(uname -s)
        ARCH=$(uname -m)
        
        case "$OS" in
            Linux*)
                case "$ARCH" in
                    x86_64) PLATFORM="x86_64-unknown-linux-gnu" ;;
                    aarch64) PLATFORM="aarch64-unknown-linux-gnu" ;;
                    *) echo "Unsupported Linux architecture: $ARCH"; exit 1 ;;
                esac
                ;;
            Darwin*)
                case "$ARCH" in
                    x86_64) PLATFORM="x86_64-apple-darwin" ;;
                    arm64) PLATFORM="aarch64-apple-darwin" ;;
                    *) echo "Unsupported Mac architecture: $ARCH"; exit 1 ;;
                esac
                ;;
            *)
                echo "Unsupported OS: $OS"
                exit 1
                ;;
        esac
        
        URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${PLATFORM}.tar.gz"
        
        # Download and extract
        if command -v curl &> /dev/null; then
            curl -L -o uv.tar.gz "$URL"
        else
            wget -O uv.tar.gz "$URL"
        fi

        tar -xzf uv.tar.gz && rm uv.tar.gz
        
        # UV extracts to a subdirectory (eg. uv-x86_64-unknown-linux-gnu/)
        # Find the binary and move it to current directory
        if [ ! -f "./uv" ]; then
            UV_BIN=$(find . -maxdepth 2 -name "uv" -type f 2>/dev/null | head -1)
            if [ -n "$UV_BIN" ]; then
                mv "$UV_BIN" ./uv
                # Clean up the extracted directory
                UVX_DIR=$(dirname "$UV_BIN")
                rm -rf "$UVX_DIR" 2>/dev/null || true
            fi
        fi
        chmod +x ./uv
    fi
    UV_CMD="./uv"
else
    UV_CMD="uv"
fi

echo "Installing Python 3.12 (this may take a moment)..."
$UV_CMD python install 3.12 --quiet

echo "Creating virtual environment..."
$UV_CMD venv --python 3.12

echo "Installing dependencies (this may take several minutes)..."

# Check for NVIDIA GPU (Linux only) before installing to download correct PyTorch version
# Note: Macs use Metal, not CUDA, so we skip this check on Darwin
if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA GPU detected; installing CUDA-enabled PyTorch..."
    $UV_CMD pip install torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cu128
else
    echo ""
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    $UV_CMD pip install torch==2.9.1
fi

# Install remaining dependencies (torch already satisfied, will skip)
$UV_CMD pip install -e .

# Verify CUDA installation on Linux if NVIDIA was detected
if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Verifying CUDA installation..."
    .venv/bin/python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "Warning: CUDA verification failed, continuing with CPU mode..."
fi

echo ""
echo "=========================================="
echo "Launching Spaudible..."
echo "=========================================="

# Direct launch in place of 'uv run' to avoid dependency sync overwriting CUDA
.venv/bin/python main.py

# Success - pause before closing
read -p "Press Enter to close..."
