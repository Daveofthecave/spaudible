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

# Ensure we run from the batch file's directory
cd "$(dirname "$0")"

# Create GUI directories
mkdir -p data/gui/fonts 2>/dev/null

# Download background.png if not present
if [ ! -f "data/gui/background.png" ]; then
    echo "Downloading GUI background image..."
    curl -L -o "data/gui/background.png" "https://raw.githubusercontent.com/wiki/Daveofthecave/spaudible/assets/gui/background.png" --silent --fail 2>/dev/null
    if [ -f "data/gui/background.png" ]; then
        echo "background.png downloaded."
    else
        echo "[Warning] Could not download background.png"
    fi
fi

# Download Spaudible Sans + Instrument Sans fonts if not present
if [ ! -f "data/gui/fonts/SpaudibleSans-Regular.ttf" ]; then
    echo "Downloading GUI fonts..."
    curl -L -o "data/gui/fonts/SpaudibleSans-Regular.ttf" "https://raw.githubusercontent.com/wiki/Daveofthecave/spaudible/assets/gui/fonts/SpaudibleSans-Regular.ttf" --silent --fail 2>/dev/null
    curl -L -o "data/gui/fonts/InstrumentSans-SemiBold.ttf" "https://raw.githubusercontent.com/wiki/Daveofthecave/spaudible/assets/gui/fonts/InstrumentSans-SemiBold.ttf" --silent --fail 2>/dev/null
    
    if [ -f "data/gui/fonts/SpaudibleSans-Regular.ttf" ]; then
        echo "Fonts downloaded."
    else
        echo "[Warning] Could not download SpaudibleSans fonts; UI may not render correctly."
    fi
fi

# Keep terminal open on error so the user can see what failed
trap 'echo ""; echo "========================================"; echo "Error: Setup failed. See message above."; echo "========================================"; read -p "Press Enter to close..."' ERR
set -e

# Fast path: already set up?
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    # Determine UV command
    if [ -f "./uv" ]; then
        UV_CMD="./uv"
    elif command -v uv &> /dev/null; then
        UV_CMD="uv"
    else
        # UV missing but .venv exists - attempt first-time setup
        echo "UV not found, attempting setup..."
        # Fall through to first_time_setup
        : # Placeholder, will continue to setup section below
    fi
    
    # Only proceed with fast launch if UV_CMD was set
    if [ -n "$UV_CMD" ]; then
        # Check if dependencies need updating by comparing current pyproject.toml 
        # with the version last used to install dependencies (backed up in .venv)
        if [ -f "pyproject.toml" ]; then
            if [ ! -f ".venv/.pyproject.toml.installed" ] || ! cmp -s "pyproject.toml" ".venv/.pyproject.toml.installed"; then
                echo "Detected changes to pyproject.toml; reinstalling dependencies..."
                $UV_CMD pip install -e . >/dev/null 2>&1
                if [ $? -eq 0 ]; then
                    # Update the marker to match current state
                    cp "pyproject.toml" ".venv/.pyproject.toml.installed"
                else
                    echo "[Warning] Failed to update dependencies; attempting launch anyway..."
                fi
            fi
        fi

        echo "Launching Spaudible..."
        .venv/bin/python main.py
        read -p "Press Enter to close..."
        exit 0
    fi
fi

:first_time_setup
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
