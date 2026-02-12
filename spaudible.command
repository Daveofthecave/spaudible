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

# Fast path: already set up?
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    echo "Launching Spaudible..."
    .venv/bin/python main.py
    read -p "Press Enter to close..."
    exit 0
fi

echo "=========================================="
echo "Spaudible First-Time Setup (Mac/Linux)"
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
$UV_CMD pip install

echo ""
echo "Launching Spaudible..."
echo "=========================================="
$UV_CMD run main.py

# Success - pause before closing (for GUI users)
read -p "Press Enter to close..."
