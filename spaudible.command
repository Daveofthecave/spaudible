#!/bin/bash
# spaudible.command
set -e
cd "$(dirname "$0")"

# Fast path: already set up?
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    .venv/bin/python main.py
    exit 0
fi

echo "=========================================="
echo "Spaudible First-Time Setup (Mac/Linux)"
echo "=========================================="

# Download UV if needed
if ! command -v uv &> /dev/null; then
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
    UV_CMD="./uv"
else
    UV_CMD="uv"
fi

echo "Installing Python 3.12..."
$UV_CMD python install 3.12 --quiet

echo "Creating environment..."
$UV_CMD venv --python 3.12

echo "Installing dependencies (5-10 minutes)..."
$UV_CMD pip install

echo "Launching..."
$UV_CMD run main.py

read -p "Press Enter to close..."
