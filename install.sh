#!/bin/bash
# Installation script for TransFilm

echo "=========================================="
echo "TransFilm Installation Script"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
echo "✓ Python version: $python_version"

# Check if FFmpeg is installed
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is installed"
else
    echo "⚠ FFmpeg is not installed"
    echo "  Please install FFmpeg:"
    echo "    Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "    macOS: brew install ffmpeg"
    echo "    Windows: Download from https://ffmpeg.org/download.html"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo ""
echo "Installing TransFilm package..."
pip install -e .

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p .cache
mkdir -p .temp
mkdir -p .cache/models

# Run structure test
echo ""
echo "Running structure tests..."
python test_structure.py

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run CLI help:"
echo "   python cli.py --help"
echo ""
echo "3. Start Web UI:"
echo "   python webui.py"
echo ""
echo "Note: First run will download AI models (may take time)"
echo "=========================================="
