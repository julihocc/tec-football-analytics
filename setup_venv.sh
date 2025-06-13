#!/bin/bash

# Virtual Environment Setup Script for Migrating to Python Project
# This script sets up a Python virtual environment and installs dependencies

set -e  # Exit on any error

PROJECT_DIR="/home/julihocc/cbbeggs/main.worktrees/migrating-to-python"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================================"
echo "Setting up Virtual Environment for Migrating to Python"
echo "============================================================"

# Change to project directory
cd "$PROJECT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python version: $(python3 --version)"

# Check if python3-venv is available
if ! python3 -m venv --help &> /dev/null; then
    echo "❌ Error: python3-venv is not installed"
    echo "Please install it with: sudo apt install python3-venv"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create new virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv --prompt "migrating-to-python" venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Successfully installed packages from requirements.txt"
else
    echo "❌ Warning: requirements.txt not found"
    echo "Installing core packages manually..."
    pip install pandas numpy requests beautifulsoup4 lxml matplotlib seaborn scipy
fi

# Test the installation
echo "🧪 Testing installation..."
python -c "import pandas, numpy, requests, bs4; print('✅ All core packages imported successfully')"

echo ""
echo "============================================================"
echo "🎉 Virtual Environment Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the virtual environment, run:"
echo "    cd $PROJECT_DIR"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate the virtual environment, run:"
echo "    deactivate"
echo ""
echo "To run the Python script:"
echo "    cd $PROJECT_DIR"
echo "    source venv/bin/activate"
echo "    cd chapter03"
echo "    python chapter03.py"
echo ""
echo "Virtual environment location: $VENV_DIR"
echo "Python executable: $VENV_DIR/bin/python"
echo "Pip executable: $VENV_DIR/bin/pip"
echo "============================================================"
