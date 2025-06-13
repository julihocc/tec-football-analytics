#!/bin/bash

# Quick activation script for the virtual environment
# Usage: source activate_venv.sh

PROJECT_DIR="/home/julihocc/cbbeggs/main.worktrees/migrating-to-python"
VENV_DIR="$PROJECT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "🐍 Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo "✅ Virtual environment activated!"
    echo "Python: $(which python)"
    echo "Project directory: $PROJECT_DIR"
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    echo ""
    echo "Available commands:"
    echo "  cd chapter03 && python chapter03.py  # Run the Python script"
    echo "  deactivate                          # Exit virtual environment"
    echo ""
else
    echo "❌ Virtual environment not found at: $VENV_DIR"
    echo "Please run setup_venv.sh first to create the virtual environment"
fi
