#!/bin/bash

# Business Intelligence Orchestrator - Quick Setup Script

echo "======================================"
echo "Business Intelligence Orchestrator"
echo "Quick Setup Script"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

echo "✓ Python 3 found"

# Check if .env exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your OPENAI_API_KEY"
    echo ""
    read -p "Press Enter to open .env in nano (or Ctrl+C to edit manually later)..."
    nano .env
else
    echo "✓ .env file already exists"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To start using the orchestrator:"
echo ""
echo "1. Make sure you've added your OPENAI_API_KEY to .env"
echo ""
echo "2. Run the CLI interface:"
echo "   source venv/bin/activate"
echo "   python cli.py"
echo ""
echo "3. Or start the FastAPI server:"
echo "   source venv/bin/activate"
echo "   uvicorn src.main:app --reload"
echo ""
echo "4. Or use Docker:"
echo "   docker-compose up --build"
echo ""
echo "See README.md for more information."
echo ""
