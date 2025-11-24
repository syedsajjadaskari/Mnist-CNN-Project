#!/bin/bash

# ANN Demo Setup Script
# This script sets up the ANN demo environment and launches the application

set -e  # Exit on error

echo "============================================================"
echo "🧠 ANN Demo - Setup Script"
echo "============================================================"
echo ""

# Navigate to the ann directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ Pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies..."
echo "   - numpy"
echo "   - matplotlib"
echo "   - scikit-learn"
echo "   - gradio"
echo "   - pillow"
echo ""

pip install -r requirements.txt --quiet

echo "✅ All dependencies installed"
echo ""

# Test imports
echo "🧪 Testing imports..."
python3 -c "import numpy; import matplotlib; import sklearn; import gradio; from PIL import Image; print('✅ All imports successful')"
echo ""

# Run a quick test
echo "🧪 Running quick ANN test..."
python3 -c "
from ann_core import ANN
from datasets import generate_xor
import numpy as np

# Quick XOR test
X, y = generate_xor(n_samples=100)
model = ANN([2, 4, 1], learning_rate=0.1)
model.train(X, y, epochs=100, verbose=False)
accuracy = model.compute_accuracy(y, model.predict(X))

if accuracy > 0.8:
    print(f'✅ ANN test passed! Accuracy: {accuracy:.2f}')
else:
    print(f'⚠️  ANN test completed with accuracy: {accuracy:.2f}')
"
echo ""

echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "🚀 Launching ANN Demo..."
echo ""
echo "📊 The demo will open in your browser at:"
echo "   http://127.0.0.1:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Launch the demo
python3 demo_app.py
