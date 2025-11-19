#!/bin/bash

# Chromasonic Quick Start Script
# This script helps you get Chromasonic up and running quickly

set -e

echo "🎨 Chromasonic Quick Start Setup 🎵"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Please run this script from the chromasonic/ directory"
    echo "   cd chromasonic && ./quickstart.sh"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Chromasonic in development mode
echo "🎨 Installing Chromasonic package..."
pip install -e .

# Create sample image if none exists
if [ ! -f "data/images/sample.png" ]; then
    echo "🖼️ Creating sample image..."
    python3 -c "
import numpy as np
from PIL import Image
import colorsys

# Create a colorful sample image
width, height = 400, 300
img = np.zeros((height, width, 3), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        hue = (j / width) * 360
        saturation = 0.8
        value = 0.9 - (i / height) * 0.3
        
        rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
        img[i, j] = [int(c * 255) for c in rgb]

Image.fromarray(img).save('data/images/sample.png')
print('✅ Sample image created: data/images/sample.png')
"
fi

# Run a quick test
echo "🧪 Running quick test..."
python3 -c "
from chromasonic import ChromasonicPipeline
print('✅ Chromasonic imported successfully!')

# Quick pipeline test
pipeline = ChromasonicPipeline(model_type='markov', scale='major')
print('✅ Pipeline initialized successfully!')
"

# Generate sample audio
echo "🎵 Generating sample audio..."
chromasonic generate \
    --image data/images/sample.png \
    --output data/audio/sample_melody.wav \
    --scale major \
    --tempo 120 \
    --duration 10 \
    --num-colors 6

echo ""
echo "🎉 Chromasonic Setup Complete!"
echo "=============================="
echo ""
echo "🎮 What's Next?"
echo "---------------"
echo "1. 🎵 Listen to your sample: data/audio/sample_melody.wav"
echo "2. 🌐 Start web interface:   chromasonic web"
echo "3. 📓 Try the notebook:      jupyter notebook notebooks/chromasonic_demo.ipynb"
echo "4. 🎨 Convert your images:   chromasonic generate --image YOUR_IMAGE.jpg"
echo ""
echo "🔗 Quick Commands:"
echo "------------------"
echo "• Web interface:     chromasonic web"
echo "• CLI help:          chromasonic --help"
echo "• Run tests:         python tests/test_chromasonic.py"
echo "• Batch processing:  chromasonic batch --input-dir photos/ --output-dir music/"
echo ""
echo "📚 Documentation:"
echo "------------------"
echo "• README.md         - Full project documentation"
echo "• ARCHITECTURE.md   - Technical architecture"
echo "• notebooks/        - Interactive tutorials"
echo ""
echo "✨ Happy music making! Transform your visual world into sound! 🎨→🎵"