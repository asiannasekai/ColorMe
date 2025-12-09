## Chromasonic 

**Transform Images into Beautiful Music Through AI and Color Science**

This repository contains **Chromasonic**, a  multimodal ML pipeline that converts images into musically coherent melodies by extracting colors, mapping them to wavelengths and frequencies, then generating harmonious compositions using advanced machine learning models.

## Quick Start

```bash
cd chromasonic
pip install -r requirements.txt
pip install -e .

# Generate music from an image
chromasonic generate --image sunset.jpg --output melody.wav --scale major

# Start web interface
chromasonic web
```

## 📖 Full Documentation

See the complete project in the [`chromasonic/`](./chromasonic/) directory:

- **[README.md](./chromasonic/README.md)** - Complete project documentation
- **[ARCHITECTURE.md](./chromasonic/ARCHITECTURE.md)** - Technical architecture overview
- **[notebooks/](./chromasonic/notebooks/)** - Interactive Jupyter tutorials
- **[src/](./chromasonic/src/)** - Full source code implementation

##  What Makes Chromasonic Special

 **Scientific Foundation**: Uses real color science to map RGB → wavelengths → frequencies  
 **AI-Powered**: Markov chains, LSTM, and Transformer models for melody generation  
 **Musical Intelligence**: Supports major, minor, pentatonic, blues, and custom scales  
 **Web Interface**: Beautiful, interactive web UI for real-time conversion  
 **Professional Audio**: High-quality synthesis with ADSR envelopes and effects  

## The Pipeline

1. **Image Processing**: Load and analyze images using computer vision
2. **Color Extraction**: Extract dominant colors via K-means clustering  
3. **Wavelength Mapping**: Convert RGB to light wavelengths (380-750nm)
4. **Frequency Translation**: Map wavelengths to musical frequencies
5. **AI Melody Generation**: Create coherent melodies using ML models
6. **Audio Synthesis**: Generate high-quality audio with customizable synthesis

Experience the magic of transforming visual art into musical art! 
