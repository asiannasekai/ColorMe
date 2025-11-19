# Chromasonic: Image → Color → Wavelength → Music Generation

**Chromasonic** is a multimodal machine learning pipeline that transforms images into musically coherent melodies by extracting colors, mapping them to wavelengths and frequencies, and generating harmonious compositions using advanced ML models.

## 🎨 → 🎵 The Pipeline

1. **Image Processing**: Load and analyze input images
2. **Color Extraction**: Extract dominant colors using clustering algorithms
3. **Wavelength Mapping**: Convert RGB colors to light wavelengths
4. **Frequency Translation**: Map wavelengths to musical frequencies using mathematical relationships
5. **Melody Generation**: Use ML models (Markov Chains, LSTM, Transformers) to create coherent melodies
6. **Audio Synthesis**: Generate high-quality audio with customizable musical scales

## 🚀 Features

- **Multi-algorithm Color Extraction**: K-means clustering, color quantization, and dominant color analysis
- **Scientific Color-to-Sound Mapping**: Physics-based wavelength to frequency conversion
- **Advanced ML Models**: 
  - Markov Chain melody generation
  - LSTM neural networks for sequence modeling
  - Transformer architecture for complex musical patterns
- **Interactive Web Interface**: Real-time image upload and music generation
- **Multiple Musical Scales**: Major, minor, pentatonic, blues, and custom scales
- **Audio Export**: Generate and download high-quality audio files

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/asiannasekai/ColorMe.git
cd ColorMe/chromasonic

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🎼 Quick Start

### Command Line Interface
```bash
# Generate music from an image
chromasonic generate --image path/to/image.jpg --output melody.wav --scale major

# Use different ML models
chromasonic generate --image sunset.jpg --model lstm --scale pentatonic
```

### Python API
```python
from chromasonic import ChromasonicPipeline

# Initialize the pipeline
pipeline = ChromasonicPipeline(model_type='transformer')

# Generate music from image
melody = pipeline.process_image('path/to/image.jpg', scale='minor')

# Save audio
pipeline.save_audio(melody, 'output.wav')
```

### Web Interface
```bash
# Start the web server
python -m chromasonic.web_interface.app

# Open http://localhost:5000 in your browser
```

## 🧠 Machine Learning Models

### Markov Chain Model
- Analyzes color transition patterns
- Generates melodies based on probabilistic note sequences
- Fast and interpretable

### LSTM Neural Network
- Deep learning approach for sequence generation
- Captures long-term musical dependencies
- Trained on classical and contemporary music datasets

### Transformer Architecture
- State-of-the-art attention-based model
- Handles complex musical relationships
- Supports multi-track composition

## 🎨 Color-to-Music Mapping

The system uses scientifically grounded mappings:

1. **RGB → Wavelength**: Convert color values to light wavelengths (380-750nm)
2. **Wavelength → Frequency**: Mathematical relationship between light and sound
3. **Frequency → Musical Notes**: Map to chromatic scale with octave adjustments
4. **Musical Harmony**: Apply music theory rules for pleasing compositions

## 📁 Project Structure

```
chromasonic/
├── src/
│   ├── image_processing/     # Image loading and preprocessing
│   ├── color_analysis/       # Color extraction algorithms
│   ├── wavelength_mapping/   # Color-to-wavelength conversion
│   ├── melody_generation/    # ML models for music generation
│   ├── audio_synthesis/      # Audio generation and processing
│   └── web_interface/        # Flask web application
├── models/                   # Trained ML models
├── data/                     # Sample images and audio
├── notebooks/                # Jupyter notebooks for experimentation
└── tests/                    # Unit tests
```

## 🔬 Scientific Background

The project is based on the fascinating relationship between light and sound waves:
- Both are wave phenomena with frequency and wavelength
- Color perception corresponds to light wavelengths (380-750 nanometers)
- Musical notes correspond to sound frequencies (20Hz-20kHz)
- Mathematical mappings create synesthetic experiences

## 🎵 Musical Scales Supported

- **Major Scale**: Happy, bright tonality
- **Minor Scale**: Melancholic, introspective
- **Pentatonic**: Universal, pleasing to most cultures  
- **Blues Scale**: Expressive, emotional
- **Chromatic**: All 12 semitones
- **Custom Scales**: Define your own note relationships

## 🛠️ Development

```bash
# Run tests
pytest tests/

# Format code
black src/

# Type checking
mypy src/

# Start development server
python -m chromasonic.web_interface.app --debug
```

## 📊 Notebooks

Explore the `notebooks/` directory for:
- Color analysis experiments
- Model training workflows  
- Audio synthesis examples
- Interactive demos

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Color theory research and wavelength mappings
- Music theory and harmonic analysis
- Open-source ML and audio processing communities
- Synesthesia research inspiring cross-modal connections

---

*Transform your visual world into a sonic landscape with Chromasonic!* 🎨🎵