#  Chromasonic Project Structure

This document provides a comprehensive overview of the Chromasonic project architecture and components.

##  Directory Structure

```
chromasonic/
├──  README.md              # Main project documentation
├──  setup.py               # Package setup and installation
├──  requirements.txt       # Python dependencies
├──  config.py             # Configuration settings
├──  Dockerfile            # Container configuration
├──  docker-compose.yml    # Multi-container setup
│
├──  src/chromasonic/       # Main source code
│   ├──  __init__.py       # Package initialization
│   ├──  pipeline.py       # Main pipeline orchestrator
│   ├──  main.py           # CLI interface
│   │
│   ├──  image_processing/  # Image loading and preprocessing
│   │   ├── __init__.py
│   │   └── loader.py         # ImageLoader class
│   │
│   ├──  color_analysis/    # Color extraction algorithms
│   │   ├── __init__.py
│   │   └── extractor.py      # ColorExtractor class
│   │
│   ├──  wavelength_mapping/ # Color-to-wavelength conversion
│   │   ├── __init__.py
│   │   └── converter.py      # WavelengthConverter class
│   │
│   ├──  melody_generation/ # AI melody generation models
│   │   ├── __init__.py
│   │   └── models.py         # Markov, LSTM, Transformer models
│   │
│   ├──  audio_synthesis/   # Audio generation and effects
│   │   ├── __init__.py
│   │   └── synthesizer.py    # AudioSynthesizer class
│   │
│   └──  web_interface/     # Flask web application
│       ├── __init__.py
│       ├── app.py            # Flask routes and API
│       ├── templates/
│       │   └── index.html    # Web UI
│       └── static/           # CSS, JS, assets
│
├──  notebooks/             # Jupyter notebooks
│   └── chromasonic_demo.ipynb # Interactive tutorial
│
├──  models/               # Trained ML models (saved states)
│
├──  data/                 # Data storage
│   ├── images/              # Sample and uploaded images
│   └── audio/               # Generated audio files
│
└──  tests/               # Unit and integration tests
    ├── __init__.py
    └── test_chromasonic.py  # Test suite
```

##  Core Components

### 1. Pipeline Orchestrator (`pipeline.py`)
- **ChromasonicPipeline**: Main class coordinating the entire process
- Configurable ML models, scales, and synthesis methods
- Batch processing capabilities
- Audio export functionality

### 2. Image Processing (`image_processing/`)
- **ImageLoader**: Handles image loading, resizing, and preprocessing
- Support for multiple formats (JPEG, PNG, GIF, BMP, TIFF)
- Automatic color space conversion and size optimization

### 3. Color Analysis (`color_analysis/`)
- **ColorExtractor**: Advanced color extraction using ML clustering
- K-means clustering for dominant colors
- Color quantization and histogram-based methods
- Color harmony and temperature analysis

### 4. Wavelength Mapping (`wavelength_mapping/`)
- **WavelengthConverter**: Scientific color-to-wavelength conversion
- HSV-based hue mapping to visible spectrum (380-750nm)
- Wavelength-to-frequency transformation
- Color validation and spectrum visualization

### 5. Melody Generation (`melody_generation/`)
- **MelodyGenerator**: AI-powered melody creation
- **MarkovChainModel**: Probabilistic sequence generation
- **LSTMModel**: Deep learning for musical patterns
- **TransformerModel**: Advanced attention-based composition

### 6. Audio Synthesis (`audio_synthesis/`)
- **AudioSynthesizer**: High-quality audio generation
- Multiple synthesis methods (additive, FM, subtractive)
- ADSR envelope shaping
- Audio effects (reverb, chorus, delay)
- WAV file export

### 7. Web Interface (`web_interface/`)
- **Flask Application**: Modern web UI for easy interaction
- Real-time image upload and processing
- Interactive parameter controls
- Audio playback and download
- RESTful API endpoints

##  Key Features

###  Advanced Color Science
- **K-means Clustering**: Intelligent dominant color extraction
- **HSV Color Space**: Perceptually accurate hue-to-wavelength mapping
- **Visible Spectrum**: Scientific 380-750nm wavelength range
- **Color Harmony**: Temperature and saturation analysis

###  Musical Intelligence
- **Multiple Scales**: Major, minor, pentatonic, blues, chromatic, modal
- **Note Quantization**: Frequency-to-scale note mapping
- **Chord Progressions**: Automatic harmonic sequence generation
- **Musical Intervals**: Semitone relationship analysis

###  Machine Learning Models
- **Markov Chains**: Fast, interpretable melody generation
- **LSTM Networks**: Deep sequence learning for musical patterns
- **Transformers**: State-of-the-art attention-based composition
- **Configurable Training**: Custom datasets and model parameters

###  Professional Audio
- **Multiple Synthesis**: Sine, additive, FM, subtractive waveforms
- **ADSR Envelopes**: Professional attack-decay-sustain-release shaping
- **Audio Effects**: Reverb, chorus, delay with configurable parameters
- **High Quality**: 44.1kHz, 16-bit WAV output

##  Usage Patterns

### Command Line Interface
```bash
# Basic conversion
chromasonic generate --image sunset.jpg --output melody.wav

# Advanced options
chromasonic generate --image art.png --scale minor --tempo 90 --model lstm

# Batch processing
chromasonic batch --input-dir photos/ --output-dir music/

# Web interface
chromasonic web --port 5000
```

### Python API
```python
from chromasonic import ChromasonicPipeline

# Initialize pipeline
pipeline = ChromasonicPipeline(
    model_type='transformer',
    scale='pentatonic',
    tempo=120
)

# Process image
result = pipeline.process_image('image.jpg')

# Save audio
pipeline.save_audio(result['audio'], 'output.wav')
```

### Web Interface
1. Upload image via drag-and-drop or file browser
2. Adjust parameters (colors, scale, tempo, model)
3. Click "Convert" to generate music
4. Listen to results and download audio

##  Scientific Foundation

### Color-to-Wavelength Conversion
```
RGB Color → HSV Hue → Light Wavelength (nm)
Red (0°)     → 700nm
Orange (30°) → 620nm  
Yellow (60°) → 580nm
Green (120°) → 520nm
Cyan (180°)  → 490nm
Blue (240°)  → 450nm
Violet (300°) → 400nm
```

### Wavelength-to-Frequency Mapping
```
Light Wavelength → Audio Frequency (Hz)
Short wavelength (blue) → High frequency
Long wavelength (red)   → Low frequency
Scaled to musical octave range (C3-C6)
```

### Musical Scale Theory
```
Major Scale:     C D E F G A B (0,2,4,5,7,9,11 semitones)
Minor Scale:     C D Eb F G Ab Bb (0,2,3,5,7,8,10)
Pentatonic:      C D E G A (0,2,4,7,9)
Blues Scale:     C Eb F F# G Bb (0,3,5,6,7,10)
```

## 🔧 Configuration

### Environment Variables
- `CHROMASONIC_MODEL_TYPE`: Default ML model
- `CHROMASONIC_SAMPLE_RATE`: Audio sample rate
- `CHROMASONIC_DATA_DIR`: Data storage directory

### Config File (`config.py`)
- Audio settings (sample rate, bit depth)
- ML model parameters
- Web interface configuration
- File paths and directories

##  Testing

### Test Coverage
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Full pipeline validation
- **Performance Tests**: Audio quality and generation speed
- **Web Tests**: API endpoints and user interface

### Running Tests
```bash
cd chromasonic
python -m pytest tests/
python tests/test_chromasonic.py
```

##  Performance

### Optimization Strategies
- **Image Sampling**: Reduce pixel count for large images
- **K-means Acceleration**: MiniBatch clustering for speed
- **Audio Buffering**: Efficient memory usage for long melodies
- **Caching**: Store intermediate results for parameter changes

### Benchmarks
- **Small Images** (< 500x500): < 2 seconds processing
- **Large Images** (> 1MP): < 5 seconds processing
- **Audio Generation**: Real-time synthesis
- **Web Interface**: < 500ms API response time

##  Future Enhancements

### Planned Features
- **Video Processing**: Extract colors from video frames
- **MIDI Export**: Generate MIDI files for DAW integration
- **Real-time Mode**: Live camera-to-music conversion
- **Advanced Models**: GPT-based music generation
- **Mobile App**: iOS/Android application
- **Collaborative Features**: Share and remix creations

### Research Directions
- **Perceptual Color Mapping**: CIE Lab color space integration
- **Musical Style Transfer**: Genre-specific melody generation
- **Harmonic Analysis**: Automatic chord detection and progression
- **Emotional Mapping**: Color psychology to musical mood

##  References and Credits

### Scientific Papers
- Color perception and wavelength mapping research
- Music information retrieval and generation
- Deep learning for sequential data

### Open Source Libraries
- **scikit-learn**: Machine learning algorithms
- **OpenCV**: Computer vision and image processing  
- **librosa**: Audio analysis and processing
- **Flask**: Web application framework
- **NumPy/SciPy**: Scientific computing foundation

### Inspiration
- Synesthesia research and cross-modal perception
- Generative art and algorithmic composition
- Color theory and digital art techniques

---

