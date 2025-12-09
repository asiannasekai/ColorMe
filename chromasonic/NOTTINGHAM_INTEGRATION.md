# Nottingham Dataset Integration - Summary

## 🎵 What Was Implemented

The Nottingham Music Database has been fully integrated into the Chromasonic training pipeline, allowing models to learn from real folk music instead of synthetic random walks.

## 📦 New Files Created

### 1. `src/chromasonic/melody_generation/data_loader.py` (450+ lines)

**NottinghamDataLoader class:**
- Auto-downloads Nottingham dataset (~1000 folk tunes in ABC notation)
- Parses ABC notation to MIDI note sequences (0-87 range)
- Handles sharps, flats, octave variations
- Caches processed data as JSON for fast re-loading
- Provides dataset statistics and metadata

**Utility functions:**
- `split_dataset()` - Split into train/val/test sets with length filtering
- ABC-to-MIDI note mapping
- Dataset validation and analysis

### 2. `data/README.md`

Comprehensive documentation covering:
- Dataset details and auto-download instructions
- Synthetic vs real data comparison
- Storage requirements and cleanup guidelines
- Instructions for adding more datasets
- Licensing and privacy information

### 3. `example_nottingham_dataset.py`

Interactive demo script showing:
- How to download and load the dataset
- Dataset statistics and examples
- Train/val/test splitting
- Processing and caching

## 🔧 Modified Files

### 1. `src/chromasonic/melody_generation/training.py`

**Added:**
- `load_music_dataset()` function
- Support for Nottingham dataset loading
- Automatic caching and filtering
- Integration with existing training infrastructure

### 2. `train_models.py`

**Added parameters:**
- `--dataset` - Choose 'synthetic' or 'nottingham'
- `--data-dir` - Custom dataset location
- `--min-melody-length` - Minimum melody length filter
- `--max-melody-length` - Maximum melody length filter

**Updated training flow:**
- Detects dataset type
- Auto-downloads Nottingham on first use
- Loads cached data on subsequent runs
- Passes real data to training functions

### 3. `TRAINING_GUIDE.md`

**Enhanced with:**
- Dataset selection guide
- Pros/cons comparison table
- Quick start examples for both datasets
- Recommendations for production training

## 🚀 Usage Examples

### Train with Nottingham Dataset (Recommended)

```bash
# Basic training with real folk music
python train_models.py --dataset nottingham --epochs 100

# LSTM only on folk music
python train_models.py \
    --dataset nottingham \
    --model lstm \
    --epochs 100 \
    --batch-size 64

# Both models with custom settings
python train_models.py \
    --dataset nottingham \
    --model both \
    --epochs 150 \
    --min-melody-length 32 \
    --max-melody-length 64 \
    --learning-rate 0.001
```

### Train with Synthetic Data (Fast Testing)

```bash
# Quick test run
python train_models.py --dataset synthetic --epochs 10

# Large synthetic dataset
python train_models.py \
    --dataset synthetic \
    --num-sequences 5000 \
    --epochs 50
```

### Test Dataset Loader

```bash
# Run the example demo
python example_nottingham_dataset.py

# Or test directly
cd src/chromasonic/melody_generation
python data_loader.py
```

##  Dataset Details

**Nottingham Music Database:**
- **Size:** ~1000 traditional folk tunes
- **Format:** ABC notation (text-based music notation)
- **Source:** https://ifdo.ca/~seymour/nottingham/nottingham.html
- **Origin:** British Isles folk music
- **License:** Public domain
- **Download size:** ~5 MB compressed
- **Processed size:** ~2-3 MB JSON

**Statistics (typical):**
- Number of melodies: ~1000
- Total notes: ~150,000
- Average length: ~150 notes per melody
- Note range: 0-87 (88-key piano)
- Common keys: D major, G major, A minor
- Common meters: 4/4, 6/8, 3/4

## Benefits of Real Data

### Before (Synthetic Data)
-  Random walk patterns
- No musical structure
- Poor phrase boundaries
-  Unnatural interval distributions
- Models produce random-sounding output

### After (Nottingham Dataset)
-  Real musical patterns
-  Proper phrase structure
-  Natural melodic contours
-  Musical interval distributions
-  Models produce folk-like melodies

##  Workflow

1. **First run:** Downloads and parses ABC files (~30 seconds)
2. **Saves processed data:** Caches to `data/nottingham/processed_melodies.json`
3. **Subsequent runs:** Loads from cache (~2 seconds)
4. **Training:** Uses real folk melodies instead of random walks

##  Testing

To verify the implementation works:

```bash
# Test data loader import
python -c "from src.chromasonic.melody_generation.data_loader import NottinghamDataLoader; print('✓ Import successful')"

# Run example demo
python example_nottingham_dataset.py

# Train a small model
python train_models.py --dataset nottingham --epochs 5 --model lstm
```

##  File Structure After First Run

```
chromasonic/
├── data/
│   ├── README.md                    # Dataset documentation
│   └── nottingham/                  # Auto-created on first run
│       ├── nottingham.zip           # Downloaded dataset
│       ├── *.abc                    # ABC notation files
│       └── processed_melodies.json  # Cached processed data
├── src/chromasonic/melody_generation/
│   ├── data_loader.py              # NEW: Dataset loading
│   └── training.py                 # Updated: Real data support
├── train_models.py                 # Updated: Dataset selection
├── example_nottingham_dataset.py   # NEW: Demo script
└── TRAINING_GUIDE.md              # Updated: Dataset guide
```

## Musical Quality Improvements

Training on Nottingham vs synthetic data produces models that generate melodies with:

1. **Better phrasing** - Natural 4, 8, or 16-bar phrases
2. **Musical intervals** - Stepwise motion vs random jumps
3. **Tonal coherence** - Stays in key, resolves to tonic
4. **Rhythmic patterns** - Recognizable folk rhythms
5. **Contour variety** - Arch shapes, ascending/descending patterns
6. **Repetition/variation** - Musical motifs that repeat and develop

##  Future Enhancements

The data loader is designed to be extensible. Future datasets could include:

- **Lakh MIDI Dataset** - 175k+ MIDI files (pop, rock, classical)
- **JSB Chorales** - Bach chorales (4-part harmony)
- **Essen Folk Song Collection** - European folk songs
- **Celtic ABC Tunes** - More folk music
- **User-uploaded MIDI** - Custom training data

##  Documentation Updates

All documentation has been updated:
- ✅ TRAINING_GUIDE.md - Dataset selection guide
- ✅ data/README.md - Dataset details and usage
- ✅ Code comments - Comprehensive docstrings
- ✅ Example script - Interactive demo



**To start training with real music:**

```bash
python train_models.py --dataset nottingham --epochs 100
```

The dataset will auto-download on first run and cache for subsequent uses!
