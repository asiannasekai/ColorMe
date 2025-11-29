# Chromasonic Data Directory

This directory contains datasets and generated outputs for the Chromasonic project.

## Directory Structure

```
data/
├── audio/              # Generated audio files (WAV)
├── images/             # Input images for conversion
├── nottingham/         # Nottingham folk music dataset (auto-downloaded)
└── models/             # Saved model checkpoints
```

## Datasets

### Nottingham Music Database

The Nottingham dataset contains ~1000 traditional folk tunes in ABC notation format.

**Auto-download:**
When you run training with `--dataset nottingham`, the dataset will be automatically downloaded and parsed.

```bash
python train_models.py --dataset nottingham --epochs 100
```

**Manual download:**
You can also manually test the dataset loader:

```bash
cd src/chromasonic/melody_generation
python data_loader.py
```

**Dataset details:**
- **Source:** https://ifdo.ca/~seymour/nottingham/nottingham.html
- **Format:** ABC notation (text-based music notation)
- **Content:** Traditional folk tunes from the British Isles
- **Size:** ~1000 melodies
- **License:** Public domain

**Processed data:**
After first load, a processed version is saved to `nottingham/processed_melodies.json` for faster subsequent loading.

## Dataset Statistics

After loading the Nottingham dataset, you'll see statistics like:

```
- Number of melodies: ~1000
- Total notes: ~150,000
- Average length: ~150 notes per melody
- Note range: 0-87 (88-key piano)
- Common keys: D major, G major, A minor
```

## Using Real vs Synthetic Data

**Synthetic Data (default):**
```bash
python train_models.py --dataset synthetic --num-sequences 1000
```
- Quick to generate
- Good for testing and prototyping
- Random walk melodies (not very musical)

**Nottingham Dataset (recommended for production):**
```bash
python train_models.py --dataset nottingham --epochs 100
```
- Real folk melodies
- Better musical structure
- Models learn actual musical patterns
- Takes longer to train but produces better results

## Adding More Datasets

To add support for additional datasets (e.g., MIDI collections, other ABC datasets):

1. Create a new loader class in `src/chromasonic/melody_generation/data_loader.py`
2. Implement `load_dataset()` method to return list of note sequences
3. Add dataset option to `train_models.py`

Example datasets you could add:
- **Lakh MIDI Dataset** (175k+ MIDI files)
- **JSB Chorales** (Bach chorales)
- **Essen Folk Song Collection** (European folk songs)
- **Celtic ABC** (More folk tunes)

## Storage Requirements

- **Nottingham dataset:** ~5 MB compressed, ~10 MB extracted
- **Processed melodies:** ~2-3 MB JSON file
- **Model checkpoints:** ~5-50 MB per model (depends on size)
- **Generated audio:** ~1-5 MB per WAV file

## Data Cleanup

Safe to delete:
- `audio/*.wav` - Regenerated on each conversion
- `nottingham/processed_melodies.json` - Will be regenerated from ABC files
- Old model checkpoints if you've trained new versions

**Do not delete:**
- `nottingham/*.abc` - Original dataset files (will need to re-download)
- Your best model checkpoints

## Privacy & Licensing

- All folk music datasets are public domain
- Generated melodies are yours to use freely
- Model weights inherit the training data license
- When sharing trained models, credit the datasets used
