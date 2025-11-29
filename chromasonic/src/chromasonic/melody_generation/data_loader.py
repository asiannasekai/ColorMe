"""
Data Loader for Music Datasets
Supports loading and parsing various music datasets for training melody models.
"""

import logging
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class NottinghamDataLoader:
    """
    Loader for the Nottingham Music Database.
    
    The Nottingham dataset contains ~1000 folk tunes in ABC notation.
    Dataset source: https://ifdo.ca/~seymour/nottingham/nottingham.html
    """
    
    DATASET_URL = "https://ifdo.ca/~seymour/nottingham/nottingham.zip"
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize Nottingham data loader.
        
        Args:
            data_dir: Directory to store dataset (default: ./data/nottingham)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data" / "nottingham"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.melodies = []
        self.metadata = []
        
        # ABC notation note mapping
        self.note_to_midi = self._create_note_mapping()
        
    def _create_note_mapping(self) -> Dict[str, int]:
        """Create mapping from ABC notation to MIDI note numbers."""
        # Base notes (middle octave starts at C4 = 60)
        base_notes = {
            'C': 60, 'D': 62, 'E': 64, 'F': 65,
            'G': 67, 'A': 69, 'B': 71,
            'c': 72, 'd': 74, 'e': 76, 'f': 77,
            'g': 79, 'a': 81, 'b': 83
        }
        
        # Add sharps and flats
        mapping = {}
        for note, midi in base_notes.items():
            mapping[note] = midi
            mapping[f"^{note}"] = midi + 1  # Sharp
            mapping[f"_{note}"] = midi - 1  # Flat
            mapping[f"={note}"] = midi      # Natural
        
        # Add octave variations
        for note in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
            # Lower octave (,)
            mapping[f"{note},"] = base_notes[note] - 12
            mapping[f"^{note},"] = base_notes[note] - 11
            mapping[f"_{note},"] = base_notes[note] - 13
            
            # Higher octave (')
            base_upper = base_notes[note.lower()]
            mapping[f"{note.lower()}'"] = base_upper + 12
            mapping[f"^{note.lower()}'"] = base_upper + 13
            mapping[f"_{note.lower()}'"] = base_upper + 11
        
        return mapping
    
    def download_dataset(self, force: bool = False) -> bool:
        """
        Download the Nottingham dataset.
        
        Args:
            force: Force re-download even if dataset exists
            
        Returns:
            True if successful
        """
        zip_path = self.data_dir / "nottingham.zip"
        
        # Check if already downloaded
        if zip_path.exists() and not force:
            logger.info("Nottingham dataset already downloaded")
            return True
        
        try:
            logger.info(f"Downloading Nottingham dataset from {self.DATASET_URL}...")
            urllib.request.urlretrieve(self.DATASET_URL, zip_path)
            
            # Extract
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            logger.info(f"Dataset downloaded and extracted to {self.data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    def parse_abc_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse an ABC notation file and extract melodies.
        
        Args:
            file_path: Path to ABC file
            
        Returns:
            List of tune dictionaries with melody and metadata
        """
        tunes = []
        current_tune = None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line or line.startswith('%'):
                        continue
                    
                    # Start of new tune
                    if line.startswith('X:'):
                        if current_tune and current_tune.get('notes'):
                            tunes.append(current_tune)
                        
                        current_tune = {
                            'id': line[2:].strip(),
                            'title': '',
                            'key': 'C',
                            'meter': '4/4',
                            'notes': [],
                            'abc_melody': ''
                        }
                    
                    elif current_tune is not None:
                        # Title
                        if line.startswith('T:'):
                            current_tune['title'] = line[2:].strip()
                        
                        # Key
                        elif line.startswith('K:'):
                            current_tune['key'] = line[2:].strip().split()[0]
                        
                        # Meter
                        elif line.startswith('M:'):
                            current_tune['meter'] = line[2:].strip()
                        
                        # Melody line (no header marker)
                        elif not line[0].isupper() or ':' not in line[:3]:
                            current_tune['abc_melody'] += ' ' + line
                
                # Add last tune
                if current_tune and current_tune.get('abc_melody'):
                    current_tune['notes'] = self._parse_abc_melody(current_tune['abc_melody'])
                    if current_tune['notes']:
                        tunes.append(current_tune)
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return tunes
    
    def _parse_abc_melody(self, abc_melody: str) -> List[int]:
        """
        Parse ABC melody notation to MIDI note numbers.
        
        Args:
            abc_melody: ABC notation melody string
            
        Returns:
            List of MIDI note numbers
        """
        notes = []
        
        # Remove bar lines, repeat signs, and other symbols
        abc_melody = abc_melody.replace('|', ' ')
        abc_melody = abc_melody.replace(':', ' ')
        abc_melody = abc_melody.replace('[', ' ')
        abc_melody = abc_melody.replace(']', ' ')
        
        # Split into tokens
        tokens = abc_melody.split()
        
        for token in tokens:
            # Skip empty tokens
            if not token:
                continue
            
            # Extract notes from token (may contain rhythm info)
            note_pattern = r'([_=\^]?[A-Ga-g][,\']?)'
            matches = re.findall(note_pattern, token)
            
            for note_str in matches:
                if note_str in self.note_to_midi:
                    midi_note = self.note_to_midi[note_str]
                    # Normalize to 0-87 range (88 notes)
                    normalized = midi_note - 21  # A0 = 21 in MIDI
                    if 0 <= normalized < 88:
                        notes.append(normalized)
        
        return notes
    
    def load_dataset(self, max_tunes: Optional[int] = None) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        """
        Load all ABC files from the Nottingham dataset.
        
        Args:
            max_tunes: Maximum number of tunes to load (None = all)
            
        Returns:
            Tuple of (melodies, metadata)
            - melodies: List of melody sequences (note indices)
            - metadata: List of metadata dictionaries
        """
        melodies = []
        metadata = []
        
        # Find ABC files
        abc_files = list(self.data_dir.glob("*.abc"))
        
        if not abc_files:
            logger.warning(f"No ABC files found in {self.data_dir}")
            logger.info("Attempting to download dataset...")
            if self.download_dataset():
                abc_files = list(self.data_dir.glob("*.abc"))
        
        if not abc_files:
            logger.error("Could not find or download ABC files")
            return [], []
        
        logger.info(f"Found {len(abc_files)} ABC files")
        
        # Parse each file
        for abc_file in abc_files:
            logger.info(f"Parsing {abc_file.name}...")
            tunes = self.parse_abc_file(abc_file)
            
            for tune in tunes:
                if tune['notes']:
                    melodies.append(tune['notes'])
                    metadata.append({
                        'title': tune['title'],
                        'key': tune['key'],
                        'meter': tune['meter'],
                        'id': tune['id'],
                        'source_file': abc_file.name
                    })
                    
                    if max_tunes and len(melodies) >= max_tunes:
                        break
            
            if max_tunes and len(melodies) >= max_tunes:
                break
        
        logger.info(f"Loaded {len(melodies)} melodies from Nottingham dataset")
        
        self.melodies = melodies
        self.metadata = metadata
        
        return melodies, metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        if not self.melodies:
            return {}
        
        lengths = [len(m) for m in self.melodies]
        all_notes = [note for melody in self.melodies for note in melody]
        
        # Count keys
        key_counts = defaultdict(int)
        for meta in self.metadata:
            key_counts[meta['key']] += 1
        
        return {
            'num_melodies': len(self.melodies),
            'total_notes': len(all_notes),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'unique_notes': len(set(all_notes)),
            'note_range': (min(all_notes), max(all_notes)),
            'keys': dict(key_counts)
        }
    
    def save_processed_data(self, output_path: Path):
        """Save processed melodies and metadata to JSON."""
        data = {
            'melodies': self.melodies,
            'metadata': self.metadata,
            'statistics': self.get_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, input_path: Path) -> bool:
        """Load previously processed melodies and metadata from JSON."""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            self.melodies = data['melodies']
            self.metadata = data['metadata']
            
            logger.info(f"Loaded {len(self.melodies)} melodies from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return False


def split_dataset(
    melodies: List[List[int]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    min_length: int = 16,
    max_length: Optional[int] = None
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Split melodies into train, validation, and test sets.
    
    Args:
        melodies: List of melody sequences
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        min_length: Minimum melody length to include
        max_length: Maximum melody length to include
        
    Returns:
        Tuple of (train, val, test) melody lists
    """
    # Filter by length
    filtered = [
        m for m in melodies 
        if len(m) >= min_length and (max_length is None or len(m) <= max_length)
    ]
    
    logger.info(f"Filtered {len(melodies)} -> {len(filtered)} melodies (length: {min_length}-{max_length or 'inf'})")
    
    # Shuffle
    import random
    shuffled = filtered.copy()
    random.shuffle(shuffled)
    
    # Split
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    return train, val, test


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Nottingham Dataset Loader Test")
    print("=" * 80)
    
    loader = NottinghamDataLoader()
    
    # Download dataset
    print("\n1. Downloading dataset...")
    loader.download_dataset()
    
    # Load dataset
    print("\n2. Loading and parsing ABC files...")
    melodies, metadata = loader.load_dataset(max_tunes=100)
    
    # Show statistics
    print("\n3. Dataset Statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Show examples
    print("\n4. Example melodies:")
    for i in range(min(3, len(melodies))):
        print(f"\n   Melody {i+1}: {metadata[i]['title']}")
        print(f"   Key: {metadata[i]['key']}, Meter: {metadata[i]['meter']}")
        print(f"   Length: {len(melodies[i])} notes")
        print(f"   First 20 notes: {melodies[i][:20]}")
    
    # Split dataset
    print("\n5. Splitting dataset...")
    train, val, test = split_dataset(melodies, min_length=16, max_length=128)
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
