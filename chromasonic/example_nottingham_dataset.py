#!/usr/bin/env python3
"""
Example: Training Models with Nottingham Dataset

This script demonstrates how to train melody generation models
using the Nottingham folk music dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chromasonic.melody_generation.data_loader import (
    NottinghamDataLoader,
    split_dataset
)


def main():
    print("=" * 80)
    print("Nottingham Dataset Example")
    print("=" * 80)
    
    # 1. Create loader
    print("\n1. Creating data loader...")
    loader = NottinghamDataLoader()
    
    # 2. Download dataset (if not already downloaded)
    print("\n2. Downloading Nottingham dataset...")
    success = loader.download_dataset()
    
    if not success:
        print("❌ Failed to download dataset")
        return
    
    print("✓ Dataset downloaded successfully")
    
    # 3. Load and parse ABC files
    print("\n3. Loading and parsing ABC files...")
    melodies, metadata = loader.load_dataset(max_tunes=100)  # Load first 100 for demo
    
    print(f"✓ Loaded {len(melodies)} melodies")
    
    # 4. Show statistics
    print("\n4. Dataset Statistics:")
    stats = loader.get_statistics()
    print(f"   Number of melodies: {stats['num_melodies']}")
    print(f"   Total notes: {stats['total_notes']}")
    print(f"   Average length: {stats['avg_length']:.1f} notes")
    print(f"   Min length: {stats['min_length']} notes")
    print(f"   Max length: {stats['max_length']} notes")
    print(f"   Unique notes: {stats['unique_notes']}")
    print(f"   Note range: {stats['note_range']}")
    
    print("\n   Most common keys:")
    sorted_keys = sorted(stats['keys'].items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_keys[:5]:
        print(f"     {key}: {count} melodies")
    
    # 5. Show example melodies
    print("\n5. Example Melodies:")
    for i in range(min(3, len(melodies))):
        print(f"\n   Melody {i+1}:")
        print(f"     Title: {metadata[i]['title']}")
        print(f"     Key: {metadata[i]['key']}")
        print(f"     Meter: {metadata[i]['meter']}")
        print(f"     Length: {len(melodies[i])} notes")
        print(f"     First 20 notes: {melodies[i][:20]}")
    
    # 6. Split dataset
    print("\n6. Splitting dataset...")
    train, val, test = split_dataset(
        melodies,
        train_ratio=0.8,
        val_ratio=0.1,
        min_length=16,
        max_length=128
    )
    
    print(f"   Train: {len(train)} melodies")
    print(f"   Validation: {len(val)} melodies")
    print(f"   Test: {len(test)} melodies")
    
    # 7. Save processed data
    print("\n7. Saving processed data...")
    output_path = loader.data_dir / "processed_melodies_demo.json"
    loader.save_processed_data(output_path)
    print(f"   ✓ Saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("✓ Demo Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Train models with: python train_models.py --dataset nottingham")
    print("  2. Compare with synthetic: python train_models.py --dataset synthetic")
    print("  3. Check data/README.md for more information")
    print()


if __name__ == "__main__":
    main()
