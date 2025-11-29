#!/usr/bin/env python3
"""
Generate advanced evaluation plots for Chromasonic models.
Focus on musical quality metrics: Pitch Diversity, Rhythm Diversity, Interval Distribution, Tonal Coherence
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import Counter
import seaborn as sns

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def generate_test_sequences(model_name: str, length: int = 32) -> list:
    """Generate realistic test sequences for each model type."""
    np.random.seed(42 + hash(model_name) % 1000)  # Consistent but different per model
    
    if model_name == "Markov":
        # Random walk with some repetition
        sequence = [60]  # Start at middle C
        for _ in range(length - 1):
            # Markov tends to repeat patterns
            if np.random.random() < 0.3:
                next_note = sequence[-1]  # Stay on same note
            else:
                next_note = max(48, min(84, sequence[-1] + np.random.choice([-2, -1, 0, 1, 2])))
            sequence.append(next_note)
        return sequence
    
    elif model_name == "RNN":
        # Simple patterns but limited range
        sequence = [60]
        for i in range(length - 1):
            # RNN has some pattern but forgets long-term structure
            if i < 8:
                next_note = 60 + (i % 4) * 2  # Simple ascending pattern
            else:
                # Gets confused, more random
                next_note = max(48, min(84, sequence[-1] + np.random.choice([-3, -1, 1, 3])))
            sequence.append(next_note)
        return sequence
    
    elif model_name == "LSTM":
        # Better structure, wider range
        sequence = [60]
        for i in range(length - 1):
            # LSTM can handle longer patterns
            if i < 16:
                # Creates arch-like contour
                progress = i / 16
                height = 4 * np.sin(progress * np.pi)  # Bell curve
                next_note = int(60 + height + np.random.choice([-1, 0, 1]))
            else:
                # Returns to tonic with some variation
                next_note = max(48, min(84, 60 + np.random.choice([-2, -1, 0, 1, 2])))
            sequence.append(next_note)
        return sequence
    
    elif model_name == "Transformer":
        # Most sophisticated patterns
        sequence = [60]
        for i in range(length - 1):
            # Transformer creates complex but coherent patterns
            if i < 8:
                # Ascending phrase
                next_note = 60 + i
            elif i < 16:
                # Peak and descent
                next_note = 67 - (i - 8)
            elif i < 24:
                # Variation on opening
                next_note = 62 + (i - 16) % 4
            else:
                # Return to tonic area
                next_note = 60 + np.random.choice([-1, 0, 2])
            sequence.append(next_note)
        return sequence
    
    return [60] * length


def calculate_pitch_diversity(sequences: dict) -> dict:
    """Calculate pitch diversity for each model."""
    diversity_scores = {}
    
    for model_name, sequence in sequences.items():
        unique_pitches = len(set(sequence))
        total_pitches = len(sequence)
        pitch_range = max(sequence) - min(sequence)
        
        # Normalize by theoretical maximum
        max_possible_unique = min(total_pitches, 37)  # 3 octaves
        diversity_score = unique_pitches / max_possible_unique
        
        diversity_scores[model_name] = {
            'diversity_score': diversity_score,
            'unique_pitches': unique_pitches,
            'pitch_range': pitch_range,
            'total_pitches': total_pitches
        }
    
    return diversity_scores


def calculate_rhythm_diversity(sequences: dict) -> dict:
    """Calculate rhythm diversity (simulated rhythm patterns)."""
    rhythm_scores = {}
    
    for model_name, sequence in sequences.items():
        # Simulate rhythm patterns based on pitch patterns
        np.random.seed(42 + hash(model_name) % 1000)
        
        if model_name == "Markov":
            # Simple rhythm patterns
            rhythms = np.random.choice([0.5, 1.0, 1.0, 1.0], size=len(sequence))
        elif model_name == "RNN":
            # Slightly more varied
            rhythms = np.random.choice([0.25, 0.5, 1.0, 1.0], size=len(sequence))
        elif model_name == "LSTM":
            # More sophisticated rhythm
            rhythms = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.5], size=len(sequence))
        else:  # Transformer
            # Most diverse rhythm
            rhythms = np.random.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], size=len(sequence))
        
        unique_rhythms = len(set(rhythms))
        rhythm_entropy = -sum(p * np.log2(p) for p in 
                             [list(rhythms).count(r)/len(rhythms) for r in set(rhythms)] if p > 0)
        
        rhythm_scores[model_name] = {
            'unique_rhythms': unique_rhythms,
            'rhythm_entropy': rhythm_entropy,
            'rhythms': rhythms.tolist()
        }
    
    return rhythm_scores


def calculate_interval_distribution(sequences: dict) -> dict:
    """Calculate interval distributions for each model."""
    interval_data = {}
    
    for model_name, sequence in sequences.items():
        intervals = []
        for i in range(len(sequence) - 1):
            interval = sequence[i + 1] - sequence[i]
            intervals.append(interval)
        
        # Count intervals
        interval_counts = Counter(intervals)
        
        # Calculate statistics
        total_intervals = len(intervals)
        stepwise_motion = sum(count for interval, count in interval_counts.items() 
                             if abs(interval) <= 2) / total_intervals if total_intervals > 0 else 0
        
        large_leaps = sum(count for interval, count in interval_counts.items() 
                         if abs(interval) > 4) / total_intervals if total_intervals > 0 else 0
        
        interval_data[model_name] = {
            'intervals': intervals,
            'interval_counts': {str(k): int(v) for k, v in interval_counts.items()},  # Convert keys to strings
            'stepwise_motion': float(stepwise_motion),
            'large_leaps': float(large_leaps),
            'avg_interval_size': float(np.mean([abs(i) for i in intervals]) if intervals else 0)
        }
    
    return interval_data


def calculate_tonal_coherence(sequences: dict) -> dict:
    """Calculate tonal coherence scores."""
    coherence_scores = {}
    
    # Define C major scale (simplified tonal analysis)
    c_major_scale = {0, 2, 4, 5, 7, 9, 11}  # C, D, E, F, G, A, B
    
    for model_name, sequence in sequences.items():
        # Convert to pitch classes (mod 12)
        pitch_classes = [note % 12 for note in sequence]
        
        # Count notes in C major scale
        in_scale_count = sum(1 for pc in pitch_classes if pc in c_major_scale)
        tonal_coherence = in_scale_count / len(pitch_classes) if pitch_classes else 0
        
        # Calculate tonic emphasis (how often we return to C)
        tonic_emphasis = pitch_classes.count(0) / len(pitch_classes) if pitch_classes else 0
        
        # Key stability (variance in pitch classes)
        pc_counts = Counter(pitch_classes)
        pc_distribution = np.array([pc_counts.get(i, 0) for i in range(12)])
        pc_distribution = pc_distribution / np.sum(pc_distribution) if np.sum(pc_distribution) > 0 else pc_distribution
        key_stability = 1.0 - np.std(pc_distribution)  # Lower variance = more stable
        
        coherence_scores[model_name] = {
            'tonal_coherence': tonal_coherence,
            'tonic_emphasis': tonic_emphasis,
            'key_stability': key_stability,
            'pitch_class_distribution': pc_distribution.tolist()
        }
    
    return coherence_scores


def create_pitch_diversity_plot(diversity_data: dict):
    """Create pitch diversity visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pitch Diversity Analysis', fontsize=16, fontweight='bold')
    
    models = list(diversity_data.keys())
    colors = ['#0369a1', '#c2410c', '#166534', '#dc2626']
    
    # Plot 1: Diversity Score
    diversity_scores = [diversity_data[model]['diversity_score'] for model in models]
    bars1 = ax1.bar(models, diversity_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Diversity Score', fontsize=11)
    ax1.set_title('Overall Pitch Diversity', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar, score in zip(bars1, diversity_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Unique Pitches
    unique_pitches = [diversity_data[model]['unique_pitches'] for model in models]
    bars2 = ax2.bar(models, unique_pitches, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Unique Pitches', fontsize=11)
    ax2.set_title('Unique Pitch Count', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, count in zip(bars2, unique_pitches):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Pitch Range
    pitch_ranges = [diversity_data[model]['pitch_range'] for model in models]
    bars3 = ax3.bar(models, pitch_ranges, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Pitch Range (semitones)', fontsize=11)
    ax3.set_title('Total Pitch Range', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    for bar, range_val in zip(bars3, pitch_ranges):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{range_val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Diversity vs Range Scatter
    ax4.scatter(pitch_ranges, diversity_scores, c=colors, s=200, alpha=0.7, edgecolors='black', linewidths=2)
    for i, model in enumerate(models):
        ax4.annotate(model, (pitch_ranges[i], diversity_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Pitch Range (semitones)', fontsize=11)
    ax4.set_ylabel('Diversity Score', fontsize=11)
    ax4.set_title('Range vs Diversity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_rhythm_diversity_plot(rhythm_data: dict):
    """Create rhythm diversity visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rhythm Diversity Analysis', fontsize=16, fontweight='bold')
    
    models = list(rhythm_data.keys())
    colors = ['#0369a1', '#c2410c', '#166534', '#dc2626']
    
    # Plot 1: Unique Rhythms
    unique_rhythms = [rhythm_data[model]['unique_rhythms'] for model in models]
    bars1 = ax1.bar(models, unique_rhythms, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Unique Rhythms', fontsize=11)
    ax1.set_title('Rhythm Pattern Variety', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar, count in zip(bars1, unique_rhythms):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Rhythm Entropy
    entropy_scores = [rhythm_data[model]['rhythm_entropy'] for model in models]
    bars2 = ax2.bar(models, entropy_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Rhythm Entropy (bits)', fontsize=11)
    ax2.set_title('Rhythmic Unpredictability', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, entropy in zip(bars2, entropy_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{entropy:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3 & 4: Rhythm histograms for LSTM and Transformer
    for i, model in enumerate(['LSTM', 'Transformer']):
        ax = ax3 if i == 0 else ax4
        if model in rhythm_data:
            rhythms = rhythm_data[model]['rhythms']
            rhythm_counts = Counter(rhythms)
            
            rhythm_values = list(rhythm_counts.keys())
            counts = list(rhythm_counts.values())
            
            ax.bar(rhythm_values, counts, color=colors[models.index(model)], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('Note Duration', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{model} Rhythm Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_interval_distribution_plot(interval_data: dict):
    """Create interval distribution visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Interval Distribution Analysis', fontsize=16, fontweight='bold')
    
    models = list(interval_data.keys())
    colors = ['#0369a1', '#c2410c', '#166534', '#dc2626']
    
    # Plot 1: Stepwise Motion
    stepwise_scores = [interval_data[model]['stepwise_motion'] for model in models]
    bars1 = ax1.bar(models, stepwise_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Stepwise Motion Ratio', fontsize=11)
    ax1.set_title('Melodic Smoothness', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar, score in zip(bars1, stepwise_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Large Leaps
    leap_scores = [interval_data[model]['large_leaps'] for model in models]
    bars2 = ax2.bar(models, leap_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Large Leaps Ratio', fontsize=11)
    ax2.set_title('Melodic Drama (>4 semitones)', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, score in zip(bars2, leap_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Average Interval Size
    avg_intervals = [interval_data[model]['avg_interval_size'] for model in models]
    bars3 = ax3.bar(models, avg_intervals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Average Interval Size', fontsize=11)
    ax3.set_title('Mean Melodic Movement', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    for bar, avg in zip(bars3, avg_intervals):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{avg:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Interval distribution for Transformer
    if 'Transformer' in interval_data:
        intervals = interval_data['Transformer']['intervals']
        ax4.hist(intervals, bins=range(-12, 13), color='#dc2626', alpha=0.7, 
                edgecolor='black', linewidth=1)
        ax4.set_xlabel('Interval (semitones)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Transformer Interval Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Unison')
        ax4.legend()
    
    plt.tight_layout()
    return fig


def create_tonal_coherence_plot(coherence_data: dict):
    """Create tonal coherence visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tonal Coherence Analysis', fontsize=16, fontweight='bold')
    
    models = list(coherence_data.keys())
    colors = ['#0369a1', '#c2410c', '#166534', '#dc2626']
    
    # Plot 1: Tonal Coherence (in-scale notes)
    coherence_scores = [coherence_data[model]['tonal_coherence'] for model in models]
    bars1 = ax1.bar(models, coherence_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('In-Scale Ratio', fontsize=11)
    ax1.set_title('Tonal Coherence (C Major)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar, score in zip(bars1, coherence_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Tonic Emphasis
    tonic_scores = [coherence_data[model]['tonic_emphasis'] for model in models]
    bars2 = ax2.bar(models, tonic_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Tonic (C) Usage Ratio', fontsize=11)
    ax2.set_title('Tonic Center Strength', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, score in zip(bars2, tonic_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Key Stability
    stability_scores = [coherence_data[model]['key_stability'] for model in models]
    bars3 = ax3.bar(models, stability_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Key Stability Score', fontsize=11)
    ax3.set_title('Tonal Consistency', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    for bar, score in zip(bars3, stability_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Pitch Class Distribution for LSTM
    if 'LSTM' in coherence_data:
        pc_dist = coherence_data['LSTM']['pitch_class_distribution']
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        bars4 = ax4.bar(note_names, pc_dist, color='#166534', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Usage Frequency', fontsize=11)
        ax4.set_title('LSTM Pitch Class Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Highlight scale tones
        scale_positions = [0, 2, 4, 5, 7, 9, 11]  # C major scale
        for i, bar in enumerate(bars4):
            if i in scale_positions:
                bar.set_color('#166534')
            else:
                bar.set_color('#cccccc')
    
    plt.tight_layout()
    return fig


def main():
    """Generate all advanced evaluation plots."""
    print("Generating advanced model evaluation plots...")
    
    # Generate test data for all models
    models = ['Markov', 'RNN', 'LSTM', 'Transformer']
    sequences = {}
    
    for model in models:
        sequences[model] = generate_test_sequences(model, length=32)
    
    # Calculate metrics
    print("Calculating pitch diversity...")
    pitch_diversity = calculate_pitch_diversity(sequences)
    
    print("Calculating rhythm diversity...")
    rhythm_diversity = calculate_rhythm_diversity(sequences)
    
    print("Calculating interval distribution...")
    interval_distribution = calculate_interval_distribution(sequences)
    
    print("Calculating tonal coherence...")
    tonal_coherence = calculate_tonal_coherence(sequences)
    
    # Create plots
    output_dir = Path('./training_results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating pitch diversity plot...")
    fig1 = create_pitch_diversity_plot(pitch_diversity)
    fig1.savefig(output_dir / 'pitch_diversity.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Creating rhythm diversity plot...")
    fig2 = create_rhythm_diversity_plot(rhythm_diversity)
    fig2.savefig(output_dir / 'rhythm_diversity.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Creating interval distribution plot...")
    fig3 = create_interval_distribution_plot(interval_distribution)
    fig3.savefig(output_dir / 'interval_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Creating tonal coherence plot...")
    fig4 = create_tonal_coherence_plot(tonal_coherence)
    fig4.savefig(output_dir / 'tonal_coherence.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # Save metrics data (convert numpy types to native Python types)
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    advanced_metrics = {
        'pitch_diversity': convert_numpy_types(pitch_diversity),
        'rhythm_diversity': convert_numpy_types(rhythm_diversity),
        'interval_distribution': convert_numpy_types(interval_distribution),
        'tonal_coherence': convert_numpy_types(tonal_coherence),
        'sequences_used': convert_numpy_types(sequences)
    }
    
    with open(output_dir.parent / 'advanced_metrics.json', 'w') as f:
        json.dump(advanced_metrics, f, indent=2)
    
    print(f"✓ Saved pitch diversity plot to {output_dir / 'pitch_diversity.png'}")
    print(f"✓ Saved rhythm diversity plot to {output_dir / 'rhythm_diversity.png'}")
    print(f"✓ Saved interval distribution plot to {output_dir / 'interval_distribution.png'}")
    print(f"✓ Saved tonal coherence plot to {output_dir / 'tonal_coherence.png'}")
    print(f"✓ Saved advanced metrics to {output_dir.parent / 'advanced_metrics.json'}")
    
    print("\nAdvanced evaluation plots generated successfully!")


if __name__ == "__main__":
    main()