"""
Melody Generation Module
Contains ML models for generating coherent melodies from color-derived frequencies.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter
import json
from pathlib import Path

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class MelodyGenerator:
    """Generates melodies using various ML approaches."""
    
    def __init__(self, model_type: str = "markov"):
        """
        Initialize melody generator.
        
        Args:
            model_type: Type of model ("markov", "lstm", "transformer")
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Musical scale definitions (semitone intervals from root)
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10],
            'chromatic': list(range(12)),
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10]
        }
        
        # Initialize the appropriate model
        if model_type == "markov":
            self.model = MarkovChainModel()
        elif model_type == "lstm" and HAS_TORCH:
            self.model = LSTMModel()
        elif model_type == "transformer" and HAS_TRANSFORMERS:
            self.model = TransformerModel()
        else:
            self.logger.warning(f"Model type {model_type} not available, using Markov chain")
            self.model = MarkovChainModel()
    
    def generate_melody(
        self,
        frequencies: List[float],
        duration: float = 30.0,
        scale: str = "major",
        tempo: int = 120,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a melody from input frequencies.
        
        Args:
            frequencies: List of input frequencies in Hz
            duration: Duration of melody in seconds
            scale: Musical scale to use
            tempo: Tempo in BPM
            
        Returns:
            Dictionary containing melody information
        """
        # Convert frequencies to scale notes
        scale_notes = self._frequencies_to_scale_notes(frequencies, scale)
        
        # Calculate number of notes needed
        notes_per_second = tempo / 60 / 4  # Assuming quarter notes
        total_notes = int(duration * notes_per_second)
        
        # Generate melody using the selected model
        melody_notes = self.model.generate(
            seed_notes=scale_notes,
            length=total_notes,
            scale=scale,
            **kwargs
        )
        
        # Convert back to frequencies
        melody_frequencies = self._scale_notes_to_frequencies(melody_notes, scale)
        
        # Generate rhythm pattern
        rhythm = self._generate_rhythm_pattern(len(melody_notes), tempo)
        
        return {
            'notes': melody_notes,
            'frequencies': melody_frequencies,
            'rhythm': rhythm,
            'scale': scale,
            'tempo': tempo,
            'duration': duration,
            'seed_frequencies': frequencies,
            'seed_notes': scale_notes
        }
    
    def _frequencies_to_scale_notes(
        self, 
        frequencies: List[float], 
        scale: str
    ) -> List[int]:
        """Convert frequencies to scale note indices."""
        
        if scale not in self.scales:
            scale = 'major'
        
        scale_intervals = self.scales[scale]
        scale_notes = []
        
        for freq in frequencies:
            # Convert frequency to MIDI note
            if freq > 0:
                midi_note = int(69 + 12 * np.log2(freq / 440.0))
            else:
                midi_note = 60  # Middle C
            
            # Map to scale
            note_in_octave = midi_note % 12
            
            # Find closest scale note
            closest_scale_note = min(
                scale_intervals,
                key=lambda x: min(abs(note_in_octave - x), abs(note_in_octave - x + 12), abs(note_in_octave - x - 12))
            )
            
            scale_index = scale_intervals.index(closest_scale_note)
            scale_notes.append(scale_index)
        
        return scale_notes
    
    def _scale_notes_to_frequencies(
        self, 
        scale_notes: List[int], 
        scale: str,
        base_octave: int = 4
    ) -> List[float]:
        """Convert scale note indices back to frequencies."""
        
        if scale not in self.scales:
            scale = 'major'
        
        scale_intervals = self.scales[scale]
        frequencies = []
        
        for note_idx in scale_notes:
            if 0 <= note_idx < len(scale_intervals):
                semitones_from_c = scale_intervals[note_idx]
                midi_note = base_octave * 12 + semitones_from_c  # C4 = 60
                frequency = 440.0 * (2 ** ((midi_note - 69) / 12))
                frequencies.append(frequency)
            else:
                frequencies.append(440.0)  # Default A4
        
        return frequencies
    
    def _generate_rhythm_pattern(self, num_notes: int, tempo: int) -> List[float]:
        """Generate a rhythm pattern for the melody."""
        
        # Basic rhythm patterns (note durations in beats)
        patterns = [
            [1.0, 1.0, 0.5, 0.5],  # Quarter, quarter, eighth, eighth
            [2.0, 1.0, 1.0],       # Half, quarter, quarter
            [0.5, 0.5, 1.0, 2.0],  # Eighth, eighth, quarter, half
            [1.0, 0.5, 0.5, 1.0, 1.0],  # Quarter, eighth, eighth, quarter, quarter
        ]
        
        rhythm = []
        remaining_notes = num_notes
        
        while remaining_notes > 0:
            pattern = random.choice(patterns)
            for duration in pattern:
                if remaining_notes <= 0:
                    break
                rhythm.append(duration)
                remaining_notes -= 1
        
        return rhythm[:num_notes]


class MarkovChainModel:
    """Markov chain-based melody generator."""
    
    def __init__(self, order: int = 2):
        """
        Initialize Markov model.
        
        Args:
            order: Order of the Markov chain (how many previous notes to consider)
        """
        self.order = order
        self.transitions = defaultdict(list)
        self.trained = False
    
    def train(self, melodies: List[List[int]]):
        """Train the Markov model on example melodies."""
        
        self.transitions.clear()
        
        for melody in melodies:
            for i in range(len(melody) - self.order):
                state = tuple(melody[i:i + self.order])
                next_note = melody[i + self.order]
                self.transitions[state].append(next_note)
        
        self.trained = True
    
    def generate(
        self,
        seed_notes: List[int],
        length: int,
        scale: str = "major",
        **kwargs
    ) -> List[int]:
        """Generate a melody using the Markov model."""
        
        if not self.trained:
            # Use a simple pattern based on seed notes if not trained
            return self._generate_simple_pattern(seed_notes, length, scale)
        
        melody = list(seed_notes[:self.order])
        
        for _ in range(length - len(melody)):
            current_state = tuple(melody[-self.order:])
            
            if current_state in self.transitions:
                next_note = random.choice(self.transitions[current_state])
            else:
                # Fallback to a random note from the scale
                next_note = random.choice(seed_notes) if seed_notes else 0
            
            melody.append(next_note)
        
        return melody
    
    def _generate_simple_pattern(
        self, 
        seed_notes: List[int], 
        length: int, 
        scale: str
    ) -> List[int]:
        """Generate a simple pattern when no training data is available."""
        
        if not seed_notes:
            seed_notes = [0, 2, 4]  # Default pattern
        
        melody = []
        
        # Create a pattern based on the seed notes
        for i in range(length):
            # Use seed notes with some variation
            base_note = seed_notes[i % len(seed_notes)]
            
            # Add some musical movement
            if i > 0:
                prev_note = melody[i - 1]
                # Tendency to move by steps
                if random.random() < 0.3:  # 30% chance of step movement
                    if random.random() < 0.5:
                        base_note = (prev_note + 1) % len(seed_notes)
                    else:
                        base_note = (prev_note - 1) % len(seed_notes)
            
            melody.append(base_note)
        
        return melody


if HAS_TORCH:
    class LSTMModel(nn.Module):
        """LSTM-based melody generator."""
        
        def __init__(self, input_size: int = 12, hidden_size: int = 128, num_layers: int = 2):
            super(LSTMModel, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)
            self.softmax = nn.Softmax(dim=-1)
            
            self.trained = False
        
        def forward(self, x, hidden=None):
            lstm_out, hidden = self.lstm(x, hidden)
            output = self.fc(lstm_out)
            return self.softmax(output), hidden
        
        def generate(
            self,
            seed_notes: List[int],
            length: int,
            scale: str = "major",
            temperature: float = 1.0,
            **kwargs
        ) -> List[int]:
            """Generate melody using LSTM."""
            
            if not self.trained:
                # Fallback to simple generation
                return self._generate_simple(seed_notes, length, scale)
            
            self.eval()
            melody = list(seed_notes)
            
            # Convert to one-hot encoding
            with torch.no_grad():
                for _ in range(length - len(seed_notes)):
                    # Prepare input
                    input_seq = torch.zeros(1, 1, self.input_size)
                    if melody:
                        last_note = melody[-1] % self.input_size
                        input_seq[0, 0, last_note] = 1.0
                    
                    # Generate next note
                    output, _ = self.forward(input_seq)
                    probabilities = output[0, -1, :].numpy()
                    
                    # Apply temperature
                    if temperature != 1.0:
                        probabilities = np.power(probabilities, 1.0 / temperature)
                        probabilities = probabilities / np.sum(probabilities)
                    
                    # Sample next note
                    next_note = np.random.choice(len(probabilities), p=probabilities)
                    melody.append(next_note)
            
            return melody
        
        def _generate_simple(self, seed_notes: List[int], length: int, scale: str) -> List[int]:
            """Simple generation fallback."""
            melody = list(seed_notes)
            while len(melody) < length:
                if seed_notes:
                    next_note = random.choice(seed_notes)
                else:
                    next_note = random.randint(0, 6)
                melody.append(next_note)
            return melody
else:
    class LSTMModel:
        """Placeholder LSTM model when PyTorch is not available."""
        
        def __init__(self, **kwargs):
            self.trained = False
        
        def generate(self, seed_notes: List[int], length: int, **kwargs) -> List[int]:
            """Simple fallback generation."""
            melody = list(seed_notes)
            while len(melody) < length:
                if seed_notes:
                    next_note = random.choice(seed_notes)
                else:
                    next_note = random.randint(0, 6)
                melody.append(next_note)
            return melody


class TransformerModel:
    """Transformer-based melody generator (placeholder for now)."""
    
    def __init__(self):
        self.trained = False
    
    def generate(
        self,
        seed_notes: List[int],
        length: int,
        scale: str = "major",
        **kwargs
    ) -> List[int]:
        """Generate melody using transformer (simplified implementation)."""
        
        # For now, use a sophisticated pattern-based approach
        melody = list(seed_notes)
        scale_patterns = {
            'major': [0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 0],
            'minor': [0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 0],
            'pentatonic': [0, 1, 2, 3, 4, 0, 2, 4, 1, 3],
        }
        
        pattern = scale_patterns.get(scale, scale_patterns['major'])
        
        while len(melody) < length:
            # Use pattern with some randomness
            if len(melody) >= 2:
                # Look at the trend of last two notes
                trend = melody[-1] - melody[-2]
                if abs(trend) > 2:  # If large jump, return to center
                    next_note = random.choice(seed_notes)
                else:
                    # Continue pattern
                    next_note = pattern[(len(melody) - len(seed_notes)) % len(pattern)]
            else:
                next_note = pattern[len(melody) % len(pattern)]
            
            melody.append(next_note)
        
        return melody