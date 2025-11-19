"""
Chords and Instrumentation Module
Generates chord progressions and selects appropriate instruments based on image characteristics.
"""

import logging
import random
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from enum import Enum


class ChordType(Enum):
    """Available chord types."""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    MAJOR7 = "major7"
    MINOR7 = "minor7"
    DOMINANT7 = "dominant7"


class InstrumentCategory(Enum):
    """Instrument categories."""
    PIANO = "piano"
    STRINGS = "strings"
    BRASS = "brass"
    WOODWINDS = "woodwinds"
    SYNTHESIZER = "synthesizer"
    PERCUSSION = "percussion"
    GUITAR = "guitar"
    ORGAN = "organ"


class ChordProgressionGenerator:
    """Generates chord progressions based on musical theory and image characteristics."""
    
    def __init__(self):
        """Initialize chord progression generator."""
        self.logger = logging.getLogger(__name__)
        
        # Chord intervals (semitones from root)
        self.chord_intervals = {
            ChordType.MAJOR: [0, 4, 7],
            ChordType.MINOR: [0, 3, 7],
            ChordType.DIMINISHED: [0, 3, 6],
            ChordType.AUGMENTED: [0, 4, 8],
            ChordType.MAJOR7: [0, 4, 7, 11],
            ChordType.MINOR7: [0, 3, 7, 10],
            ChordType.DOMINANT7: [0, 4, 7, 10]
        }
        
        # Common chord progressions by key
        self.progressions = {
            'major': {
                'happy': [[0, 'major'], [5, 'major'], [3, 'minor'], [4, 'major']],  # I-V-vi-IV
                'hopeful': [[0, 'major'], [3, 'minor'], [4, 'major'], [5, 'major']],  # I-vi-IV-V
                'peaceful': [[0, 'major'], [4, 'major'], [0, 'major'], [5, 'major']],  # I-IV-I-V
                'uplifting': [[5, 'major'], [3, 'minor'], [4, 'major'], [0, 'major']]   # V-vi-IV-I
            },
            'minor': {
                'melancholic': [[0, 'minor'], [6, 'major'], [3, 'major'], [7, 'major']],  # i-VI-III-VII
                'dramatic': [[0, 'minor'], [7, 'major'], [0, 'minor'], [7, 'major']],   # i-VII-i-VII
                'mysterious': [[0, 'minor'], [2, 'diminished'], [5, 'minor'], [0, 'minor']],  # i-ii°-v-i
                'contemplative': [[0, 'minor'], [3, 'major'], [6, 'major'], [7, 'major']]   # i-III-VI-VII
            }
        }
    
    def generate_progression(
        self,
        key: int,
        mode: str,
        length: int = 8,
        image_features: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, ChordType]]:
        """
        Generate a chord progression.
        
        Args:
            key: Root key (0-11, where 0 = C)
            mode: Major or minor mode
            length: Number of chords in progression
            image_features: Image features to influence progression style
            
        Returns:
            List of (root_note, chord_type) tuples
        """
        
        # Select progression style based on image features
        style = self._select_progression_style(image_features, mode)
        
        # Get base progression pattern
        if mode in self.progressions and style in self.progressions[mode]:
            base_pattern = self.progressions[mode][style]
        else:
            # Fallback to simple I-V-vi-IV for major, i-VI-III-VII for minor
            if mode == 'major':
                base_pattern = [[0, 'major'], [5, 'major'], [3, 'minor'], [4, 'major']]
            else:
                base_pattern = [[0, 'minor'], [6, 'major'], [3, 'major'], [7, 'major']]
        
        # Extend pattern to desired length
        progression = []
        for i in range(length):
            pattern_index = i % len(base_pattern)
            degree, chord_type_str = base_pattern[pattern_index]
            
            # Convert scale degree to actual note
            chord_root = (key + degree) % 12
            chord_type = ChordType(chord_type_str)
            
            progression.append((chord_root, chord_type))
        
        # Add variations based on image features
        progression = self._add_variations(progression, image_features)
        
        self.logger.info(f"Generated {length}-chord progression in {mode} mode")
        return progression
    
    def _select_progression_style(
        self, 
        image_features: Optional[Dict[str, Any]], 
        mode: str
    ) -> str:
        """Select progression style based on image characteristics."""
        
        if not image_features:
            return 'happy' if mode == 'major' else 'melancholic'
        
        # Map image features to musical moods
        brightness = image_features.get('brightness', 0.5)
        saturation = image_features.get('mean_saturation', 0.5)
        complexity = image_features.get('complexity_score', 0.5)
        
        if mode == 'major':
            if brightness > 0.7 and saturation > 0.6:
                return 'happy'
            elif brightness > 0.6:
                return 'uplifting'
            elif complexity < 0.4:
                return 'peaceful'
            else:
                return 'hopeful'
        else:  # minor mode
            if complexity > 0.6:
                return 'dramatic'
            elif brightness < 0.3:
                return 'mysterious'
            elif saturation < 0.4:
                return 'contemplative'
            else:
                return 'melancholic'
    
    def _add_variations(
        self, 
        progression: List[Tuple[int, ChordType]], 
        image_features: Optional[Dict[str, Any]]
    ) -> List[Tuple[int, ChordType]]:
        """Add variations to the base progression based on image features."""
        
        if not image_features:
            return progression
        
        varied_progression = []
        complexity = image_features.get('complexity_score', 0.5)
        
        for i, (root, chord_type) in enumerate(progression):
            
            # Add seventh chords for higher complexity
            if complexity > 0.6 and random.random() < 0.3:
                if chord_type == ChordType.MAJOR:
                    chord_type = ChordType.MAJOR7
                elif chord_type == ChordType.MINOR:
                    chord_type = ChordType.MINOR7
            
            # Occasionally substitute chords
            if complexity > 0.4 and i > 0 and random.random() < 0.2:
                # Simple substitutions
                if chord_type == ChordType.MAJOR:
                    # Substitute with relative minor (down a minor third)
                    root = (root - 3) % 12
                    chord_type = ChordType.MINOR
            
            varied_progression.append((root, chord_type))
        
        return varied_progression
    
    def progression_to_notes(
        self, 
        progression: List[Tuple[int, ChordType]], 
        octave: int = 4
    ) -> List[List[int]]:
        """
        Convert chord progression to MIDI note lists.
        
        Args:
            progression: List of (root, chord_type) tuples
            octave: Base octave for chords
            
        Returns:
            List of chord note lists (MIDI note numbers)
        """
        
        chord_notes = []
        
        for root, chord_type in progression:
            intervals = self.chord_intervals[chord_type]
            
            # Convert to MIDI notes
            midi_notes = []
            for interval in intervals:
                midi_note = octave * 12 + root + interval
                midi_notes.append(midi_note)
            
            chord_notes.append(midi_notes)
        
        return chord_notes


class InstrumentSelector:
    """Selects appropriate instruments based on image characteristics and musical context."""
    
    def __init__(self):
        """Initialize instrument selector."""
        self.logger = logging.getLogger(__name__)
        
        # Instrument mappings (General MIDI program numbers)
        self.instruments = {
            InstrumentCategory.PIANO: {
                'acoustic_grand': 0,
                'bright_acoustic': 1,
                'electric_grand': 2,
                'honky_tonk': 3,
                'electric_piano1': 4,
                'electric_piano2': 5,
            },
            InstrumentCategory.STRINGS: {
                'violin': 40,
                'viola': 41,
                'cello': 42,
                'contrabass': 43,
                'string_ensemble1': 48,
                'string_ensemble2': 49,
                'synth_strings1': 50,
                'synth_strings2': 51,
            },
            InstrumentCategory.BRASS: {
                'trumpet': 56,
                'trombone': 57,
                'tuba': 58,
                'french_horn': 60,
                'brass_section': 61,
                'synth_brass1': 62,
                'synth_brass2': 63,
            },
            InstrumentCategory.WOODWINDS: {
                'flute': 73,
                'clarinet': 71,
                'oboe': 68,
                'bassoon': 70,
                'saxophone': 64,
            },
            InstrumentCategory.SYNTHESIZER: {
                'lead_1_square': 80,
                'lead_2_sawtooth': 81,
                'lead_3_calliope': 82,
                'lead_4_chiff': 83,
                'pad_1_new_age': 88,
                'pad_2_warm': 89,
                'pad_3_polysynth': 90,
                'pad_4_choir': 91,
            },
            InstrumentCategory.GUITAR: {
                'acoustic_guitar_nylon': 24,
                'acoustic_guitar_steel': 25,
                'electric_guitar_jazz': 26,
                'electric_guitar_clean': 27,
                'electric_guitar_muted': 28,
            },
            InstrumentCategory.ORGAN: {
                'drawbar_organ': 16,
                'percussive_organ': 17,
                'rock_organ': 18,
                'church_organ': 19,
            },
            InstrumentCategory.PERCUSSION: {
                'drums': 128,  # Channel 10 for drums
            }
        }
        
        # Emotional mapping for instruments
        self.emotion_instruments = {
            'bright': [InstrumentCategory.PIANO, InstrumentCategory.BRASS, InstrumentCategory.WOODWINDS],
            'warm': [InstrumentCategory.STRINGS, InstrumentCategory.GUITAR, InstrumentCategory.ORGAN],
            'cool': [InstrumentCategory.SYNTHESIZER, InstrumentCategory.PIANO],
            'dark': [InstrumentCategory.STRINGS, InstrumentCategory.ORGAN, InstrumentCategory.SYNTHESIZER],
            'energetic': [InstrumentCategory.SYNTHESIZER, InstrumentCategory.BRASS, InstrumentCategory.PERCUSSION],
            'peaceful': [InstrumentCategory.PIANO, InstrumentCategory.STRINGS, InstrumentCategory.WOODWINDS]
        }
    
    def select_instruments(
        self,
        image_features: Dict[str, Any],
        num_tracks: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Select instruments for different tracks based on image characteristics.
        
        Args:
            image_features: Extracted image features
            num_tracks: Number of instrument tracks to select
            
        Returns:
            Dictionary mapping track names to instrument information
        """
        
        # Analyze image emotional content
        emotion = self._analyze_image_emotion(image_features)
        
        # Select instrument categories
        categories = self._select_instrument_categories(emotion, num_tracks)
        
        # Select specific instruments
        instruments = {}
        
        for i, category in enumerate(categories):
            track_name = f"track_{i+1}"
            
            # Select specific instrument from category
            instrument_dict = self.instruments[category]
            instrument_name = random.choice(list(instrument_dict.keys()))
            program_number = instrument_dict[instrument_name]
            
            instruments[track_name] = {
                'category': category.value,
                'name': instrument_name,
                'program': program_number,
                'channel': i + 1,  # MIDI channels 1-16
                'volume': self._calculate_track_volume(category, image_features),
                'pan': self._calculate_track_pan(i, num_tracks)
            }
        
        # Always add drums if we have space and energy is high
        if (num_tracks > 2 and 
            image_features.get('energy', 0.5) > 0.6 and 
            'percussion' not in [inst['category'] for inst in instruments.values()]):
            
            instruments['drums'] = {
                'category': InstrumentCategory.PERCUSSION.value,
                'name': 'drums',
                'program': 128,
                'channel': 10,  # Standard drum channel
                'volume': 80,
                'pan': 64  # Center
            }
        
        self.logger.info(f"Selected {len(instruments)} instruments for emotion: {emotion}")
        return instruments
    
    def _analyze_image_emotion(self, image_features: Dict[str, Any]) -> str:
        """Analyze emotional content of image."""
        
        brightness = image_features.get('brightness', 0.5)
        saturation = image_features.get('mean_saturation', 0.5)
        temperature = image_features.get('color_temperature', 0.5)
        energy = image_features.get('energy', 0.5)
        
        # Decision tree for emotion classification
        if energy > 0.7:
            return 'energetic'
        elif brightness > 0.7 and saturation > 0.6:
            return 'bright'
        elif brightness < 0.3 or saturation < 0.3:
            return 'dark'
        elif temperature > 0.6:  # Warm colors dominant
            return 'warm'
        elif temperature < 0.4:  # Cool colors dominant
            return 'cool'
        else:
            return 'peaceful'
    
    def _select_instrument_categories(
        self, 
        emotion: str, 
        num_tracks: int
    ) -> List[InstrumentCategory]:
        """Select instrument categories based on emotion."""
        
        # Get preferred categories for this emotion
        preferred = self.emotion_instruments.get(emotion, [
            InstrumentCategory.PIANO, 
            InstrumentCategory.STRINGS
        ])
        
        # Ensure we have enough categories
        all_categories = list(InstrumentCategory)
        if len(preferred) < num_tracks:
            # Add additional categories not already in preferred
            additional = [cat for cat in all_categories if cat not in preferred]
            preferred.extend(additional[:num_tracks - len(preferred)])
        
        # Select requested number of categories
        selected = preferred[:num_tracks]
        
        # Ensure we always have a melody instrument (piano/strings/synth)
        melody_instruments = [
            InstrumentCategory.PIANO, 
            InstrumentCategory.STRINGS, 
            InstrumentCategory.SYNTHESIZER
        ]
        
        if not any(cat in melody_instruments for cat in selected):
            # Replace last category with a melody instrument
            selected[-1] = InstrumentCategory.PIANO
        
        return selected
    
    def _calculate_track_volume(
        self, 
        category: InstrumentCategory, 
        image_features: Dict[str, Any]
    ) -> int:
        """Calculate appropriate volume for instrument category."""
        
        base_volumes = {
            InstrumentCategory.PIANO: 100,
            InstrumentCategory.STRINGS: 90,
            InstrumentCategory.BRASS: 85,
            InstrumentCategory.WOODWINDS: 80,
            InstrumentCategory.SYNTHESIZER: 95,
            InstrumentCategory.GUITAR: 85,
            InstrumentCategory.ORGAN: 90,
            InstrumentCategory.PERCUSSION: 80
        }
        
        base_volume = base_volumes.get(category, 80)
        
        # Adjust based on image characteristics
        energy = image_features.get('energy', 0.5)
        brightness = image_features.get('brightness', 0.5)
        
        # Brighter, more energetic images get louder instruments
        volume_modifier = (energy + brightness) / 2.0
        adjusted_volume = int(base_volume * (0.7 + 0.6 * volume_modifier))
        
        return max(40, min(127, adjusted_volume))
    
    def _calculate_track_pan(self, track_index: int, total_tracks: int) -> int:
        """Calculate stereo pan position for track."""
        
        if total_tracks == 1:
            return 64  # Center
        
        # Distribute tracks across stereo field
        pan_positions = np.linspace(20, 108, total_tracks)  # Avoid extreme edges
        return int(pan_positions[track_index])


class ArrangementGenerator:
    """Generates musical arrangements combining melody, chords, and instrumentation."""
    
    def __init__(self):
        """Initialize arrangement generator."""
        self.logger = logging.getLogger(__name__)
        self.chord_generator = ChordProgressionGenerator()
        self.instrument_selector = InstrumentSelector()
    
    def create_arrangement(
        self,
        melody: List[int],
        key: int,
        mode: str,
        image_features: Dict[str, Any],
        tempo: int = 120
    ) -> Dict[str, Any]:
        """
        Create a complete musical arrangement.
        
        Args:
            melody: Melody note sequence
            key: Musical key (0-11)
            mode: Major or minor mode
            image_features: Image characteristics
            tempo: Tempo in BPM
            
        Returns:
            Complete arrangement with melody, chords, and instrumentation
        """
        
        # Generate chord progression
        num_chords = max(4, len(melody) // 4)  # One chord per 4 melody notes
        chord_progression = self.chord_generator.generate_progression(
            key, mode, num_chords, image_features
        )
        
        # Convert chords to MIDI notes
        chord_notes = self.chord_generator.progression_to_notes(chord_progression)
        
        # Select instruments
        instruments = self.instrument_selector.select_instruments(image_features)
        
        # Create track assignments
        tracks = {}
        
        # Melody track
        if 'track_1' in instruments:
            tracks['melody'] = {
                'notes': melody,
                'instrument': instruments['track_1'],
                'type': 'melody'
            }
        
        # Chord track
        if 'track_2' in instruments:
            tracks['chords'] = {
                'notes': chord_notes,
                'instrument': instruments['track_2'],
                'type': 'chords'
            }
        
        # Bass line (simplified - just chord roots)
        if 'track_3' in instruments:
            bass_notes = [chord[0] - 12 for chord in chord_notes]  # One octave lower
            tracks['bass'] = {
                'notes': bass_notes,
                'instrument': instruments['track_3'],
                'type': 'bass'
            }
        
        # Drums (if selected)
        if 'drums' in instruments:
            drum_pattern = self._create_drum_pattern(len(melody), tempo)
            tracks['drums'] = {
                'notes': drum_pattern,
                'instrument': instruments['drums'],
                'type': 'drums'
            }
        
        arrangement = {
            'tracks': tracks,
            'chord_progression': chord_progression,
            'key': key,
            'mode': mode,
            'tempo': tempo,
            'time_signature': [4, 4],  # 4/4 time
            'metadata': {
                'emotion': self.instrument_selector._analyze_image_emotion(image_features),
                'complexity': image_features.get('complexity_score', 0.5)
            }
        }
        
        self.logger.info(f"Created arrangement with {len(tracks)} tracks")
        return arrangement
    
    def _create_drum_pattern(self, melody_length: int, tempo: int) -> List[Dict[str, Any]]:
        """Create a basic drum pattern."""
        
        # Basic 4/4 drum pattern
        # Kick on 1 and 3, snare on 2 and 4, hi-hat on all beats
        
        pattern_length = max(16, (melody_length // 4) * 4)  # Ensure multiple of 4
        drum_events = []
        
        for beat in range(pattern_length):
            beat_in_measure = beat % 4
            
            # Kick drum (MIDI note 36)
            if beat_in_measure in [0, 2]:
                drum_events.append({
                    'time': beat,
                    'note': 36,  # Kick drum
                    'velocity': 100,
                    'duration': 0.25
                })
            
            # Snare drum (MIDI note 38)
            if beat_in_measure in [1, 3]:
                drum_events.append({
                    'time': beat,
                    'note': 38,  # Snare drum
                    'velocity': 90,
                    'duration': 0.25
                })
            
            # Hi-hat (MIDI note 42) - every beat
            drum_events.append({
                'time': beat,
                'note': 42,  # Closed hi-hat
                'velocity': 60,
                'duration': 0.125
            })
        
        return drum_events