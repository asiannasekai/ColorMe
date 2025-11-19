"""
Audio Synthesis Module
Converts melody data into high-quality audio using various synthesis techniques.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import wave
import struct

# Try to import audio libraries
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from pydub import AudioSegment
    from pydub.generators import Sine, Square, Sawtooth, Triangle
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False


class AudioSynthesizer:
    """Synthesizes audio from melody data."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        scale: str = "major",
        tempo: int = 120,
        synthesis_method: str = "additive"
    ):
        """
        Initialize audio synthesizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            scale: Musical scale for reference
            tempo: Tempo in BPM
            synthesis_method: Synthesis method ("additive", "fm", "subtractive")
        """
        self.sample_rate = sample_rate
        self.scale = scale
        self.tempo = tempo
        self.synthesis_method = synthesis_method
        self.logger = logging.getLogger(__name__)
        
        # Audio parameters
        self.bit_depth = 16
        self.channels = 1  # Mono for now, can be extended to stereo
        
        # Synthesis parameters
        self.envelope_attack = 0.1   # Attack time in seconds
        self.envelope_decay = 0.2    # Decay time in seconds
        self.envelope_sustain = 0.7  # Sustain level (0-1)
        self.envelope_release = 0.3  # Release time in seconds
        
        # Effects parameters
        self.reverb_enabled = True
        self.reverb_wet = 0.2
    
    def synthesize(self, melody_data: Dict[str, Any]) -> np.ndarray:
        """
        Synthesize audio from melody data.
        
        Args:
            melody_data: Dictionary containing melody information
            
        Returns:
            Audio data as numpy array
        """
        frequencies = melody_data['frequencies']
        rhythm = melody_data.get('rhythm', [1.0] * len(frequencies))
        
        # Calculate note durations in seconds
        beat_duration = 60.0 / self.tempo  # Duration of one beat in seconds
        note_durations = [beat_duration * r for r in rhythm]
        
        # Generate audio for each note
        audio_segments = []
        
        for i, (freq, duration) in enumerate(zip(frequencies, note_durations)):
            note_audio = self._synthesize_note(freq, duration)
            audio_segments.append(note_audio)
        
        # Concatenate all notes
        full_audio = np.concatenate(audio_segments)
        
        # Apply effects
        if self.reverb_enabled:
            full_audio = self._apply_reverb(full_audio)
        
        # Normalize audio
        full_audio = self._normalize_audio(full_audio)
        
        self.logger.info(f"Synthesized {len(full_audio)/self.sample_rate:.2f}s of audio")
        return full_audio
    
    def _synthesize_note(self, frequency: float, duration: float) -> np.ndarray:
        """Synthesize a single note."""
        
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        
        if self.synthesis_method == "additive":
            audio = self._additive_synthesis(frequency, t)
        elif self.synthesis_method == "fm":
            audio = self._fm_synthesis(frequency, t)
        elif self.synthesis_method == "subtractive":
            audio = self._subtractive_synthesis(frequency, t)
        else:
            audio = self._simple_sine_wave(frequency, t)
        
        # Apply ADSR envelope
        envelope = self._generate_adsr_envelope(duration, num_samples)
        audio = audio * envelope
        
        return audio
    
    def _simple_sine_wave(self, frequency: float, t: np.ndarray) -> np.ndarray:
        """Generate a simple sine wave."""
        return np.sin(2 * np.pi * frequency * t)
    
    def _additive_synthesis(self, frequency: float, t: np.ndarray) -> np.ndarray:
        """Generate sound using additive synthesis (multiple harmonics)."""
        
        # Fundamental frequency
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add harmonics with decreasing amplitude
        harmonics = [2, 3, 4, 5, 6, 7, 8]
        amplitudes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        
        for harmonic, amplitude in zip(harmonics, amplitudes):
            harmonic_freq = frequency * harmonic
            if harmonic_freq < self.sample_rate / 2:  # Avoid aliasing
                audio += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        
        return audio
    
    def _fm_synthesis(
        self, 
        frequency: float, 
        t: np.ndarray,
        modulation_ratio: float = 2.0,
        modulation_index: float = 5.0
    ) -> np.ndarray:
        """Generate sound using FM synthesis."""
        
        modulator_freq = frequency * modulation_ratio
        modulator = modulation_index * np.sin(2 * np.pi * modulator_freq * t)
        carrier = np.sin(2 * np.pi * frequency * t + modulator)
        
        return carrier
    
    def _subtractive_synthesis(self, frequency: float, t: np.ndarray) -> np.ndarray:
        """Generate sound using subtractive synthesis (filtered sawtooth)."""
        
        # Generate sawtooth wave
        audio = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        
        # Apply simple low-pass filter
        cutoff_freq = frequency * 4  # Cutoff at 4th harmonic
        audio = self._simple_lowpass_filter(audio, cutoff_freq)
        
        return audio
    
    def _simple_lowpass_filter(self, signal: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply a simple low-pass filter."""
        
        # Simple RC low-pass filter approximation
        dt = 1.0 / self.sample_rate
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        alpha = dt / (rc + dt)
        
        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        
        for i in range(1, len(signal)):
            filtered[i] = filtered[i-1] + alpha * (signal[i] - filtered[i-1])
        
        return filtered
    
    def _generate_adsr_envelope(self, duration: float, num_samples: int) -> np.ndarray:
        """Generate ADSR envelope."""
        
        envelope = np.ones(num_samples)
        
        # Calculate sample indices for each phase
        attack_samples = int(self.envelope_attack * self.sample_rate)
        decay_samples = int(self.envelope_decay * self.sample_rate)
        release_samples = int(self.envelope_release * self.sample_rate)
        
        # Ensure phases don't exceed total duration
        attack_samples = min(attack_samples, num_samples // 4)
        decay_samples = min(decay_samples, num_samples // 4)
        release_samples = min(release_samples, num_samples // 2)
        
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples
        
        if sustain_samples < 0:
            # If note is too short, just use attack and release
            attack_samples = num_samples // 2
            release_samples = num_samples - attack_samples
            decay_samples = 0
            sustain_samples = 0
        
        idx = 0
        
        # Attack phase
        if attack_samples > 0:
            envelope[idx:idx+attack_samples] = np.linspace(0, 1, attack_samples)
            idx += attack_samples
        
        # Decay phase
        if decay_samples > 0:
            envelope[idx:idx+decay_samples] = np.linspace(1, self.envelope_sustain, decay_samples)
            idx += decay_samples
        
        # Sustain phase
        if sustain_samples > 0:
            envelope[idx:idx+sustain_samples] = self.envelope_sustain
            idx += sustain_samples
        
        # Release phase
        if release_samples > 0:
            envelope[idx:idx+release_samples] = np.linspace(self.envelope_sustain, 0, release_samples)
        
        return envelope
    
    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple reverb effect."""
        
        # Simple reverb using multiple delayed copies
        delay_times = [0.03, 0.06, 0.09, 0.12]  # Delay times in seconds
        decay_factors = [0.6, 0.4, 0.3, 0.2]     # Decay factors for each delay
        
        reverb_audio = audio.copy()
        
        for delay_time, decay_factor in zip(delay_times, decay_factors):
            delay_samples = int(delay_time * self.sample_rate)
            if delay_samples < len(audio):
                delayed_audio = np.zeros_like(audio)
                delayed_audio[delay_samples:] = audio[:-delay_samples] * decay_factor
                reverb_audio += delayed_audio
        
        # Mix wet and dry signal
        return (1 - self.reverb_wet) * audio + self.reverb_wet * reverb_audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9  # Leave some headroom
        
        return audio
    
    def save_audio(
        self,
        audio_data: np.ndarray,
        filename: str,
        format: str = "wav"
    ):
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data as numpy array
            filename: Output filename
            format: Audio format ("wav", "mp3", "flac")
        """
        filename = Path(filename)
        
        if isinstance(format, str) and (format.lower() == "wav" or filename.suffix.lower() == ".wav"):
            self._save_wav(audio_data, filename)
        elif HAS_LIBROSA and isinstance(format, str) and format.lower() in ["mp3", "flac"]:
            sf.write(filename, audio_data, self.sample_rate)
        else:
            # Fallback to WAV
            self._save_wav(audio_data, filename.with_suffix('.wav'))
        
        self.logger.info(f"Audio saved to: {filename}")
    
    def _save_wav(self, audio_data: np.ndarray, filename: Path):
        """Save audio as WAV file using built-in wave module."""
        
        # Convert to 16-bit integers
        audio_int = (audio_data * (2**(self.bit_depth-1) - 1)).astype(np.int16)
        
        with wave.open(str(filename), 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.bit_depth // 8)
            wav_file.setframerate(self.sample_rate)
            
            # Write audio data
            for sample in audio_int:
                wav_file.writeframesraw(struct.pack('<h', sample))