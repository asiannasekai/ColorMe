"""
Wavelength Mapping Module
Converts RGB colors to wavelengths and then to musical frequencies.
"""

import logging
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class WavelengthConverter:
    """Converts colors to wavelengths and wavelengths to musical frequencies."""
    
    def __init__(self):
        """Initialize the wavelength converter with scientific constants."""
        self.logger = logging.getLogger(__name__)
        
        # Visible light spectrum (nanometers)
        self.min_wavelength = 380  # Violet
        self.max_wavelength = 750  # Red
        
        # Musical constants
        self.a4_frequency = 440.0  # A4 note frequency in Hz
        self.a4_midi_note = 69     # MIDI note number for A4
        
        # Speed of light (m/s)
        self.speed_of_light = 299792458
        
        # Wavelength to RGB mapping for validation
        self.wavelength_rgb_map = self._create_wavelength_rgb_map()
    
    def rgb_to_wavelengths(
        self,
        colors: List[Tuple[int, int, int]],
        method: str = "dominant_component"
    ) -> List[float]:
        """
        Convert RGB colors to approximate wavelengths.
        
        Args:
            colors: List of RGB color tuples
            method: Conversion method ("dominant_component", "weighted_average", "hue_based")
            
        Returns:
            List of wavelengths in nanometers
        """
        wavelengths = []
        
        for r, g, b in colors:
            if method == "dominant_component":
                wavelength = self._rgb_to_wavelength_dominant(r, g, b)
            elif method == "weighted_average":
                wavelength = self._rgb_to_wavelength_weighted(r, g, b)
            elif method == "hue_based":
                wavelength = self._rgb_to_wavelength_hue(r, g, b)
            else:
                raise ValueError(f"Unknown conversion method: {method}")
            
            wavelengths.append(wavelength)
        
        self.logger.info(f"Converted {len(colors)} colors to wavelengths")
        return wavelengths
    
    def _rgb_to_wavelength_dominant(self, r: int, g: int, b: int) -> float:
        """Convert RGB to wavelength based on dominant color component."""
        
        # Normalize RGB values
        total = r + g + b
        if total == 0:
            return (self.min_wavelength + self.max_wavelength) / 2
        
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
        
        # Map dominant component to wavelength ranges
        if r_norm > g_norm and r_norm > b_norm:
            # Red dominant: 620-750 nm
            wavelength = 620 + (750 - 620) * (r / 255)
        elif g_norm > r_norm and g_norm > b_norm:
            # Green dominant: 495-570 nm
            wavelength = 495 + (570 - 495) * (g / 255)
        else:
            # Blue dominant: 380-495 nm
            wavelength = 380 + (495 - 380) * (b / 255)
        
        return wavelength
    
    def _rgb_to_wavelength_weighted(self, r: int, g: int, b: int) -> float:
        """Convert RGB to wavelength using weighted average of spectrum."""
        
        # Define wavelength ranges for each color component
        red_range = (620, 750)    # Red wavelengths
        green_range = (495, 570)  # Green wavelengths
        blue_range = (380, 495)   # Blue wavelengths
        
        # Normalize RGB
        total = max(r + g + b, 1)
        r_weight = r / total
        g_weight = g / total
        b_weight = b / total
        
        # Calculate weighted wavelength
        wavelength = (
            r_weight * np.mean(red_range) +
            g_weight * np.mean(green_range) +
            b_weight * np.mean(blue_range)
        )
        
        return wavelength
    
    def _rgb_to_wavelength_hue(self, r: int, g: int, b: int) -> float:
        """Convert RGB to wavelength based on HSV hue."""
        
        # Convert RGB to HSV
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        diff = max_val - min_val
        
        if diff == 0:
            hue = 0
        elif max_val == r_norm:
            hue = (60 * ((g_norm - b_norm) / diff) + 360) % 360
        elif max_val == g_norm:
            hue = (60 * ((b_norm - r_norm) / diff) + 120) % 360
        else:
            hue = (60 * ((r_norm - g_norm) / diff) + 240) % 360
        
        # Map hue (0-360°) to wavelength (380-750 nm)
        # Red: 0°, Yellow: 60°, Green: 120°, Cyan: 180°, Blue: 240°, Magenta: 300°
        wavelength = self._hue_to_wavelength(hue)
        
        return wavelength
    
    def _hue_to_wavelength(self, hue: float) -> float:
        """Map HSV hue to wavelength."""
        
        # Hue to wavelength mapping
        if 0 <= hue <= 60:  # Red to Yellow
            return 700 - (hue / 60) * (700 - 580)
        elif 60 < hue <= 120:  # Yellow to Green
            return 580 - ((hue - 60) / 60) * (580 - 520)
        elif 120 < hue <= 180:  # Green to Cyan
            return 520 - ((hue - 120) / 60) * (520 - 490)
        elif 180 < hue <= 240:  # Cyan to Blue
            return 490 - ((hue - 180) / 60) * (490 - 450)
        elif 240 < hue <= 300:  # Blue to Magenta
            return 450 - ((hue - 240) / 60) * (450 - 400)
        else:  # Magenta to Red
            return 400 + ((hue - 300) / 60) * (700 - 400)
    
    def wavelengths_to_frequencies(
        self,
        wavelengths: List[float],
        octave_range: Tuple[int, int] = (3, 6),
        scale_factor: float = 1e12
    ) -> List[float]:
        """
        Convert light wavelengths to musical frequencies.
        
        Args:
            wavelengths: List of wavelengths in nanometers
            octave_range: Musical octave range to map frequencies to
            scale_factor: Factor to scale light frequencies to audio range
            
        Returns:
            List of musical frequencies in Hz
        """
        musical_frequencies = []
        
        for wavelength in wavelengths:
            # Convert wavelength to light frequency
            # f = c / λ (where λ is in meters)
            light_freq = self.speed_of_light / (wavelength * 1e-9)
            
            # Scale down to audio frequency range
            # This is a creative mapping, not physically accurate
            audio_freq = light_freq / scale_factor
            
            # Map to musical octave range
            min_freq = self._midi_to_frequency(octave_range[0] * 12)  # C of min octave
            max_freq = self._midi_to_frequency(octave_range[1] * 12)  # C of max octave
            
            # Normalize to frequency range
            normalized_freq = np.interp(
                audio_freq,
                [min(wavelengths), max(wavelengths)],
                [min_freq, max_freq]
            )
            
            musical_frequencies.append(normalized_freq)
        
        self.logger.info(f"Converted {len(wavelengths)} wavelengths to frequencies")
        return musical_frequencies
    
    def frequencies_to_midi_notes(self, frequencies: List[float]) -> List[int]:
        """Convert frequencies to MIDI note numbers."""
        midi_notes = []
        
        for freq in frequencies:
            # Calculate MIDI note number from frequency
            # MIDI note = 69 + 12 * log2(f/440)
            if freq > 0:
                midi_note = int(69 + 12 * math.log2(freq / self.a4_frequency))
                midi_note = max(0, min(127, midi_note))  # Clamp to MIDI range
            else:
                midi_note = 69  # Default to A4
            
            midi_notes.append(midi_note)
        
        return midi_notes
    
    def _midi_to_frequency(self, midi_note: int) -> float:
        """Convert MIDI note number to frequency."""
        return self.a4_frequency * (2 ** ((midi_note - self.a4_midi_note) / 12))
    
    def get_color_spectrum_info(self, wavelength: float) -> Dict[str, Any]:
        """Get information about a wavelength's position in the visible spectrum."""
        
        if wavelength < self.min_wavelength or wavelength > self.max_wavelength:
            return {
                'visible': False,
                'color_name': 'infrared' if wavelength > self.max_wavelength else 'ultraviolet',
                'spectrum_position': -1
            }
        
        # Determine color name based on wavelength
        if 380 <= wavelength <= 450:
            color_name = 'violet'
        elif 450 < wavelength <= 495:
            color_name = 'blue'
        elif 495 < wavelength <= 570:
            color_name = 'green'
        elif 570 < wavelength <= 590:
            color_name = 'yellow'
        elif 590 < wavelength <= 620:
            color_name = 'orange'
        else:  # 620 < wavelength <= 750
            color_name = 'red'
        
        # Calculate position in spectrum (0-1)
        spectrum_position = (wavelength - self.min_wavelength) / (
            self.max_wavelength - self.min_wavelength
        )
        
        return {
            'visible': True,
            'color_name': color_name,
            'spectrum_position': spectrum_position,
            'wavelength': wavelength
        }
    
    def _create_wavelength_rgb_map(self) -> Dict[int, Tuple[int, int, int]]:
        """Create a mapping of wavelengths to approximate RGB values."""
        wavelength_map = {}
        
        for wl in range(380, 781, 5):
            rgb = self._wavelength_to_rgb(wl)
            wavelength_map[wl] = rgb
        
        return wavelength_map
    
    def _wavelength_to_rgb(self, wavelength: float) -> Tuple[int, int, int]:
        """
        Convert wavelength to RGB (approximate).
        Based on Dan Bruton's algorithm.
        """
        if wavelength < 380 or wavelength > 750:
            return (0, 0, 0)
        
        if 380 <= wavelength <= 440:
            red = -(wavelength - 440) / (440 - 380)
            green = 0.0
            blue = 1.0
        elif 440 <= wavelength <= 490:
            red = 0.0
            green = (wavelength - 440) / (490 - 440)
            blue = 1.0
        elif 490 <= wavelength <= 510:
            red = 0.0
            green = 1.0
            blue = -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength <= 580:
            red = (wavelength - 510) / (580 - 510)
            green = 1.0
            blue = 0.0
        elif 580 <= wavelength <= 645:
            red = 1.0
            green = -(wavelength - 645) / (645 - 580)
            blue = 0.0
        elif 645 <= wavelength <= 750:
            red = 1.0
            green = 0.0
            blue = 0.0
        
        # Intensity correction near the vision limits
        if 380 <= wavelength <= 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif 420 <= wavelength <= 700:
            factor = 1.0
        elif 700 <= wavelength <= 750:
            factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
        else:
            factor = 0.0
        
        # Convert to 8-bit RGB
        r = int(255 * red * factor) if red > 0 else 0
        g = int(255 * green * factor) if green > 0 else 0
        b = int(255 * blue * factor) if blue > 0 else 0
        
        return (r, g, b)