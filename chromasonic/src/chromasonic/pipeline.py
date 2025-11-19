"""
Chromasonic: Main Pipeline
Orchestrates the image-to-music conversion process.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from .image_processing.loader import ImageLoader
from .color_analysis.extractor import ColorExtractor
from .wavelength_mapping.converter import WavelengthConverter
from .melody_generation.models import MelodyGenerator
from .audio_synthesis.synthesizer import AudioSynthesizer


class ChromasonicPipeline:
    """Main pipeline for converting images to music."""
    
    def __init__(
        self,
        model_type: str = "markov",
        scale: str = "major",
        tempo: int = 120,
        duration: float = 30.0,
        **kwargs
    ):
        """
        Initialize the Chromasonic pipeline.
        
        Args:
            model_type: ML model type ("markov", "lstm", "transformer")
            scale: Musical scale ("major", "minor", "pentatonic", "blues", "chromatic")
            tempo: BPM for generated music
            duration: Duration of generated music in seconds
        """
        self.model_type = model_type
        self.scale = scale
        self.tempo = tempo
        self.duration = duration
        
        # Initialize components
        self.image_loader = ImageLoader()
        self.color_extractor = ColorExtractor()
        self.wavelength_converter = WavelengthConverter()
        self.melody_generator = MelodyGenerator(model_type=model_type)
        self.audio_synthesizer = AudioSynthesizer(scale=scale, tempo=tempo)
        
        self.logger = logging.getLogger(__name__)
    
    def process_image(
        self, 
        image_path: str, 
        num_colors: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert an image to music through the complete pipeline.
        
        Args:
            image_path: Path to input image
            num_colors: Number of dominant colors to extract
            
        Returns:
            Dictionary containing:
            - colors: Extracted RGB colors
            - wavelengths: Corresponding wavelengths
            - frequencies: Musical frequencies
            - melody: Generated melody sequence
            - audio: Synthesized audio data
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Step 1: Load and preprocess image
        image = self.image_loader.load(image_path)
        
        # Step 2: Extract dominant colors
        colors = self.color_extractor.extract_colors(
            image, 
            num_colors=num_colors,
            method=kwargs.get('color_method', 'kmeans')
        )
        
        # Step 3: Convert colors to wavelengths
        wavelengths = self.wavelength_converter.rgb_to_wavelengths(colors)
        
        # Step 4: Map wavelengths to musical frequencies
        frequencies = self.wavelength_converter.wavelengths_to_frequencies(
            wavelengths,
            octave_range=(3, 6)  # Musical octave range
        )
        
        # Step 5: Generate melody using ML model
        melody = self.melody_generator.generate_melody(
            frequencies=frequencies,
            duration=self.duration,
            scale=self.scale,
            tempo=self.tempo,
            **kwargs
        )
        
        # Step 6: Synthesize audio
        audio = self.audio_synthesizer.synthesize(melody)
        
        return {
            'colors': colors,
            'wavelengths': wavelengths,
            'frequencies': frequencies,
            'melody': melody,
            'audio': audio,
            'metadata': {
                'model_type': self.model_type,
                'scale': self.scale,
                'tempo': self.tempo,
                'duration': self.duration,
                'num_colors': num_colors
            }
        }
    
    def save_audio(self, audio_data: np.ndarray, output_path: str, sample_rate: int = 44100):
        """Save audio data to file."""
        self.audio_synthesizer.save_audio(audio_data, output_path, format="wav")
        self.logger.info(f"Audio saved to: {output_path}")
    
    def batch_process(
        self, 
        image_paths: List[str], 
        output_dir: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple images and save results."""
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.process_image(image_path, **kwargs)
                
                # Save audio
                audio_file = output_path / f"chromasonic_{i:03d}.wav"
                self.save_audio(result['audio'], str(audio_file))
                result['audio_file'] = str(audio_file)
                
                results.append(result)
                self.logger.info(f"Processed {i+1}/{len(image_paths)}: {image_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                continue
        
        return results
    
    def update_scale(self, new_scale: str):
        """Update the musical scale for future generations."""
        self.scale = new_scale
        self.audio_synthesizer.scale = new_scale
    
    def update_tempo(self, new_tempo: int):
        """Update the tempo for future generations."""
        self.tempo = new_tempo
        self.audio_synthesizer.tempo = new_tempo