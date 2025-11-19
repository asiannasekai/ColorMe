"""
Test suite for Chromasonic pipeline components.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from chromasonic.color_analysis.extractor import ColorExtractor
from chromasonic.wavelength_mapping.converter import WavelengthConverter


class TestColorExtractor(unittest.TestCase):
    """Test color extraction functionality."""
    
    def setUp(self):
        self.extractor = ColorExtractor()
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some colors
        self.test_image[:50, :50] = [255, 0, 0]  # Red
        self.test_image[:50, 50:] = [0, 255, 0]  # Green
        self.test_image[50:, :50] = [0, 0, 255]  # Blue
        self.test_image[50:, 50:] = [255, 255, 0]  # Yellow
    
    def test_extract_colors_kmeans(self):
        """Test K-means color extraction."""
        colors = self.extractor.extract_colors(self.test_image, num_colors=4, method='kmeans')
        
        self.assertEqual(len(colors), 4)
        
        # Check that colors are RGB tuples
        for color in colors:
            self.assertIsInstance(color, tuple)
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, (int, np.integer))
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)
    
    def test_extract_colors_quantization(self):
        """Test quantization color extraction."""
        colors = self.extractor.extract_colors(self.test_image, num_colors=4, method='quantization')
        
        self.assertEqual(len(colors), 4)
        
        # Verify colors are valid RGB
        for color in colors:
            self.assertIsInstance(color, tuple)
            self.assertEqual(len(color), 3)
    
    def test_color_harmony_analysis(self):
        """Test color harmony analysis."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        harmony = self.extractor.get_color_harmony(colors)
        
        self.assertIsInstance(harmony, dict)
        self.assertIn('dominant_temperature', harmony)
        self.assertIn('average_saturation', harmony)
        self.assertIn('average_brightness', harmony)


class TestWavelengthConverter(unittest.TestCase):
    """Test wavelength conversion functionality."""
    
    def setUp(self):
        self.converter = WavelengthConverter()
    
    def test_rgb_to_wavelength_hue_based(self):
        """Test RGB to wavelength conversion."""
        # Test primary colors
        test_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
        ]
        
        wavelengths = self.converter.rgb_to_wavelengths(test_colors)
        
        self.assertEqual(len(wavelengths), len(test_colors))
        
        # Check wavelengths are in visible range
        for wl in wavelengths:
            self.assertGreaterEqual(wl, self.converter.min_wavelength)
            self.assertLessEqual(wl, self.converter.max_wavelength)
    
    def test_wavelengths_to_frequencies(self):
        """Test wavelength to frequency conversion."""
        wavelengths = [400, 500, 600, 700]  # Sample wavelengths
        frequencies = self.converter.wavelengths_to_frequencies(wavelengths)
        
        self.assertEqual(len(frequencies), len(wavelengths))
        
        # Check frequencies are positive
        for freq in frequencies:
            self.assertGreater(freq, 0)
    
    def test_spectrum_info(self):
        """Test spectrum information retrieval."""
        # Test visible wavelength
        info = self.converter.get_color_spectrum_info(500)
        self.assertTrue(info['visible'])
        self.assertEqual(info['color_name'], 'green')
        
        # Test invisible wavelength
        info = self.converter.get_color_spectrum_info(300)
        self.assertFalse(info['visible'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.extractor = ColorExtractor()
        self.converter = WavelengthConverter()
    
    def test_full_pipeline(self):
        """Test the complete image-to-wavelength pipeline."""
        # Create test image
        test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        
        # Extract colors
        colors = self.extractor.extract_colors(test_image, num_colors=5)
        
        # Convert to wavelengths
        wavelengths = self.converter.rgb_to_wavelengths(colors)
        
        # Convert to frequencies
        frequencies = self.converter.wavelengths_to_frequencies(wavelengths)
        
        # Verify pipeline integrity
        self.assertEqual(len(colors), len(wavelengths))
        self.assertEqual(len(wavelengths), len(frequencies))
        
        # Check all values are reasonable
        for wl in wavelengths:
            self.assertGreater(wl, 300)
            self.assertLess(wl, 800)
        
        for freq in frequencies:
            self.assertGreater(freq, 50)
            self.assertLess(freq, 5000)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestColorExtractor))
    suite.addTest(unittest.makeSuite(TestWavelengthConverter))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
    print(f"{'='*50}")