"""
Color Analysis Module
Extracts dominant colors from images using various algorithms.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils import shuffle
from collections import Counter
import colorsys


class ColorExtractor:
    """Extracts dominant colors from images using multiple algorithms."""
    
    def __init__(self, sample_size: int = 10000):
        """
        Initialize ColorExtractor.
        
        Args:
            sample_size: Number of pixels to sample for large images
        """
        self.sample_size = sample_size
        self.logger = logging.getLogger(__name__)
    
    def extract_colors(
        self,
        image: np.ndarray,
        num_colors: int = 8,
        method: str = "kmeans",
        **kwargs
    ) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from an image.
        
        Args:
            image: Input image array (RGB format)
            num_colors: Number of colors to extract
            method: Extraction method ("kmeans", "quantization", "histogram")
            
        Returns:
            List of RGB tuples representing dominant colors
        """
        if method == "kmeans":
            return self._extract_kmeans(image, num_colors, **kwargs)
        elif method == "quantization":
            return self._extract_quantization(image, num_colors, **kwargs)
        elif method == "histogram":
            return self._extract_histogram(image, num_colors, **kwargs)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def _extract_kmeans(
        self,
        image: np.ndarray,
        num_colors: int,
        use_mini_batch: bool = True,
        **kwargs
    ) -> List[Tuple[int, int, int]]:
        """Extract colors using K-means clustering."""
        
        # Reshape image to pixel array
        pixels = image.reshape(-1, 3)
        
        # Sample pixels for large images
        if len(pixels) > self.sample_size:
            pixels = shuffle(pixels, n_samples=self.sample_size, random_state=42)
        
        # Apply K-means clustering
        if use_mini_batch and len(pixels) > 1000:
            kmeans = MiniBatchKMeans(
                n_clusters=num_colors,
                random_state=42,
                batch_size=1000
            )
        else:
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        
        kmeans.fit(pixels)
        
        # Get cluster centers as colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort by cluster size (most prominent first)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        sorted_colors = []
        
        for label, _ in label_counts.most_common():
            color = tuple(colors[label])
            sorted_colors.append(color)
        
        self.logger.info(f"Extracted {len(sorted_colors)} colors using K-means")
        return sorted_colors
    
    def _extract_quantization(
        self,
        image: np.ndarray,
        num_colors: int,
        **kwargs
    ) -> List[Tuple[int, int, int]]:
        """Extract colors using color quantization."""
        
        # Reshape and sample
        pixels = image.reshape(-1, 3).astype(np.float32)
        if len(pixels) > self.sample_size:
            pixels = shuffle(pixels, n_samples=self.sample_size, random_state=42)
        
        # Quantize colors by reducing bit depth
        quantized = (pixels // (256 // num_colors)) * (256 // num_colors)
        
        # Count unique colors
        unique_colors, counts = np.unique(
            quantized.astype(int), 
            axis=0, 
            return_counts=True
        )
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        colors = unique_colors[sorted_indices[:num_colors]]
        
        color_list = [tuple(color) for color in colors]
        
        self.logger.info(f"Extracted {len(color_list)} colors using quantization")
        return color_list
    
    def _extract_histogram(
        self,
        image: np.ndarray,
        num_colors: int,
        bins_per_channel: int = 8,
        **kwargs
    ) -> List[Tuple[int, int, int]]:
        """Extract colors using 3D color histogram."""
        
        # Create 3D histogram
        hist, edges = np.histogramdd(
            image.reshape(-1, 3),
            bins=bins_per_channel,
            range=[(0, 256), (0, 256), (0, 256)]
        )
        
        # Find peaks in histogram
        flat_hist = hist.flatten()
        peak_indices = np.argsort(flat_hist)[-num_colors:][::-1]
        
        # Convert indices back to RGB coordinates
        colors = []
        for idx in peak_indices:
            # Convert flat index to 3D coordinates
            coords = np.unravel_index(idx, hist.shape)
            
            # Convert bin coordinates to RGB values
            rgb = []
            for i, coord in enumerate(coords):
                bin_center = (edges[i][coord] + edges[i][coord + 1]) / 2
                rgb.append(int(bin_center))
            
            colors.append(tuple(rgb))
        
        self.logger.info(f"Extracted {len(colors)} colors using histogram")
        return colors
    
    def get_color_harmony(self, colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """
        Analyze color harmony and relationships.
        
        Args:
            colors: List of RGB color tuples
            
        Returns:
            Dictionary with harmony analysis
        """
        if not colors:
            return {}
        
        # Convert to HSV for better analysis
        hsv_colors = []
        for r, g, b in colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hsv_colors.append((h * 360, s * 100, v * 100))
        
        # Calculate color temperature
        temperatures = []
        for r, g, b in colors:
            # Simplified color temperature calculation
            if b > r:
                temp = 'cool'
            elif r > b:
                temp = 'warm'
            else:
                temp = 'neutral'
            temperatures.append(temp)
        
        # Dominant temperature
        temp_counter = Counter(temperatures)
        dominant_temp = temp_counter.most_common(1)[0][0]
        
        # Calculate average saturation and brightness
        saturations = [hsv[1] for hsv in hsv_colors]
        brightnesses = [hsv[2] for hsv in hsv_colors]
        
        return {
            'dominant_temperature': dominant_temp,
            'temperature_distribution': dict(temp_counter),
            'average_saturation': np.mean(saturations),
            'average_brightness': np.mean(brightnesses),
            'saturation_range': (min(saturations), max(saturations)),
            'brightness_range': (min(brightnesses), max(brightnesses)),
            'hsv_colors': hsv_colors
        }
    
    def get_color_weights(
        self,
        image: np.ndarray,
        colors: List[Tuple[int, int, int]],
        tolerance: int = 30
    ) -> List[float]:
        """
        Calculate the relative weight/prominence of each color in the image.
        
        Args:
            image: Original image array
            colors: List of extracted colors
            tolerance: Color matching tolerance
            
        Returns:
            List of weights (0-1) for each color
        """
        pixels = image.reshape(-1, 3)
        weights = []
        
        for color in colors:
            # Count pixels similar to this color
            distances = np.sqrt(np.sum((pixels - np.array(color)) ** 2, axis=1))
            matching_pixels = np.sum(distances <= tolerance)
            weight = matching_pixels / len(pixels)
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights