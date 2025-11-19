"""
Image Processing Module
Handles loading, preprocessing, and basic analysis of images.
"""

import logging
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


class ImageLoader:
    """Handles image loading and preprocessing for color analysis."""
    
    def __init__(self, max_size: Tuple[int, int] = (800, 600)):
        """
        Initialize ImageLoader.
        
        Args:
            max_size: Maximum dimensions to resize images to for processing
        """
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
    
    def load(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess an image for color analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array (RGB format)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image with PIL for better format support
        try:
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize if too large
            if (pil_image.size[0] > self.max_size[0] or 
                pil_image.size[1] > self.max_size[1]):
                pil_image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            self.logger.info(
                f"Loaded image: {image_path.name}, "
                f"shape: {image.shape}, "
                f"dtype: {image.dtype}"
            )
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def preprocess(
        self, 
        image: np.ndarray,
        enhance_colors: bool = True,
        blur_kernel: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply preprocessing to enhance color extraction.
        
        Args:
            image: Input image array
            enhance_colors: Whether to enhance color saturation
            blur_kernel: Size of gaussian blur kernel (None for no blur)
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Apply gaussian blur to reduce noise
        if blur_kernel and blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        # Enhance color saturation in HSV space
        if enhance_colors:
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Boost saturation
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return processed
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get statistical information about the image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary with image statistics
        """
        return {
            'shape': image.shape,
            'mean_rgb': np.mean(image, axis=(0, 1)),
            'std_rgb': np.std(image, axis=(0, 1)),
            'brightness': np.mean(image),
            'contrast': np.std(image),
            'pixel_count': image.shape[0] * image.shape[1]
        }
    
    def create_thumbnail(
        self, 
        image: np.ndarray, 
        size: Tuple[int, int] = (150, 150)
    ) -> np.ndarray:
        """Create a thumbnail version of the image."""
        pil_image = Image.fromarray(image)
        pil_image.thumbnail(size, Image.Resampling.LANCZOS)
        return np.array(pil_image)
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate that the image is in the correct format.
        
        Args:
            image: Image array to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) != 3:
            return False
        
        if image.shape[2] != 3:
            return False
        
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            return False
        
        return True