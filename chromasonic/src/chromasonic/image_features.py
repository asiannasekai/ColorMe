"""
Image Features Module
Extracts comprehensive features from images to predict musical parameters.
Implements Model A from the system description.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import joblib

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ImageFeatureExtractor:
    """Extracts features from images for musical parameter prediction."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
        # Feature extractors
        self.color_features = ColorFeatures()
        self.texture_features = TextureFeatures()
        self.composition_features = CompositionFeatures()
        
        if HAS_TORCH:
            self.cnn_features = CNNFeatures()
        else:
            self.cnn_features = None
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract all available features from an image.
        
        Args:
            image: Input image array (RGB format)
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Color-based features
        features.update(self.color_features.extract(image))
        
        # Texture features
        features.update(self.texture_features.extract(image))
        
        # Composition features
        features.update(self.composition_features.extract(image))
        
        # CNN features (if available)
        if self.cnn_features:
            features.update(self.cnn_features.extract(image))
        
        return features
    
    def features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numerical vector."""
        
        feature_vector = []
        
        # Color features
        feature_vector.extend([
            features['mean_hue'],
            features['mean_saturation'],
            features['mean_value'],
            features['hue_std'],
            features['saturation_std'],
            features['value_std'],
            features['dominant_hue'],
            features['color_temperature'],
            features['color_harmony_score']
        ])
        
        # Texture features
        feature_vector.extend([
            features['contrast'],
            features['dissimilarity'],
            features['homogeneity'],
            features['energy'],
            features['correlation'],
            features['edge_density']
        ])
        
        # Composition features
        feature_vector.extend([
            features['brightness'],
            features['dynamic_range'],
            features['symmetry_score'],
            features['complexity_score']
        ])
        
        # CNN features (if available)
        if 'cnn_embedding' in features:
            feature_vector.extend(features['cnn_embedding'])
        
        return np.array(feature_vector)


class ColorFeatures:
    """Extracts color-based features from images."""
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color features."""
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Basic HSV statistics
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        features = {
            'mean_hue': np.mean(h) / 179.0,  # Normalize to 0-1
            'mean_saturation': np.mean(s) / 255.0,
            'mean_value': np.mean(v) / 255.0,
            'hue_std': np.std(h) / 179.0,
            'saturation_std': np.std(s) / 255.0,
            'value_std': np.std(v) / 255.0
        }
        
        # Dominant hue
        hue_hist = np.histogram(h, bins=18, range=(0, 180))[0]
        features['dominant_hue'] = np.argmax(hue_hist) / 18.0
        
        # Color temperature (simplified)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        color_temp = np.mean(b) / (np.mean(r) + 1e-6)  # Blue/red ratio
        features['color_temperature'] = min(color_temp, 2.0) / 2.0
        
        # Color harmony score (variance in hue)
        features['color_harmony_score'] = 1.0 - (np.std(h) / 179.0)
        
        return features


class TextureFeatures:
    """Extracts texture features using GLCM and edge detection."""
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features."""
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # GLCM features (simplified implementation)
        glcm_features = self._compute_glcm_features(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        features = {
            'contrast': glcm_features['contrast'],
            'dissimilarity': glcm_features['dissimilarity'],
            'homogeneity': glcm_features['homogeneity'],
            'energy': glcm_features['energy'],
            'correlation': glcm_features['correlation'],
            'edge_density': edge_density
        }
        
        return features
    
    def _compute_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute simplified GLCM features."""
        
        # Reduce image size for faster computation
        small_gray = cv2.resize(gray, (64, 64))
        
        # Compute co-occurrence matrix for horizontal direction
        levels = 32  # Reduce gray levels for efficiency
        normalized = (small_gray / 255.0 * (levels - 1)).astype(np.uint8)
        
        # Simple co-occurrence matrix (horizontal, distance=1)
        glcm = np.zeros((levels, levels))
        
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1] - 1):
                glcm[normalized[i, j], normalized[i, j + 1]] += 1
        
        # Normalize
        glcm = glcm / np.sum(glcm)
        
        # Compute features
        i, j = np.meshgrid(range(levels), range(levels), indexing='ij')
        
        contrast = np.sum(glcm * (i - j) ** 2)
        dissimilarity = np.sum(glcm * np.abs(i - j))
        homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
        energy = np.sum(glcm ** 2)
        
        # Correlation
        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)
        sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * glcm))
        sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * glcm))
        
        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j)
        else:
            correlation = 0
        
        return {
            'contrast': min(contrast / 100.0, 1.0),  # Normalize
            'dissimilarity': min(dissimilarity / 20.0, 1.0),
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': max(0, correlation)
        }


class CompositionFeatures:
    """Extracts composition and structure features."""
    
    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """Extract composition features."""
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Brightness
        brightness = np.mean(gray) / 255.0
        
        # Dynamic range
        dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
        
        # Symmetry score (vertical symmetry)
        left_half = gray[:, :gray.shape[1]//2]
        right_half = np.fliplr(gray[:, gray.shape[1]//2:])
        
        # Resize to same size if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_score = 1.0 - (np.mean(np.abs(left_half - right_half)) / 255.0)
        
        # Complexity score (based on gradients)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        complexity_score = np.mean(gradient_magnitude) / 255.0
        
        features = {
            'brightness': brightness,
            'dynamic_range': dynamic_range,
            'symmetry_score': max(0, symmetry_score),
            'complexity_score': min(complexity_score, 1.0)
        }
        
        return features


if HAS_TORCH:
    class CNNFeatures:
        """Extracts deep features using pre-trained CNN."""
        
        def __init__(self, model_name: str = 'resnet18'):
            """Initialize CNN feature extractor."""
            
            # Load pre-trained model
            if model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
                self.model.fc = nn.Identity()  # Remove final classification layer
                self.feature_dim = 512
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            self.model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def extract(self, image: np.ndarray) -> Dict[str, List[float]]:
            """Extract CNN features."""
            
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.squeeze().numpy()
            
            # Reduce dimensionality (take first 64 features)
            reduced_features = features[:64].tolist()
            
            return {'cnn_embedding': reduced_features}
else:
    class CNNFeatures:
        """Placeholder CNN feature extractor when PyTorch is not available."""
        
        def extract(self, image: np.ndarray) -> Dict[str, List[float]]:
            """Return empty features when PyTorch is not available."""
            return {}


class MusicalParameterPredictor:
    """Predicts musical parameters from image features (Model A)."""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the parameter predictor.
        
        Args:
            model_type: Type of ML model ("linear", "random_forest", "neural_network")
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Initialize models for each parameter
        self.models = {}
        self.scalers = {}
        self.trained = False
        
        # Parameters to predict
        self.parameters = ['tempo', 'key_root', 'major', 'energy', 'valence']
        
        for param in self.parameters:
            if model_type == "random_forest":
                self.models[param] = RandomForestRegressor(n_estimators=50, random_state=42)
            elif model_type == "linear":
                self.models[param] = LinearRegression()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.scalers[param] = StandardScaler()
    
    def train(self, images: List[np.ndarray], parameters: List[Dict[str, float]]):
        """
        Train the models on image-parameter pairs.
        
        Args:
            images: List of training images
            parameters: List of corresponding musical parameters
        """
        
        feature_extractor = ImageFeatureExtractor()
        
        # Extract features from all images
        all_features = []
        for image in images:
            features = feature_extractor.extract_all_features(image)
            feature_vector = feature_extractor.features_to_vector(features)
            all_features.append(feature_vector)
        
        X = np.array(all_features)
        
        # Train models for each parameter
        for param in self.parameters:
            y = np.array([p[param] for p in parameters])
            
            # Scale features
            X_scaled = self.scalers[param].fit_transform(X)
            
            # Train model
            self.models[param].fit(X_scaled, y)
        
        self.trained = True
        self.logger.info(f"Trained models for {len(self.parameters)} parameters on {len(images)} images")
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict musical parameters from an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with predicted parameters
        """
        
        if not self.trained:
            # Return default values if not trained
            return self._get_default_parameters()
        
        # Extract features
        feature_extractor = ImageFeatureExtractor()
        features = feature_extractor.extract_all_features(image)
        feature_vector = feature_extractor.features_to_vector(features)
        
        # Predict parameters
        predictions = {}
        for param in self.parameters:
            X_scaled = self.scalers[param].transform([feature_vector])
            prediction = self.models[param].predict(X_scaled)[0]
            predictions[param] = float(prediction)
        
        # Post-process predictions
        predictions = self._postprocess_predictions(predictions)
        
        return predictions
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default musical parameters when no model is trained."""
        
        return {
            'tempo': 120.0,
            'key_root': 0.0,  # C major
            'major': 1.0,     # Major scale
            'energy': 0.5,
            'valence': 0.5
        }
    
    def _postprocess_predictions(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Postprocess predictions to ensure valid ranges."""
        
        # Clamp values to valid ranges
        predictions['tempo'] = max(60.0, min(180.0, predictions['tempo']))
        predictions['key_root'] = max(0.0, min(11.0, predictions['key_root']))
        predictions['major'] = max(0.0, min(1.0, predictions['major']))
        predictions['energy'] = max(0.0, min(1.0, predictions['energy']))
        predictions['valence'] = max(0.0, min(1.0, predictions['valence']))
        
        return predictions
    
    def save_models(self, save_path: str):
        """Save trained models to disk."""
        
        if not self.trained:
            raise ValueError("No trained models to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model and scaler
        for param in self.parameters:
            model_file = save_path / f"{param}_model.pkl"
            scaler_file = save_path / f"{param}_scaler.pkl"
            
            joblib.dump(self.models[param], model_file)
            joblib.dump(self.scalers[param], scaler_file)
        
        self.logger.info(f"Saved models to {save_path}")
    
    def load_models(self, load_path: str):
        """Load trained models from disk."""
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise ValueError(f"Model path does not exist: {load_path}")
        
        # Load each model and scaler
        for param in self.parameters:
            model_file = load_path / f"{param}_model.pkl"
            scaler_file = load_path / f"{param}_scaler.pkl"
            
            if model_file.exists() and scaler_file.exists():
                self.models[param] = joblib.load(model_file)
                self.scalers[param] = joblib.load(scaler_file)
            else:
                self.logger.warning(f"Model files not found for parameter: {param}")
        
        self.trained = True
        self.logger.info(f"Loaded models from {load_path}")