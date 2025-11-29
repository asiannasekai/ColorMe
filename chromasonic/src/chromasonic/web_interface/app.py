"""
Flask Web Application for Chromasonic.
Provides a user-friendly web interface for image-to-music conversion.
"""

import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, url_for
from flask_cors import CORS
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import tempfile
import uuid

# Import Chromasonic components
from ..pipeline import ChromasonicPipeline


app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global pipeline instance
pipeline = ChromasonicPipeline()


@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze image for colors, wavelengths, and frequencies without generating music."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get processing parameters
        num_colors = int(request.form.get('num_colors', 8))
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{file.filename}")
        file.save(file_path)
        
        # Load and analyze image (without music generation)
        image = pipeline.image_loader.load(file_path)
        
        # Extract colors
        colors = pipeline.color_extractor.extract_colors(image, num_colors=num_colors)
        
        # Convert colors to wavelengths and frequencies
        wavelengths = pipeline.wavelength_converter.rgb_to_wavelengths(colors)
        frequencies = pipeline.wavelength_converter.wavelengths_to_frequencies(
            wavelengths, octave_range=(3, 6)
        )
        
        # Convert numpy types to Python native types for JSON serialization
        colors_list = [[int(r), int(g), int(b)] for r, g, b in colors]
        wavelengths_list = [float(w) for w in wavelengths]
        frequencies_list = [float(f) for f in frequencies]
        
        # Convert colors to hex for display
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'colors': hex_colors,
            'rgb_colors': colors_list,
            'wavelengths': wavelengths_list,
            'frequencies': frequencies_list,
            'num_colors': num_colors
        })
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing (legacy endpoint)."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get processing parameters
        num_colors = int(request.form.get('num_colors', 8))
        scale = request.form.get('scale', 'major')
        tempo = int(request.form.get('tempo', 120))
        duration = float(request.form.get('duration', 30.0))
        model_type = request.form.get('model_type', 'markov')
        
        # Update pipeline parameters
        pipeline.update_scale(scale)
        pipeline.update_tempo(tempo)
        pipeline.duration = duration
        pipeline.model_type = model_type
        pipeline.melody_generator.model_type = model_type
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{file.filename}")
        file.save(file_path)
        
        # Process the image
        result = pipeline.process_image(
            file_path,
            num_colors=num_colors
        )
        
        # Save generated audio
        audio_filename = f"chromasonic_{file_id}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        pipeline.save_audio(result['audio'], audio_path)
        
        # Convert colors to hex for display
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in result['colors']]
        
        # Extract melody notes for visualization
        melody_notes = result.get('melody', {}).get('notes', [])
        if isinstance(melody_notes, np.ndarray):
            melody_notes = melody_notes.tolist()
        
        # Update metadata with melody
        metadata = result['metadata'].copy()
        metadata['melody'] = melody_notes[:64]  # Limit to first 64 notes
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'audio_url': url_for('download_audio', filename=audio_filename),
            'colors': hex_colors,
            'wavelengths': result['wavelengths'],
            'frequencies': result['frequencies'],
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_base64', methods=['POST'])
def upload_base64_image():
    """Handle base64 image upload from drag-and-drop or paste."""
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Get processing parameters
        num_colors = data.get('num_colors', 8)
        scale = data.get('scale', 'major')
        tempo = data.get('tempo', 120)
        duration = data.get('duration', 30.0)
        model_type = data.get('model_type', 'markov')
        
        # Update pipeline parameters
        pipeline.update_scale(scale)
        pipeline.update_tempo(tempo)
        pipeline.duration = duration
        pipeline.model_type = model_type
        
        # Save image temporarily
        file_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.png")
        
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        # Process the image
        result = pipeline.process_image(
            file_path,
            num_colors=num_colors
        )
        
        # Save generated audio
        audio_filename = f"chromasonic_{file_id}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        pipeline.save_audio(result['audio'], audio_path)
        
        # Convert colors to hex for display
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in result['colors']]
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'audio_url': url_for('download_audio', filename=audio_filename),
            'colors': hex_colors,
            'wavelengths': result['wavelengths'],
            'frequencies': result['frequencies'],
            'metadata': result['metadata']
        })
        
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_audio(filename):
    """Serve generated audio files."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/scales')
def get_scales():
    """Get available musical scales."""
    return jsonify({
        'scales': list(pipeline.melody_generator.scales.keys())
    })


@app.route('/api/models')
def get_models():
    """Get available ML models."""
    return jsonify({
        'models': ['markov', 'lstm', 'transformer']
    })


@app.route('/api/regenerate', methods=['POST'])
def regenerate_melody():
    """Regenerate melody with different parameters using cached colors."""
    try:
        data = request.json
        
        # Required parameters
        colors = data.get('colors', [])
        if not colors:
            return jsonify({'error': 'No colors provided'}), 400
        
        # Convert hex colors back to RGB tuples
        rgb_colors = []
        for hex_color in colors:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_colors.append((r, g, b))
        
        # Get new parameters
        scale = data.get('scale', 'major')
        tempo = data.get('tempo', 120)
        duration = data.get('duration', 30.0)
        model_type = data.get('model_type', 'markov')
        
        # Update pipeline
        pipeline.update_scale(scale)
        pipeline.update_tempo(tempo)
        pipeline.model_type = model_type
        
        # Convert colors to wavelengths and frequencies
        wavelengths = pipeline.wavelength_converter.rgb_to_wavelengths(rgb_colors)
        frequencies = pipeline.wavelength_converter.wavelengths_to_frequencies(
            wavelengths, octave_range=(3, 6)
        )
        
        # Generate new melody
        melody = pipeline.melody_generator.generate_melody(
            frequencies,
            duration=duration,
            scale=scale,
            tempo=tempo
        )
        
        # Synthesize audio
        audio = pipeline.audio_synthesizer.synthesize(melody)
        
        # Save audio
        file_id = str(uuid.uuid4())
        audio_filename = f"chromasonic_regen_{file_id}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        pipeline.save_audio(audio, audio_path)
        
        # Extract melody notes for visualization
        melody_notes = melody.get('notes', [])
        if isinstance(melody_notes, np.ndarray):
            melody_notes = melody_notes.tolist()
        
        return jsonify({
            'success': True,
            'audio_url': url_for('download_audio', filename=audio_filename),
            'frequencies': melody['frequencies'],
            'metadata': {
                'scale': scale,
                'tempo': tempo,
                'duration': duration,
                'model_type': model_type,
                'melody': melody_notes[:64]  # Limit to first 64 notes for visualization
            }
        })
        
    except Exception as e:
        logger.error(f"Error regenerating melody: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)