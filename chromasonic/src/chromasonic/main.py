"""Command-line interface for Chromasonic."""

import argparse
import logging
from pathlib import Path
import sys

from chromasonic import ChromasonicPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chromasonic: Convert images to music through color analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate --image sunset.jpg --output melody.wav
  %(prog)s generate --image photo.png --scale minor --tempo 90 --duration 45
  %(prog)s generate --image art.jpg --model lstm --num-colors 12
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate music from an image')
    gen_parser.add_argument('--image', '-i', required=True, 
                           help='Path to input image file')
    gen_parser.add_argument('--output', '-o', 
                           help='Path for output audio file (default: auto-generated)')
    gen_parser.add_argument('--scale', '-s', default='major',
                           choices=['major', 'minor', 'pentatonic', 'blues', 'chromatic', 'dorian', 'mixolydian'],
                           help='Musical scale to use (default: major)')
    gen_parser.add_argument('--tempo', '-t', type=int, default=120,
                           help='Tempo in BPM (default: 120)')
    gen_parser.add_argument('--duration', '-d', type=float, default=30.0,
                           help='Duration in seconds (default: 30.0)')
    gen_parser.add_argument('--model', '-m', default='markov',
                           choices=['markov', 'lstm', 'transformer'],
                           help='ML model to use (default: markov)')
    gen_parser.add_argument('--num-colors', '-n', type=int, default=8,
                           help='Number of colors to extract (default: 8)')
    gen_parser.add_argument('--color-method', default='kmeans',
                           choices=['kmeans', 'quantization', 'histogram'],
                           help='Color extraction method (default: kmeans)')
    gen_parser.add_argument('--synthesis', default='additive',
                           choices=['additive', 'fm', 'subtractive'],
                           help='Audio synthesis method (default: additive)')
    gen_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose logging')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('--input-dir', '-i', required=True,
                             help='Directory containing input images')
    batch_parser.add_argument('--output-dir', '-o', required=True,
                             help='Directory for output audio files')
    batch_parser.add_argument('--pattern', '-p', default='*.{jpg,jpeg,png,gif}',
                             help='File pattern to match (default: *.{jpg,jpeg,png,gif})')
    batch_parser.add_argument('--scale', '-s', default='major',
                             help='Musical scale to use (default: major)')
    batch_parser.add_argument('--tempo', '-t', type=int, default=120,
                             help='Tempo in BPM (default: 120)')
    batch_parser.add_argument('--duration', '-d', type=float, default=30.0,
                             help='Duration in seconds (default: 30.0)')
    batch_parser.add_argument('--model', '-m', default='markov',
                             help='ML model to use (default: markov)')
    batch_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Enable verbose logging')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='127.0.0.1',
                           help='Host to bind to (default: 127.0.0.1)')
    web_parser.add_argument('--port', '-p', type=int, default=5000,
                           help='Port to bind to (default: 5000)')
    web_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'generate':
        return generate_command(args)
    elif args.command == 'batch':
        return batch_command(args)
    elif args.command == 'web':
        return web_command(args)
    else:
        parser.print_help()
        return 1


def generate_command(args):
    """Handle the generate command."""
    image_path = Path(args.image)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    # Generate output filename if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_chromasonic.wav"
    
    try:
        # Initialize pipeline
        pipeline = ChromasonicPipeline(
            model_type=args.model,
            scale=args.scale,
            tempo=args.tempo,
            duration=args.duration
        )
        
        # Set synthesis method
        pipeline.audio_synthesizer.synthesis_method = args.synthesis
        
        print(f"Processing image: {image_path}")
        print(f"Scale: {args.scale}, Tempo: {args.tempo} BPM, Duration: {args.duration}s")
        print(f"Model: {args.model}, Colors: {args.num_colors}")
        
        # Process image
        result = pipeline.process_image(
            str(image_path),
            num_colors=args.num_colors,
            color_method=args.color_method
        )
        
        # Save audio
        pipeline.save_audio(result['audio'], str(output_path))
        
        print(f"\n✅ Success!")
        print(f"Generated audio saved to: {output_path}")
        print(f"Colors extracted: {len(result['colors'])}")
        print(f"Wavelengths: {[f'{w:.1f}nm' for w in result['wavelengths']]}")
        print(f"Frequencies: {[f'{f:.1f}Hz' for f in result['frequencies']]}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1


def batch_command(args):
    """Handle the batch command."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))
    
    if not image_files:
        print(f"No image files found in: {input_dir}")
        return 1
    
    try:
        # Initialize pipeline
        pipeline = ChromasonicPipeline(
            model_type=args.model,
            scale=args.scale,
            tempo=args.tempo,
            duration=args.duration
        )
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {output_dir}")
        
        # Process images
        results = pipeline.batch_process(
            [str(f) for f in image_files],
            str(output_dir)
        )
        
        print(f"\n✅ Batch processing complete!")
        print(f"Successfully processed: {len(results)} images")
        
        return 0
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return 1


def web_command(args):
    """Handle the web command."""
    try:
        from chromasonic.web_interface.app import app
        
        print(f"Starting Chromasonic web interface...")
        print(f"Server: http://{args.host}:{args.port}")
        print(f"Debug mode: {args.debug}")
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
        return 0
        
    except ImportError:
        print("Error: Web interface dependencies not available.")
        print("Install with: pip install flask flask-cors")
        return 1
    except Exception as e:
        print(f"Error starting web server: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())