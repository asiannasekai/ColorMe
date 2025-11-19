#!/usr/bin/env python3
"""
🎵 Chromasonic CLI - Command Line Interface for Image-to-Music Generation
========================================================================

Quick and easy command-line tool to convert images to music.

Usage Examples:
    python chromasonic_cli.py photo.jpg
    python chromasonic_cli.py sunset.png --scale minor --tempo 90
    python chromasonic_cli.py landscape.jpg --colors 10 --duration 30
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from chromasonic.pipeline import ChromasonicPipeline

def main():
    parser = argparse.ArgumentParser(
        description='🎵 Convert images to music using Chromasonic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg                           # Basic conversion
  %(prog)s sunset.png --scale minor            # Use minor scale
  %(prog)s art.jpg --tempo 140 --colors 8      # Custom tempo and colors
  %(prog)s image.png --duration 60 --output music.wav  # Long song
  
Supported image formats: JPG, PNG, GIF, BMP, TIFF, WEBP
Supported scales: major, minor, pentatonic, blues, chromatic, dorian
        """
    )
    
    # Required arguments
    parser.add_argument('image', help='Path to the input image file')
    
    # Optional arguments
    parser.add_argument('--scale', '-s', 
                       choices=['major', 'minor', 'pentatonic', 'blues', 'chromatic', 'dorian'],
                       default='major',
                       help='Musical scale to use (default: major)')
    
    parser.add_argument('--tempo', '-t', 
                       type=int, default=120,
                       help='Tempo in BPM (default: 120)')
    
    parser.add_argument('--colors', '-c', 
                       type=int, default=8,
                       help='Number of colors to extract (default: 8)')
    
    parser.add_argument('--duration', '-d', 
                       type=float, default=15.0,
                       help='Duration in seconds (default: 15.0)')
    
    parser.add_argument('--output', '-o', 
                       default=None,
                       help='Output audio file (default: auto-generated)')
    
    parser.add_argument('--model', '-m',
                       choices=['markov', 'lstm', 'transformer'],
                       default='markov',
                       help='AI melody model (default: markov)')
    
    parser.add_argument('--fusion', '-f',
                       choices=['hard', 'soft', 'weighted', 'alternating', 'harmonic', 'adaptive'],
                       default='adaptive',
                       help='Fusion strategy (default: adaptive)')
    
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Show detailed processing info')
    
    args = parser.parse_args()
    
    # Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image file '{args.image}' not found!")
        return 1
    
    # Set output filename if not specified
    if args.output is None:
        stem = image_path.stem
        args.output = f"chromasonic_{stem}_{args.scale}_{args.tempo}bpm.wav"
    
    print("🎵 Chromasonic - Image to Music Converter")
    print("=" * 45)
    print(f"📁 Input:     {args.image}")
    print(f"🎼 Scale:     {args.scale}")
    print(f"🎵 Tempo:     {args.tempo} BPM")
    print(f"🎨 Colors:    {args.colors}")
    print(f"⏱️ Duration:  {args.duration}s")
    print(f"🧠 Model:     {args.model}")
    print(f"🔀 Fusion:    {args.fusion}")
    print(f"💾 Output:    {args.output}")
    print("")
    
    try:
        # Initialize pipeline
        if args.verbose:
            print("🔧 Initializing Chromasonic pipeline...")
        
        pipeline = ChromasonicPipeline()
        
        # Configure pipeline
        pipeline.update_scale(args.scale)
        pipeline.update_tempo(args.tempo)
        pipeline.duration = args.duration
        
        # Process image
        if args.verbose:
            print("🖼️ Processing image...")
        
        result = pipeline.process_image(
            str(image_path),
            num_colors=args.colors
        )
        
        # Display results
        colors = result['colors']
        melody = result['melody']
        
        print(f"✅ Processing complete!")
        print(f"   🎨 Extracted {len(colors)} dominant colors")
        print(f"   🌈 Color palette: {colors[:3]}{'...' if len(colors) > 3 else ''}")
        print(f"   🎵 Generated {len(melody['notes'])} note melody")
        print(f"   🎼 Key signature: {melody.get('key', 'C major')}")
        
        if args.verbose:
            print("📊 Technical details:")
            wavelengths = result.get('wavelengths', [])
            frequencies = result.get('frequencies', [])
            print(f"   🌈 Wavelength range: {min(wavelengths):.0f}-{max(wavelengths):.0f} nm")
            print(f"   🎵 Frequency range: {min(frequencies):.0f}-{max(frequencies):.0f} Hz")
            print(f"   ⏱️ Audio length: {len(result['audio'])/44100:.1f}s")
        
        # Save audio
        if args.verbose:
            print(f"💾 Saving audio to {args.output}...")
        
        pipeline.save_audio(result['audio'], args.output)
        
        print("")
        print(f"🎉 Success! Music saved to: {args.output}")
        print("🎧 You can now play your generated music!")
        
        # Show next steps
        print("")
        print("💡 Next steps:")
        print(f"   🎧 Play: $BROWSER file://{Path(args.output).absolute()}")
        print("   🌐 Try web interface: python -m flask --app src/chromasonic/web_interface/app.py run")
        print("   📓 Explore notebook: jupyter notebook notebooks/chromasonic_demo.ipynb")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)