#!/usr/bin/env python3
"""
🎵 Chromasonic Quick Demo - Multiple Image Input Methods
================================================================

This script demonstrates all the ways you can input images to Chromasonic.
"""

import sys
from pathlib import Path
sys.path.append('/workspaces/ColorMe/chromasonic/src')

from chromasonic.pipeline import ChromasonicPipeline
from chromasonic.image_processing.loader import ImageLoader
import numpy as np

def main():
    print("🎵 Chromasonic - Image Input Methods Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ChromasonicPipeline()
    
    # Method 1: Command Line Interface
    print("\n🖥️ Method 1: Command Line Interface")
    print("-" * 40)
    print("Usage:")
    print("  python -m chromasonic.main <image_path> [options]")
    print("  python -m chromasonic.main photo.jpg --scale major --tempo 120")
    print("")
    
    # Method 2: Python API  
    print("🐍 Method 2: Python API")
    print("-" * 40)
    print("Code example:")
    print("""
    from chromasonic.pipeline import ChromasonicPipeline
    
    pipeline = ChromasonicPipeline()
    result = pipeline.process_image('your_image.jpg', 
                                  scale='major', 
                                  tempo=120)
    pipeline.save_audio(result['audio'], 'output.wav')
    """)
    
    # Method 3: Web Interface
    print("\n🌐 Method 3: Web Interface")
    print("-" * 40) 
    print("Start server: python -m flask --app src/chromasonic/web_interface/app.py run")
    print("Then visit:  http://127.0.0.1:5000")
    print("Features:    Drag & drop, real-time controls, instant playback")
    
    # Method 4: Jupyter Notebook
    print("\n📓 Method 4: Jupyter Notebook")
    print("-" * 40)
    print("File:        notebooks/chromasonic_demo.ipynb") 
    print("Features:    Interactive widgets, visualizations, tutorials")
    
    # Demo with sample image if available
    sample_images = [
        '/workspaces/ColorMe/chromasonic/data/images/test_sunset.png',
        '/workspaces/ColorMe/chromasonic/data/images/sample.jpg',
        '/workspaces/ColorMe/chromasonic/data/images/test_image.png'
    ]
    
    print("\n🎯 Live Demo:")
    print("-" * 40)
    
    # Try to find an existing image
    demo_image = None
    for img_path in sample_images:
        if Path(img_path).exists():
            demo_image = img_path
            break
    
    if demo_image:
        print(f"✅ Using sample image: {demo_image}")
        
        try:
            # Quick processing
            result = pipeline.process_image(demo_image, num_colors=6)
            
            print(f"   🎨 Extracted colors: {len(result['colors'])}")
            print(f"   🎵 Generated notes: {len(result['melody']['notes'])}")  
            print(f"   ⏱️ Audio duration: {len(result['audio'])/44100:.1f}s")
            print(f"   🌈 Color palette: {result['colors'][:3]}...")
            
            # Save output
            output_file = "/tmp/chromasonic_demo.wav"
            pipeline.save_audio(result['audio'], output_file)
            print(f"   💾 Audio saved to: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Demo error: {e}")
    else:
        print("⚠️ No sample images found - let's create one!")
        
        # Create a simple test image
        try:
            from PIL import Image, ImageDraw
            import random
            
            # Create colorful test image
            img = Image.new('RGB', (400, 300), 'white')
            draw = ImageDraw.Draw(img)
            
            # Add colorful circles
            colors = [(255,100,100), (100,255,100), (100,100,255), 
                     (255,255,100), (255,100,255), (100,255,255)]
            
            for i, color in enumerate(colors):
                x = (i % 3) * 120 + 50
                y = (i // 3) * 120 + 50
                draw.ellipse([x, y, x+80, y+80], fill=color)
            
            # Save test image
            test_path = "/workspaces/ColorMe/chromasonic/data/images/test_colors.png"
            Path(test_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(test_path)
            
            print(f"   ✅ Created test image: {test_path}")
            
            # Process it
            result = pipeline.process_image(test_path, num_colors=6)
            print(f"   🎨 Extracted {len(result['colors'])} colors")
            print(f"   🎵 Generated {len(result['melody']['notes'])} notes")
            
        except Exception as e:
            print(f"   ❌ Could not create test image: {e}")
    
    print("\n🚀 Next Steps:")
    print("-" * 40)
    print("1. 🌐 Try the web interface for easiest use")
    print("2. 📓 Open the Jupyter notebook for interactive demo") 
    print("3. 🐍 Use the Python API for programmatic access")
    print("4. 🖥️ Use CLI for batch processing")
    
    print("\n💡 Tips:")
    print("• Any image format works: JPG, PNG, GIF, BMP, etc.")
    print("• Larger, more colorful images work best")
    print("• Try different scales: major, minor, pentatonic, blues")
    print("• Experiment with different tempos and durations")

if __name__ == "__main__":
    main()