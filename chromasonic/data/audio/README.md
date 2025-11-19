# Generated Audio Files
This directory stores the generated audio files from the Chromasonic pipeline.

## File Naming Convention
- `chromasonic_YYYYMMDD_HHMMSS.wav` - Timestamped audio files
- `chromasonic_[image_name]_[scale].wav` - Named by source image and scale
- `batch_output_NNN.wav` - Batch processing results

## Audio Specifications
- Format: WAV (16-bit, 44.1kHz)
- Mono channel
- Variable duration (typically 10-60 seconds)

## Cleanup
Old audio files can be safely deleted. The pipeline generates new files for each conversion.