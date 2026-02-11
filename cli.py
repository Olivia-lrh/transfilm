#!/usr/bin/env python3
"""
Command-line interface for TransFilm
"""
import argparse
import sys
from pathlib import Path

from transfilm import VideoDubbingPipeline
from transfilm.utils import setup_logger, get_video_info
import config

logger = setup_logger("cli")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="TransFilm - AI Video Dubbing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py --input video.mp4 --output output.mp4
  
  # Use specific models
  python cli.py --input video.mp4 --output output.mp4 \\
    --asr-model Qwen/Qwen3-ASR --tts-model Qwen/Qwen3-TTS
  
  # Run on CPU with custom chunk size
  python cli.py --input video.mp4 --output output.mp4 \\
    --device cpu --chunk-size 15
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output video file path"
    )
    
    # Model arguments
    parser.add_argument(
        "--asr-model",
        type=str,
        default=config.DEFAULT_ASR_MODEL,
        help=f"ASR model name or path (default: {config.DEFAULT_ASR_MODEL})"
    )
    parser.add_argument(
        "--translation-model",
        type=str,
        default=config.DEFAULT_TRANSLATION_MODEL,
        help=f"Translation model name or path (default: {config.DEFAULT_TRANSLATION_MODEL})"
    )
    parser.add_argument(
        "--forced-aligner-model",
        type=str,
        default=config.DEFAULT_FORCED_ALIGNER_MODEL,
        help=f"Forced aligner model name or path (default: {config.DEFAULT_FORCED_ALIGNER_MODEL})"
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default=config.DEFAULT_TTS_MODEL,
        help=f"TTS model name or path (default: {config.DEFAULT_TTS_MODEL})"
    )
    
    # Processing arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=config.DEFAULT_DEVICE,
        help=f"Device to run on (default: {config.DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=config.DEFAULT_CHUNK_SIZE,
        help=f"Audio chunk size in seconds (default: {config.DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["zh", "en"],
        default=config.DEFAULT_LANGUAGE,
        help=f"Source language code (default: {config.DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--target-language",
        type=str,
        choices=["zh", "en"],
        default=config.DEFAULT_TARGET_LANGUAGE,
        help=f"Target language code (default: {config.DEFAULT_TARGET_LANGUAGE})"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=config.DEFAULT_SAMPLE_RATE,
        help=f"Audio sample rate in Hz (default: {config.DEFAULT_SAMPLE_RATE})"
    )
    
    # Additional options
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after processing"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display video information and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def progress_callback(message: str, progress: float):
    """Print progress updates"""
    bar_length = 40
    filled_length = int(bar_length * progress / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r[{bar}] {progress:.1f}% - {message}', end='', flush=True)
    
    if progress >= 100:
        print()  # New line when complete


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        config.LOG_LEVEL = "DEBUG"
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Show video info if requested
    if args.info:
        try:
            info = get_video_info(input_path)
            print(f"\n{'='*60}")
            print(f"Video Information: {input_path.name}")
            print(f"{'='*60}")
            print(f"Duration:     {info['duration']:.2f} seconds")
            print(f"Format:       {info['format']}")
            print(f"Size:         {info['size'] / (1024*1024):.2f} MB")
            print(f"Has Video:    {info['has_video']}")
            print(f"Has Audio:    {info['has_audio']}")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            sys.exit(1)
        sys.exit(0)
    
    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Display configuration
        print("\n" + "="*60)
        print("TransFilm - AI Video Dubbing (Enhanced Workflow)")
        print("="*60)
        print(f"Input:             {input_path}")
        print(f"Output:            {output_path}")
        print(f"ASR Model:         {args.asr_model}")
        print(f"Translation Model: {args.translation_model}")
        print(f"Aligner Model:     {args.forced_aligner_model}")
        print(f"TTS Model:         {args.tts_model}")
        print(f"Device:            {args.device}")
        print(f"Chunk Size:        {args.chunk_size}s")
        print(f"Source Language:   {args.language}")
        print(f"Target Language:   {args.target_language}")
        print(f"Sample Rate:       {args.sample_rate} Hz")
        print("="*60 + "\n")
        
        # Create pipeline
        pipeline = VideoDubbingPipeline(
            asr_model=args.asr_model,
            translation_model=args.translation_model,
            forced_aligner_model=args.forced_aligner_model,
            tts_model=args.tts_model,
            device=args.device,
            chunk_size=args.chunk_size,
            source_language=args.language,
            target_language=args.target_language,
            sample_rate=args.sample_rate,
            progress_callback=progress_callback
        )
        
        # Process video
        result = pipeline.process_video(
            input_path,
            output_path,
            keep_temp=args.keep_temp
        )
        
        print(f"\n✅ Success! Output saved to: {result}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=args.verbose)
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
