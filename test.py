#!/usr/bin/env python3
"""
Test script for TransFilm
Tests basic functionality without requiring actual models
"""
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from transfilm.video_processor import VideoProcessor
from transfilm.audio_processor import AudioProcessor
from transfilm.utils import setup_logger, format_time

logger = setup_logger("test")


def test_audio_processor():
    """Test audio processor functionality"""
    print("\n" + "="*60)
    print("Testing Audio Processor")
    print("="*60)
    
    processor = AudioProcessor(sample_rate=16000)
    
    # Test audio generation
    duration = 5.0  # seconds
    sample_rate = 16000
    audio_data = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.1
    
    print(f"✓ Generated test audio: {duration}s at {sample_rate} Hz")
    
    # Test chunking
    chunks = processor.chunk_audio(audio_data, chunk_size_seconds=2.0, overlap_seconds=0.5)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Test concatenation
    chunk_data = [chunk[0] for chunk in chunks]
    concatenated = processor.concatenate_audio(chunk_data)
    print(f"✓ Concatenated audio: {len(concatenated)/sample_rate:.2f}s")
    
    # Test length adjustment
    target_length = int(3.5 * sample_rate)
    adjusted = processor.adjust_audio_length(audio_data, target_length, method='pad')
    print(f"✓ Adjusted audio length: {len(adjusted)/sample_rate:.2f}s")
    
    # Test feature extraction
    features = processor.extract_voice_features(audio_data)
    print(f"✓ Extracted features: {list(features.keys())}")
    
    print("\n✅ Audio Processor tests passed!\n")


def test_utilities():
    """Test utility functions"""
    print("\n" + "="*60)
    print("Testing Utilities")
    print("="*60)
    
    # Test time formatting
    test_times = [0, 1.5, 60, 125.75, 3661.123]
    for t in test_times:
        formatted = format_time(t)
        print(f"✓ {t}s -> {formatted}")
    
    print("\n✅ Utility tests passed!\n")


def test_config():
    """Test configuration"""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)
    
    import config
    
    print(f"✓ ASR Model: {config.DEFAULT_ASR_MODEL}")
    print(f"✓ TTS Model: {config.DEFAULT_TTS_MODEL}")
    print(f"✓ Device: {config.DEFAULT_DEVICE}")
    print(f"✓ Chunk Size: {config.DEFAULT_CHUNK_SIZE}s")
    print(f"✓ Sample Rate: {config.DEFAULT_SAMPLE_RATE} Hz")
    print(f"✓ Language: {config.DEFAULT_LANGUAGE}")
    print(f"✓ Cache Dir: {config.CACHE_DIR}")
    print(f"✓ Temp Dir: {config.TEMP_DIR}")
    print(f"✓ Models Dir: {config.MODELS_DIR}")
    
    print("\n✅ Configuration tests passed!\n")


def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    try:
        from transfilm import VideoDubbingPipeline, VideoProcessor, AudioProcessor
        print("✓ Main modules imported successfully")
        
        from transfilm.asr_engine import ASREngine
        print("✓ ASR engine imported successfully")
        
        from transfilm.tts_engine import TTSEngine
        print("✓ TTS engine imported successfully")
        
        from transfilm.pipeline import VideoDubbingPipeline
        print("✓ Pipeline imported successfully")
        
        from transfilm.utils import setup_logger, format_time
        print("✓ Utils imported successfully")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TransFilm Test Suite")
    print("="*60)
    
    try:
        # Test imports
        if not test_imports():
            sys.exit(1)
        
        # Test configuration
        test_config()
        
        # Test utilities
        test_utilities()
        
        # Test audio processor
        test_audio_processor()
        
        print("\n" + "="*60)
        print("✅ All Tests Passed!")
        print("="*60 + "\n")
        
        print("Note: This test suite does not test video processing or actual")
        print("ASR/TTS functionality as those require FFmpeg and model downloads.")
        print("\nTo test the full pipeline, use:")
        print("  python cli.py --input sample_video.mp4 --output output.mp4")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
