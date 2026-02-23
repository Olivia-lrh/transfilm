#!/usr/bin/env python3
"""
Simple test script for TransFilm structure
Tests that all files exist and basic structure is correct
"""
import sys
from pathlib import Path


def test_project_structure():
    """Test that all expected files exist"""
    print("\n" + "="*60)
    print("Testing Project Structure")
    print("="*60)
    
    root = Path(__file__).parent
    
    # Required files
    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        "config.py",
        "cli.py",
        "webui.py",
        ".gitignore",
        "transfilm/__init__.py",
        "transfilm/video_processor.py",
        "transfilm/audio_processor.py",
        "transfilm/asr_engine.py",
        "transfilm/tts_engine.py",
        "transfilm/pipeline.py",
        "transfilm/utils.py",
        "examples/example_config.yaml",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files")
        return False
    
    print("\n✅ All required files present!\n")
    return True


def test_imports():
    """Test that modules can be imported"""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    try:
        import config
        print("✓ config module imported")
        
        # Test transfilm package
        sys.path.insert(0, str(Path(__file__).parent))
        import transfilm
        print(f"✓ transfilm package imported (version {transfilm.__version__})")
        
        print("\n✅ Basic imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_config():
    """Test configuration values"""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)
    
    import config
    
    print(f"✓ DEFAULT_ASR_MODEL: {config.DEFAULT_ASR_MODEL}")
    print(f"✓ DEFAULT_TTS_MODEL: {config.DEFAULT_TTS_MODEL}")
    print(f"✓ DEFAULT_DEVICE: {config.DEFAULT_DEVICE}")
    print(f"✓ DEFAULT_CHUNK_SIZE: {config.DEFAULT_CHUNK_SIZE}s")
    print(f"✓ DEFAULT_SAMPLE_RATE: {config.DEFAULT_SAMPLE_RATE} Hz")
    print(f"✓ CACHE_DIR: {config.CACHE_DIR}")
    
    # Check directories exist
    if config.CACHE_DIR.exists():
        print(f"✓ Cache directory exists")
    if config.TEMP_DIR.exists():
        print(f"✓ Temp directory exists")
    if config.MODELS_DIR.exists():
        print(f"✓ Models directory exists")
    
    print("\n✅ Configuration valid!\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TransFilm Structure Test")
    print("="*60)
    
    all_passed = True
    
    # Test structure
    if not test_project_structure():
        all_passed = False
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test config
    if not test_config():
        all_passed = False
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ All Structure Tests Passed!")
        print("="*60 + "\n")
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full tests: python test.py")
        print("3. Test CLI: python cli.py --help")
        print("4. Test WebUI: python webui.py")
        print("\n")
    else:
        print("\n❌ Some tests failed\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
