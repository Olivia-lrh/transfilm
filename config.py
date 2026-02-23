"""
Configuration file for TransFilm
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / ".cache"
TEMP_DIR = PROJECT_ROOT / ".temp"
MODELS_DIR = CACHE_DIR / "models"

# Create directories if they don't exist
CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configurations
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_TRANSLATION_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # LLM for segmentation and analysis

# Device configuration
DEFAULT_DEVICE = "cuda"  # Options: "cuda", "cpu"
DEFAULT_DTYPE = "float16"  # Options: "float32", "float16", "bfloat16"

# Processing parameters
DEFAULT_CHUNK_SIZE = 30  # seconds
DEFAULT_SAMPLE_RATE = 16000  # Hz
DEFAULT_LANGUAGE = "zh"  # Options: "zh", "en"
DEFAULT_TARGET_LANGUAGE = "en"  # Translation target language

# Translation settings
ENABLE_CHARACTER_COUNT_MATCHING = True  # Match character counts for similar audio duration
CHARACTER_COUNT_TOLERANCE = 0.2  # Allow 20% variation in character count
ENABLE_DURATION_MATCHING = True  # Match audio duration to timestamps
DURATION_TOLERANCE = 0.1  # Allow 10% variation in duration

# Memory optimization settings
ENABLE_MODEL_OFFLOADING = True  # Offload models to CPU when not in use
MAX_MEMORY_PER_GPU = None  # Set to limit GPU memory usage (e.g., "8GB")
USE_8BIT = False  # Use 8-bit quantization for models
USE_4BIT = False  # Use 4-bit quantization for models

# Audio processing
AUDIO_FORMAT = "wav"
AUDIO_CODEC = "pcm_s16le"
VIDEO_CODEC = "libx264"
AUDIO_BITRATE = "192k"

# Web UI configuration
WEBUI_HOST = "0.0.0.0"
WEBUI_PORT = 7860
WEBUI_SHARE = False  # Set to True to create public link

# Logging
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE = PROJECT_ROOT / "transfilm.log"

# Advanced settings
ENABLE_VOICE_CLONING = True  # Enable voice characteristic preservation
ENABLE_AUDIO_ENHANCEMENT = False  # Post-process audio for better quality
MAX_CONCURRENT_CHUNKS = 1  # Number of chunks to process simultaneously

# Speaker analysis settings
ENABLE_SPEAKER_DIARIZATION = True  # Enable speaker identification
MIN_SPEAKERS = 1  # Minimum number of speakers to detect
MAX_SPEAKERS = 10  # Maximum number of speakers to detect
VOICE_SAMPLE_DURATION = 5.0  # Duration of voice sample per speaker (seconds)

# LLM-enhanced features
ENABLE_LLM_SEGMENTATION = True  # Use LLM for intelligent sentence segmentation
ENABLE_LLM_SPEAKER_ANALYSIS = True  # Use LLM to enhance speaker identification
ENABLE_LLM_LENGTH_ADJUSTMENT = True  # Use LLM to adjust translation length

# Model cache settings (for HuggingFace and ModelScope)
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("MODELSCOPE_CACHE", str(MODELS_DIR / "modelscope"))
