"""
Utility functions for TransFilm
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import config


def setup_logger(name: str = "transfilm", level: Optional[str] = None) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    if level is None:
        level = config.LOG_LEVEL
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def get_video_info(video_path: Path) -> dict:
    """Get basic video information"""
    import ffmpeg
    
    try:
        probe = ffmpeg.probe(str(video_path))
        video_info = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )
        audio_info = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
            None
        )
        
        return {
            'has_video': video_info is not None,
            'has_audio': audio_info is not None,
            'duration': float(probe['format'].get('duration', 0)),
            'format': probe['format'].get('format_name', ''),
            'size': int(probe['format'].get('size', 0)),
        }
    except Exception as e:
        raise ValueError(f"Failed to get video info: {e}")


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_temp_files(temp_dir: Path) -> None:
    """Clean up temporary files"""
    import shutil
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
