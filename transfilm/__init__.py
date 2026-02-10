"""
TransFilm - AI Video Dubbing Tool
"""

__version__ = "0.1.0"
__author__ = "Olivia-lrh"

from .pipeline import VideoDubbingPipeline
from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor

__all__ = [
    "VideoDubbingPipeline",
    "VideoProcessor", 
    "AudioProcessor",
]
