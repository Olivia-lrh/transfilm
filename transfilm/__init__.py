"""
TransFilm - AI Video Dubbing Tool
"""

__version__ = "0.1.0"
__author__ = "Olivia-lrh"

# Lazy imports to avoid loading heavy dependencies at package level
def __getattr__(name):
    if name == "VideoDubbingPipeline":
        from .pipeline import VideoDubbingPipeline
        return VideoDubbingPipeline
    elif name == "VideoProcessor":
        from .video_processor import VideoProcessor
        return VideoProcessor
    elif name == "AudioProcessor":
        from .audio_processor import AudioProcessor
        return AudioProcessor
    elif name == "ASREngine":
        from .asr_engine import ASREngine
        return ASREngine
    elif name == "TranslationEngine":
        from .translation_engine import TranslationEngine
        return TranslationEngine
    elif name == "ForcedAlignerEngine":
        from .forced_aligner_engine import ForcedAlignerEngine
        return ForcedAlignerEngine
    elif name == "TTSEngine":
        from .tts_engine import TTSEngine
        return TTSEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "VideoDubbingPipeline",
    "VideoProcessor", 
    "AudioProcessor",
    "ASREngine",
    "TranslationEngine",
    "ForcedAlignerEngine",
    "TTSEngine",
]
