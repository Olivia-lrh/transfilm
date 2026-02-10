"""
Video processing module for TransFilm
Handles video/audio separation and recombination
"""
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import ffmpeg
from .utils import setup_logger

logger = setup_logger(__name__)


class VideoProcessor:
    """Handle video file operations"""
    
    def __init__(self):
        """Initialize video processor"""
        self.logger = logger
    
    @staticmethod
    def _parse_frame_rate(frame_rate_str: str) -> float:
        """
        Safely parse frame rate string (e.g., '30000/1001' or '30')
        
        Args:
            frame_rate_str: Frame rate as string (fraction or decimal)
            
        Returns:
            Frame rate as float
        """
        try:
            if '/' in frame_rate_str:
                numerator, denominator = frame_rate_str.split('/')
                return float(numerator) / float(denominator)
            else:
                return float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def extract_audio(
        self,
        video_path: Path,
        audio_path: Path,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Path:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video file
            audio_path: Path to output audio file
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            
        Returns:
            Path to extracted audio file
        """
        self.logger.info(f"Extracting audio from {video_path}")
        
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(audio_path),
                    acodec='pcm_s16le',
                    ac=channels,
                    ar=sample_rate
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            self.logger.info(f"Audio extracted successfully to {audio_path}")
            return audio_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"Failed to extract audio: {error_msg}")
            raise RuntimeError(f"Audio extraction failed: {error_msg}")
    
    def combine_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        video_codec: str = 'libx264',
        audio_codec: str = 'aac',
        audio_bitrate: str = '192k'
    ) -> Path:
        """
        Combine audio and video into a single file
        
        Args:
            video_path: Path to input video file (without audio)
            audio_path: Path to input audio file
            output_path: Path to output combined file
            video_codec: Video codec to use
            audio_codec: Audio codec to use
            audio_bitrate: Audio bitrate
            
        Returns:
            Path to combined video file
        """
        self.logger.info(f"Combining audio and video into {output_path}")
        
        try:
            video_stream = ffmpeg.input(str(video_path))
            audio_stream = ffmpeg.input(str(audio_path))
            
            (
                ffmpeg
                .output(
                    video_stream,
                    audio_stream,
                    str(output_path),
                    vcodec=video_codec,
                    acodec=audio_codec,
                    audio_bitrate=audio_bitrate,
                    shortest=None  # Use shortest stream duration
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            self.logger.info(f"Video combined successfully to {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"Failed to combine audio and video: {error_msg}")
            raise RuntimeError(f"Video combination failed: {error_msg}")
    
    def get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        try:
            probe = ffmpeg.probe(str(video_path))
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            self.logger.error(f"Failed to get video duration: {e}")
            raise
    
    def get_video_info(self, video_path: Path) -> dict:
        """Get detailed video information"""
        try:
            probe = ffmpeg.probe(str(video_path))
            
            video_info = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            audio_info = next(
                (s for s in probe['streams'] if s['codec_type'] == 'audio'),
                None
            )
            
            return {
                'duration': float(probe['format'].get('duration', 0)),
                'size': int(probe['format'].get('size', 0)),
                'format': probe['format'].get('format_name', ''),
                'video': {
                    'codec': video_info.get('codec_name', '') if video_info else '',
                    'width': video_info.get('width', 0) if video_info else 0,
                    'height': video_info.get('height', 0) if video_info else 0,
                    'fps': self._parse_frame_rate(video_info.get('r_frame_rate', '0/1')) if video_info else 0,
                } if video_info else None,
                'audio': {
                    'codec': audio_info.get('codec_name', '') if audio_info else '',
                    'sample_rate': int(audio_info.get('sample_rate', 0)) if audio_info else 0,
                    'channels': audio_info.get('channels', 0) if audio_info else 0,
                } if audio_info else None,
            }
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            raise
    
    def extract_video_without_audio(
        self,
        video_path: Path,
        output_path: Path,
        video_codec: str = 'copy'
    ) -> Path:
        """
        Extract video stream without audio
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file (no audio)
            video_codec: Video codec ('copy' to copy without re-encoding)
            
        Returns:
            Path to video file without audio
        """
        self.logger.info(f"Extracting video without audio from {video_path}")
        
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    vcodec=video_codec,
                    an=None  # Disable audio
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            self.logger.info(f"Video extracted successfully to {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            self.logger.error(f"Failed to extract video: {error_msg}")
            raise RuntimeError(f"Video extraction failed: {error_msg}")
