"""
Audio processing module for TransFilm
Handles audio chunking, loading, and manipulation
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import soundfile as sf
from pydub import AudioSegment
from .utils import setup_logger

logger = setup_logger(__name__)


class AudioProcessor:
    """Handle audio file operations and processing"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.logger = logger
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        self.logger.info(f"Loading audio from {audio_path}")
        
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            self.logger.info(f"Audio loaded: {len(audio)} samples at {sr} Hz")
            return audio, sr
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            raise
    
    def save_audio(
        self,
        audio_data: np.ndarray,
        output_path: Path,
        sample_rate: Optional[int] = None
    ) -> Path:
        """
        Save audio to file
        
        Args:
            audio_data: Audio data as numpy array
            output_path: Path to output file
            sample_rate: Sample rate (uses default if not specified)
            
        Returns:
            Path to saved audio file
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        self.logger.info(f"Saving audio to {output_path}")
        
        try:
            sf.write(str(output_path), audio_data, sample_rate)
            self.logger.info(f"Audio saved successfully")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise
    
    def chunk_audio(
        self,
        audio_data: np.ndarray,
        chunk_size_seconds: float,
        overlap_seconds: float = 0.5
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into overlapping chunks
        
        Args:
            audio_data: Audio data as numpy array
            chunk_size_seconds: Size of each chunk in seconds
            overlap_seconds: Overlap between chunks in seconds
            
        Returns:
            List of tuples (chunk_data, start_time, end_time)
        """
        self.logger.info(
            f"Chunking audio: {len(audio_data)/self.sample_rate:.2f}s "
            f"into {chunk_size_seconds}s chunks with {overlap_seconds}s overlap"
        )
        
        chunk_size = int(chunk_size_seconds * self.sample_rate)
        overlap_size = int(overlap_seconds * self.sample_rate)
        step_size = chunk_size - overlap_size
        
        chunks = []
        start = 0
        
        while start < len(audio_data):
            end = min(start + chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            
            chunks.append((chunk, start_time, end_time))
            
            if end >= len(audio_data):
                break
            
            start += step_size
        
        self.logger.info(f"Created {len(chunks)} audio chunks")
        return chunks
    
    def concatenate_audio(
        self,
        audio_chunks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Concatenate audio chunks
        
        Args:
            audio_chunks: List of audio data arrays
            
        Returns:
            Concatenated audio data
        """
        self.logger.info(f"Concatenating {len(audio_chunks)} audio chunks")
        
        try:
            concatenated = np.concatenate(audio_chunks)
            self.logger.info(
                f"Concatenated audio: {len(concatenated)/self.sample_rate:.2f}s"
            )
            return concatenated
        except Exception as e:
            self.logger.error(f"Failed to concatenate audio: {e}")
            raise
    
    def adjust_audio_length(
        self,
        audio_data: np.ndarray,
        target_length: int,
        method: str = 'stretch'
    ) -> np.ndarray:
        """
        Adjust audio length to match target
        
        Args:
            audio_data: Audio data to adjust
            target_length: Target length in samples
            method: Method to use ('stretch', 'pad', 'trim')
            
        Returns:
            Adjusted audio data
        """
        current_length = len(audio_data)
        
        if current_length == target_length:
            return audio_data
        
        if method == 'stretch':
            # Time stretch using librosa
            rate = current_length / target_length
            stretched = librosa.effects.time_stretch(audio_data, rate=rate)
            # Trim or pad to exact length
            if len(stretched) > target_length:
                return stretched[:target_length]
            elif len(stretched) < target_length:
                padding = target_length - len(stretched)
                return np.pad(stretched, (0, padding), mode='constant')
            return stretched
        
        elif method == 'pad':
            if current_length < target_length:
                padding = target_length - current_length
                return np.pad(audio_data, (0, padding), mode='constant')
            else:
                return audio_data[:target_length]
        
        elif method == 'trim':
            return audio_data[:target_length]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def extract_voice_features(
        self,
        audio_data: np.ndarray
    ) -> dict:
        """
        Extract voice characteristics from audio
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary of voice features
        """
        self.logger.info("Extracting voice features")
        
        try:
            # Extract various audio features
            features = {}
            
            # Pitch (F0)
            f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=self.sample_rate)
            features['mean_pitch'] = float(np.nanmean(f0))
            features['std_pitch'] = float(np.nanstd(f0))
            
            # Energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features['mean_energy'] = float(np.mean(rms))
            features['std_energy'] = float(np.std(rms))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            features['mean_spectral_centroid'] = float(np.mean(spectral_centroids))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['tempo'] = float(tempo)
            
            self.logger.info(f"Extracted features: {features}")
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to extract some features: {e}")
            return {}
    
    def get_audio_duration(self, audio_data: np.ndarray) -> float:
        """Get audio duration in seconds"""
        return len(audio_data) / self.sample_rate
