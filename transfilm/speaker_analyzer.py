"""
Speaker analysis and diarization module for TransFilm
Identifies different speakers in audio and extracts voice samples
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch

from .utils import setup_logger
import config

logger = setup_logger(__name__)


class SpeakerAnalyzer:
    """Speaker diarization and voice analysis"""
    
    def __init__(
        self,
        device: str = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        voice_sample_duration: float = 5.0
    ):
        """
        Initialize speaker analyzer
        
        Args:
            device: Device to run on
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            voice_sample_duration: Duration of voice sample to extract per speaker (seconds)
        """
        self.device = device or config.DEFAULT_DEVICE
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.voice_sample_duration = voice_sample_duration
        
        self.logger = logger
        self.model = None
        self.is_loaded = False
    
    def load_model(self) -> None:
        """Load speaker diarization model"""
        if self.is_loaded:
            self.logger.info("Speaker analyzer model already loaded")
            return
        
        self.logger.info("Loading speaker diarization model")
        
        try:
            # Note: This is a placeholder for actual speaker diarization model
            # Popular options: pyannote.audio, speechbrain, or custom models
            # For production, integrate with models like:
            # - pyannote/speaker-diarization
            # - speechbrain/spkrec-ecapa-voxceleb
            
            # Placeholder initialization
            self.logger.warning(
                "Using placeholder speaker analyzer. "
                "Integrate with pyannote.audio or similar for production."
            )
            
            self.is_loaded = True
            self.logger.info("Speaker analyzer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to load speaker analyzer: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_loaded = False
            self.logger.info("Speaker analyzer model unloaded")
    
    def analyze_speakers(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict]:
        """
        Analyze audio and identify speaker segments
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            List of speaker segments with format:
            [
                {
                    'speaker_id': 'SPEAKER_00',
                    'start': 0.0,  # seconds
                    'end': 5.2,     # seconds
                    'confidence': 0.95
                },
                ...
            ]
        """
        self.load_model()
        
        self.logger.info("Analyzing speakers in audio")
        
        try:
            # Placeholder implementation
            # In production, use actual diarization model
            
            audio_duration = len(audio_data) / sample_rate
            
            # Simulate speaker detection
            # This should be replaced with actual diarization logic
            speaker_segments = self._simulate_speaker_diarization(
                audio_data, sample_rate, audio_duration
            )
            
            self.logger.info(
                f"Detected {len(set(s['speaker_id'] for s in speaker_segments))} speakers "
                f"in {len(speaker_segments)} segments"
            )
            
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Speaker analysis failed: {e}")
            raise
    
    def _simulate_speaker_diarization(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        duration: float
    ) -> List[Dict]:
        """
        Simulate speaker diarization (placeholder)
        
        In production, replace with actual diarization algorithm:
        - Use pyannote.audio for speaker diarization
        - Use speechbrain for speaker embeddings
        - Apply VAD (Voice Activity Detection) first
        - Cluster speaker embeddings
        """
        # Simple simulation: divide audio into segments
        # Alternate between 2 speakers for demonstration
        segments = []
        current_time = 0.0
        speaker_id = 0
        segment_duration = 5.0  # Average segment duration
        
        while current_time < duration:
            end_time = min(current_time + segment_duration, duration)
            
            segments.append({
                'speaker_id': f'SPEAKER_{speaker_id:02d}',
                'start': current_time,
                'end': end_time,
                'confidence': 0.85 + np.random.random() * 0.15  # Simulated confidence
            })
            
            current_time = end_time
            # Alternate speakers (simplified logic)
            speaker_id = (speaker_id + 1) % min(2, self.max_speakers)
        
        return segments
    
    def extract_voice_samples(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_segments: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """
        Extract voice samples for each speaker (for voice cloning)
        
        Args:
            audio_data: Full audio data
            sample_rate: Sample rate
            speaker_segments: Speaker diarization results
            
        Returns:
            Dictionary mapping speaker_id to voice sample audio:
            {
                'SPEAKER_00': audio_sample_array,
                'SPEAKER_01': audio_sample_array,
                ...
            }
        """
        self.logger.info("Extracting voice samples for each speaker")
        
        voice_samples = {}
        
        # Group segments by speaker
        speaker_groups = {}
        for segment in speaker_segments:
            speaker_id = segment['speaker_id']
            if speaker_id not in speaker_groups:
                speaker_groups[speaker_id] = []
            speaker_groups[speaker_id].append(segment)
        
        # Extract sample for each speaker
        for speaker_id, segments in speaker_groups.items():
            # Find best segment for voice sample
            # Prefer longer segments with high confidence
            segments_sorted = sorted(
                segments,
                key=lambda s: (s['end'] - s['start']) * s['confidence'],
                reverse=True
            )
            
            # Extract voice sample from best segment(s)
            voice_sample = self._extract_speaker_sample(
                audio_data,
                sample_rate,
                segments_sorted
            )
            
            voice_samples[speaker_id] = voice_sample
            
            self.logger.info(
                f"Extracted {len(voice_sample)/sample_rate:.2f}s "
                f"voice sample for {speaker_id}"
            )
        
        return voice_samples
    
    def _extract_speaker_sample(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict]
    ) -> np.ndarray:
        """
        Extract voice sample from speaker segments
        
        Tries to extract exactly voice_sample_duration seconds,
        combining multiple segments if needed
        """
        target_samples = int(self.voice_sample_duration * sample_rate)
        voice_sample = np.array([], dtype=audio_data.dtype)
        
        for segment in segments:
            if len(voice_sample) >= target_samples:
                break
            
            # Extract segment audio
            start_idx = int(segment['start'] * sample_rate)
            end_idx = int(segment['end'] * sample_rate)
            segment_audio = audio_data[start_idx:end_idx]
            
            # Append to voice sample
            remaining = target_samples - len(voice_sample)
            voice_sample = np.concatenate([
                voice_sample,
                segment_audio[:remaining]
            ])
        
        # Pad if needed
        if len(voice_sample) < target_samples:
            padding = np.zeros(target_samples - len(voice_sample), dtype=audio_data.dtype)
            voice_sample = np.concatenate([voice_sample, padding])
        
        return voice_sample
    
    def get_speaker_timeline(
        self,
        speaker_segments: List[Dict]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get timeline of when each speaker talks
        
        Args:
            speaker_segments: Speaker diarization results
            
        Returns:
            Dictionary mapping speaker_id to list of (start, end) times
        """
        timeline = {}
        
        for segment in speaker_segments:
            speaker_id = segment['speaker_id']
            if speaker_id not in timeline:
                timeline[speaker_id] = []
            timeline[speaker_id].append((segment['start'], segment['end']))
        
        return timeline
    
    def assign_speaker_to_text(
        self,
        text_segments: List[Dict],
        speaker_segments: List[Dict]
    ) -> List[Dict]:
        """
        Assign speaker IDs to text segments based on timestamps
        
        Args:
            text_segments: Text segments with timestamps
                [{'text': '...', 'start': 0.0, 'end': 2.5}, ...]
            speaker_segments: Speaker diarization results
            
        Returns:
            Text segments with speaker_id added
        """
        self.logger.info("Assigning speakers to text segments")
        
        for text_seg in text_segments:
            # Find speaker with maximum overlap
            text_start = text_seg['start']
            text_end = text_seg['end']
            text_duration = text_end - text_start
            
            best_speaker = None
            max_overlap = 0.0
            
            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(text_start, speaker_seg['start'])
                overlap_end = min(text_end, speaker_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_seg['speaker_id']
            
            # Assign speaker (or 'UNKNOWN' if no overlap)
            text_seg['speaker_id'] = best_speaker or 'UNKNOWN'
            text_seg['speaker_confidence'] = max_overlap / text_duration if text_duration > 0 else 0
        
        return text_segments
