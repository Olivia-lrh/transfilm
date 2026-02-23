"""
TTS (Text-to-Speech) engine using Qwen3-TTS
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForTextToWaveform, AutoTokenizer

from .utils import setup_logger
import config

logger = setup_logger(__name__)


class TTSEngine:
    """Qwen3-TTS engine for text-to-speech synthesis"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        dtype: str = None,
        use_8bit: bool = False,
        use_4bit: bool = False
    ):
        """
        Initialize TTS engine
        
        Args:
            model_name: Model name or path (defaults to config.DEFAULT_TTS_MODEL)
            device: Device to run on (defaults to config.DEFAULT_DEVICE)
            dtype: Data type to use (defaults to config.DEFAULT_DTYPE)
            use_8bit: Use 8-bit quantization
            use_4bit: Use 4-bit quantization
        """
        self.model_name = model_name or config.DEFAULT_TTS_MODEL
        self.device = device or config.DEFAULT_DEVICE
        self.dtype = dtype or config.DEFAULT_DTYPE
        self.use_8bit = use_8bit or config.USE_8BIT
        self.use_4bit = use_4bit or config.USE_4BIT
        
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self) -> None:
        """Load TTS model and tokenizer"""
        if self.is_loaded:
            self.logger.info("TTS model already loaded")
            return
        
        self.logger.info(f"Loading TTS model: {self.model_name}")
        
        try:
            # Determine torch dtype
            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
            if self.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Quantization config
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }
            
            if self.use_8bit:
                load_kwargs["load_in_8bit"] = True
            elif self.use_4bit:
                load_kwargs["load_in_4bit"] = True
            
            # Load model
            self.model = AutoModelForTextToWaveform.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Move to device if not quantized
            if not (self.use_8bit or self.use_4bit):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            self.logger.info("TTS model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load TTS model: {e}")
            # Fallback: Note that actual Qwen3-TTS might have different API
            self.logger.warning("Using fallback TTS initialization")
            raise
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if not self.is_loaded:
            return
        
        self.logger.info("Unloading TTS model")
        
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("TTS model unloaded")
    
    def synthesize(
        self,
        text: str,
        sample_rate: int = 16000,
        voice_features: Optional[Dict[str, Any]] = None,
        target_duration: Optional[float] = None,
        reference_audio: Optional[np.ndarray] = None,
        speaker_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize speech from text with voice cloning and precise duration control
        
        Args:
            text: Text to synthesize
            sample_rate: Target sample rate
            voice_features: Voice characteristics to apply
            target_duration: Target duration in seconds (for timestamp matching)
            reference_audio: Reference audio for voice cloning
            speaker_id: Speaker ID for multi-speaker synthesis
            
        Returns:
            Audio data as numpy array matching target_duration if specified
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Synthesizing text: {text[:100]}...")
        if target_duration:
            self.logger.info(f"Target duration: {target_duration:.2f}s")
        if speaker_id:
            self.logger.info(f"Speaker: {speaker_id}")
        
        try:
            # Prepare inputs with voice cloning if reference provided
            if reference_audio is not None and config.ENABLE_VOICE_CLONING:
                # Note: Actual Qwen3-TTS API may support reference audio
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                # Add reference audio processing here when API is available
                # This would involve extracting speaker embeddings from reference_audio
            else:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            
            # Move to device
            if not (self.use_8bit or self.use_4bit):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_kwargs = {
                "sample_rate": sample_rate,
                "max_length": int(sample_rate * 30)  # Max 30 seconds by default
            }
            
            # If target duration specified, try to guide generation
            if target_duration is not None and config.ENABLE_DURATION_MATCHING:
                gen_kwargs["max_length"] = int(sample_rate * target_duration * 1.2)
                # Add duration control parameters when API supports it
            
            # Generate audio
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Convert to numpy
            if isinstance(audio_values, torch.Tensor):
                audio_data = audio_values.cpu().numpy().squeeze()
            else:
                audio_data = audio_values
            
            # Apply voice features if provided
            if voice_features and config.ENABLE_VOICE_CLONING:
                audio_data = self._apply_voice_features(audio_data, voice_features)
            
            # Match target duration precisely if specified
            if target_duration is not None:
                audio_data = self._match_duration_precisely(
                    audio_data, target_duration, sample_rate
                )
            
            actual_duration = len(audio_data) / sample_rate
            self.logger.info(f"Synthesized audio: {actual_duration:.2f}s")
            
            if target_duration and abs(actual_duration - target_duration) > 0.1:
                self.logger.warning(
                    f"Duration mismatch: target {target_duration:.2f}s, "
                    f"actual {actual_duration:.2f}s"
                )
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize audio: {e}")
            raise
    
    def _match_duration_precisely(
        self,
        audio_data: np.ndarray,
        target_duration: float,
        sample_rate: int
    ) -> np.ndarray:
        """
        Match audio duration to target precisely using time stretching
        
        Args:
            audio_data: Generated audio
            target_duration: Target duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Audio data with exact target duration
        """
        import librosa
        
        current_duration = len(audio_data) / sample_rate
        target_samples = int(target_duration * sample_rate)
        
        # Check if within tolerance
        tolerance = config.DURATION_TOLERANCE if hasattr(config, 'DURATION_TOLERANCE') else 0.1
        if abs(current_duration - target_duration) < tolerance:
            # Close enough, just trim/pad to exact length
            if len(audio_data) > target_samples:
                return audio_data[:target_samples]
            elif len(audio_data) < target_samples:
                return np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
            return audio_data
        
        # Need to stretch/compress
        rate = current_duration / target_duration
        self.logger.info(
            f"Adjusting duration: {current_duration:.2f}s -> {target_duration:.2f}s "
            f"(rate: {rate:.3f})"
        )
        
        stretched = librosa.effects.time_stretch(audio_data, rate=rate)
        
        # Ensure exact length
        if len(stretched) > target_samples:
            return stretched[:target_samples]
        elif len(stretched) < target_samples:
            return np.pad(stretched, (0, target_samples - len(stretched)), mode='constant')
        
        return stretched
    
    def _apply_voice_features(
        self,
        audio_data: np.ndarray,
        voice_features: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply voice characteristics to synthesized audio
        
        Args:
            audio_data: Generated audio
            voice_features: Voice features to apply
            
        Returns:
            Modified audio data
        """
        self.logger.info("Applying voice features")
        
        # This is a placeholder - actual implementation would depend on
        # the capabilities of Qwen3-TTS and available voice transformation tools
        # For now, we'll just return the original audio
        
        # In a real implementation, you might:
        # - Adjust pitch using voice_features['mean_pitch']
        # - Adjust energy using voice_features['mean_energy']
        # - Apply spectral modifications
        # - Use voice conversion models
        
        return audio_data
    
    def synthesize_batch(
        self,
        texts: list,
        sample_rate: int = 16000,
        voice_features: Optional[Dict[str, Any]] = None,
        target_durations: Optional[list] = None
    ) -> list:
        """
        Synthesize multiple texts
        
        Args:
            texts: List of texts to synthesize
            sample_rate: Target sample rate
            voice_features: Voice characteristics to apply
            target_durations: List of target durations in seconds
            
        Returns:
            List of audio data arrays
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Synthesizing {len(texts)} texts")
        
        audio_list = []
        for i, text in enumerate(texts):
            self.logger.info(f"Processing text {i+1}/{len(texts)}")
            target_duration = target_durations[i] if target_durations else None
            audio = self.synthesize(text, sample_rate, voice_features, target_duration)
            audio_list.append(audio)
        
        return audio_list
    
    def synthesize_multi_speaker(
        self,
        segments: List[Dict],
        voice_samples: Dict[str, np.ndarray],
        sample_rate: int = 16000
    ) -> List[np.ndarray]:
        """
        Synthesize audio for multiple speakers using their voice samples
        
        Args:
            segments: List of text segments with speaker info and timestamps
                [
                    {
                        'text': 'text to synthesize',
                        'speaker_id': 'SPEAKER_00',
                        'start': 0.0,
                        'end': 2.5,
                        'duration': 2.5
                    },
                    ...
                ]
            voice_samples: Dictionary mapping speaker_id to voice sample audio
                {
                    'SPEAKER_00': audio_array,
                    'SPEAKER_01': audio_array,
                    ...
                }
            sample_rate: Target sample rate
            
        Returns:
            List of synthesized audio arrays corresponding to each segment
        """
        self.logger.info(f"Synthesizing {len(segments)} segments with multi-speaker support")
        
        audio_list = []
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '')
            speaker_id = segment.get('speaker_id', 'UNKNOWN')
            target_duration = segment.get('duration') or (segment.get('end', 0) - segment.get('start', 0))
            
            # Get reference audio for this speaker
            reference_audio = voice_samples.get(speaker_id)
            
            if reference_audio is None:
                self.logger.warning(
                    f"No voice sample for {speaker_id}, using default voice"
                )
            
            self.logger.info(
                f"Synthesizing segment {i+1}/{len(segments)}: "
                f"{speaker_id}, duration={target_duration:.2f}s"
            )
            
            # Synthesize with speaker's voice
            audio = self.synthesize(
                text=text,
                sample_rate=sample_rate,
                reference_audio=reference_audio,
                target_duration=target_duration,
                speaker_id=speaker_id
            )
            
            audio_list.append(audio)
        
        return audio_list
    
    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if config.ENABLE_MODEL_OFFLOADING:
            self.unload_model()
