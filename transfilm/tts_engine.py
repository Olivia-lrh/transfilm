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
        target_duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            sample_rate: Target sample rate
            voice_features: Voice characteristics to apply
            target_duration: Target duration in seconds
            
        Returns:
            Audio data as numpy array
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Synthesizing text: {text[:100]}...")
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if not (self.use_8bit or self.use_4bit):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate audio
            with torch.no_grad():
                # Note: Actual Qwen3-TTS API may differ
                # This is a placeholder implementation
                audio_values = self.model.generate(
                    **inputs,
                    sample_rate=sample_rate,
                    max_length=int(sample_rate * 30)  # Max 30 seconds
                )
            
            # Convert to numpy
            if isinstance(audio_values, torch.Tensor):
                audio_data = audio_values.cpu().numpy().squeeze()
            else:
                audio_data = audio_values
            
            # Apply voice features if provided
            if voice_features and config.ENABLE_VOICE_CLONING:
                audio_data = self._apply_voice_features(audio_data, voice_features)
            
            # Adjust duration if specified
            if target_duration is not None:
                target_samples = int(target_duration * sample_rate)
                if len(audio_data) != target_samples:
                    # Simple time stretching
                    import librosa
                    rate = len(audio_data) / target_samples
                    audio_data = librosa.effects.time_stretch(audio_data, rate=rate)
                    # Ensure exact length
                    if len(audio_data) > target_samples:
                        audio_data = audio_data[:target_samples]
                    elif len(audio_data) < target_samples:
                        audio_data = np.pad(
                            audio_data,
                            (0, target_samples - len(audio_data)),
                            mode='constant'
                        )
            
            self.logger.info(f"Synthesized audio: {len(audio_data)/sample_rate:.2f}s")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to synthesize audio: {e}")
            raise
    
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
    
    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if config.ENABLE_MODEL_OFFLOADING:
            self.unload_model()
