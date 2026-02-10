"""
ASR (Automatic Speech Recognition) engine using Qwen3-ASR
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .utils import setup_logger
import config

logger = setup_logger(__name__)


class ASREngine:
    """Qwen3-ASR engine for speech recognition"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        dtype: str = None,
        use_8bit: bool = False,
        use_4bit: bool = False
    ):
        """
        Initialize ASR engine
        
        Args:
            model_name: Model name or path (defaults to config.DEFAULT_ASR_MODEL)
            device: Device to run on (defaults to config.DEFAULT_DEVICE)
            dtype: Data type to use (defaults to config.DEFAULT_DTYPE)
            use_8bit: Use 8-bit quantization
            use_4bit: Use 4-bit quantization
        """
        self.model_name = model_name or config.DEFAULT_ASR_MODEL
        self.device = device or config.DEFAULT_DEVICE
        self.dtype = dtype or config.DEFAULT_DTYPE
        self.use_8bit = use_8bit or config.USE_8BIT
        self.use_4bit = use_4bit or config.USE_4BIT
        
        self.logger = logger
        self.model = None
        self.processor = None
        self.is_loaded = False
    
    def load_model(self) -> None:
        """Load ASR model and processor"""
        if self.is_loaded:
            self.logger.info("ASR model already loaded")
            return
        
        self.logger.info(f"Loading ASR model: {self.model_name}")
        
        try:
            # Determine torch dtype
            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
            if self.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
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
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Move to device if not quantized
            if not (self.use_8bit or self.use_4bit):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            self.logger.info("ASR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if not self.is_loaded:
            return
        
        self.logger.info("Unloading ASR model")
        
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("ASR model unloaded")
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: str = "zh"
    ) -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Language code
            
        Returns:
            Transcribed text
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Transcribing audio: {len(audio_data)/sample_rate:.2f}s")
        
        try:
            # Prepare input
            inputs = self.processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            
            # Move to device
            if not (self.use_8bit or self.use_4bit):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=448,
                    language=language
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            self.logger.info(f"Transcription: {transcription[:100]}...")
            return transcription
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            raise
    
    def transcribe_batch(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000,
        language: str = "zh"
    ) -> List[str]:
        """
        Transcribe multiple audio chunks
        
        Args:
            audio_chunks: List of audio data arrays
            sample_rate: Sample rate of audio
            language: Language code
            
        Returns:
            List of transcribed texts
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Transcribing {len(audio_chunks)} audio chunks")
        
        transcriptions = []
        for i, audio in enumerate(audio_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
            transcription = self.transcribe(audio, sample_rate, language)
            transcriptions.append(transcription)
        
        return transcriptions
    
    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if config.ENABLE_MODEL_OFFLOADING:
            self.unload_model()
