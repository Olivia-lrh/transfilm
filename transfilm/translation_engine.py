"""
Translation engine using Qwen3-0.6B
Handles text segmentation and translation with character count matching
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import setup_logger
import config

logger = setup_logger(__name__)


class TranslationEngine:
    """Qwen3-0.6B engine for translation with sentence segmentation"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        dtype: str = None,
        use_8bit: bool = False,
        use_4bit: bool = False
    ):
        """
        Initialize translation engine
        
        Args:
            model_name: Model name or path (defaults to config.DEFAULT_TRANSLATION_MODEL)
            device: Device to run on (defaults to config.DEFAULT_DEVICE)
            dtype: Data type to use (defaults to config.DEFAULT_DTYPE)
            use_8bit: Use 8-bit quantization
            use_4bit: Use 4-bit quantization
        """
        self.model_name = model_name or config.DEFAULT_TRANSLATION_MODEL
        self.device = device or config.DEFAULT_DEVICE
        self.dtype = dtype or config.DEFAULT_DTYPE
        self.use_8bit = use_8bit or config.USE_8BIT
        self.use_4bit = use_4bit or config.USE_4BIT
        
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self) -> None:
        """Load translation model and tokenizer"""
        if self.is_loaded:
            self.logger.info("Translation model already loaded")
            return
        
        self.logger.info(f"Loading translation model: {self.model_name}")
        
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
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Move to device if not quantized
            if not (self.use_8bit or self.use_4bit):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            self.logger.info("Translation model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load translation model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if not self.is_loaded:
            return
        
        self.logger.info("Unloading translation model")
        
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
        
        self.logger.info("Translation model unloaded")
    
    def segment_text(
        self,
        text: str,
        language: str = "zh"
    ) -> List[str]:
        """
        Segment text into natural sentences using shortest speaking method
        
        Args:
            text: Text to segment
            language: Source language
            
        Returns:
            List of segmented sentences
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Segmenting text: {text[:100]}...")
        
        try:
            # Prompt for sentence segmentation
            prompt = f"""Please segment the following text into natural speaking sentences. 
Use the shortest natural breaks for speaking. Return only the segmented sentences, one per line.

Text: {text}

Segmented sentences:"""
            
            # Generate segmentation
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if not (self.use_8bit or self.use_4bit):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=False
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract sentences from result (after the prompt)
            result_text = result.split("Segmented sentences:")[-1].strip()
            sentences = [s.strip() for s in result_text.split('\n') if s.strip()]
            
            self.logger.info(f"Segmented into {len(sentences)} sentences")
            return sentences
            
        except Exception as e:
            self.logger.error(f"Failed to segment text: {e}")
            # Fallback: simple period-based segmentation
            import re
            sentences = re.split(r'[。！？.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def translate_with_character_matching(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en"
    ) -> Tuple[str, float]:
        """
        Translate text while trying to match character count
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Tuple of (translated_text, character_ratio)
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Translating: {text[:50]}...")
        
        try:
            source_char_count = len(text)
            
            # Prompt for translation with character count consideration
            if config.ENABLE_CHARACTER_COUNT_MATCHING:
                prompt = f"""Translate the following text from {source_lang} to {target_lang}. 
Try to keep the translated text length similar to the original ({source_char_count} characters) 
while maintaining natural expression and meaning.

Original text: {text}

Translation:"""
            else:
                prompt = f"""Translate the following text from {source_lang} to {target_lang}.

Original text: {text}

Translation:"""
            
            # Generate translation
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if not (self.use_8bit or self.use_4bit):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=False
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract translation from result
            translation = result.split("Translation:")[-1].strip()
            
            # Calculate character ratio
            translation_char_count = len(translation)
            char_ratio = translation_char_count / source_char_count if source_char_count > 0 else 1.0
            
            self.logger.info(
                f"Translation: {translation[:50]}... "
                f"(chars: {source_char_count} -> {translation_char_count}, ratio: {char_ratio:.2f})"
            )
            
            return translation, char_ratio
            
        except Exception as e:
            self.logger.error(f"Failed to translate text: {e}")
            # Fallback: return original text
            return text, 1.0
    
    def segment_and_translate(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Segment text and translate each segment with character matching
        
        Args:
            text: Text to process
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of dicts with keys: 'original', 'translation', 'char_ratio'
        """
        if not self.is_loaded:
            self.load_model()
        
        self.logger.info(f"Segmenting and translating text: {text[:100]}...")
        
        # Segment text
        sentences = self.segment_text(text, source_lang)
        
        # Translate each segment
        results = []
        for i, sentence in enumerate(sentences):
            self.logger.info(f"Processing segment {i+1}/{len(sentences)}")
            translation, char_ratio = self.translate_with_character_matching(
                sentence, source_lang, target_lang
            )
            
            results.append({
                'original': sentence,
                'translation': translation,
                'char_ratio': char_ratio,
                'original_length': len(sentence),
                'translation_length': len(translation)
            })
        
        # Log overall statistics
        total_original = sum(r['original_length'] for r in results)
        total_translation = sum(r['translation_length'] for r in results)
        overall_ratio = total_translation / total_original if total_original > 0 else 1.0
        
        self.logger.info(
            f"Processed {len(results)} segments. "
            f"Overall char ratio: {overall_ratio:.2f}"
        )
        
        return results
    
    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if config.ENABLE_MODEL_OFFLOADING:
            self.unload_model()
