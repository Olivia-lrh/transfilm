"""
LLM engine for TransFilm
Provides LLM-based sentence segmentation, speaker identification, and translation adjustment
"""
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import setup_logger
import config

logger = setup_logger(__name__)


class LLMEngine:
    """LLM engine for intelligent text processing"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        dtype: str = None,
        use_8bit: bool = False,
        use_4bit: bool = False
    ):
        """
        Initialize LLM engine
        
        Args:
            model_name: LLM model name or path (defaults to config.DEFAULT_LLM_MODEL)
            device: Device to run on
            dtype: Data type to use
            use_8bit: Use 8-bit quantization
            use_4bit: Use 4-bit quantization
        """
        self.model_name = model_name or getattr(config, 'DEFAULT_LLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
        self.device = device or config.DEFAULT_DEVICE
        self.dtype = dtype or config.DEFAULT_DTYPE
        self.use_8bit = use_8bit or config.USE_8BIT
        self.use_4bit = use_4bit or config.USE_4BIT
        
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self) -> None:
        """Load LLM model and tokenizer"""
        if self.is_loaded:
            self.logger.info("LLM model already loaded")
            return
        
        self.logger.info(f"Loading LLM model: {self.model_name}")
        
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
            
            self.logger.info("LLM model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
            self.logger.warning("LLM features will use fallback methods")
            # Don't raise - allow fallback to simpler methods
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_loaded = False
            self.logger.info("LLM model unloaded")
    
    def segment_sentences(
        self,
        text: str,
        speaker_info: Optional[List[Dict]] = None,
        language: str = 'zh'
    ) -> List[Dict]:
        """
        Segment text into natural sentences using LLM
        
        Args:
            text: Full text to segment
            speaker_info: Optional speaker diarization info
            language: Language code
            
        Returns:
            List of sentence segments:
            [
                {
                    'text': 'sentence text',
                    'speaker_id': 'SPEAKER_00',
                    'start_char': 0,
                    'end_char': 20
                },
                ...
            ]
        """
        self.logger.info(f"Segmenting text with LLM ({len(text)} chars)")
        
        if not self.is_loaded:
            # Fallback to simple segmentation
            self.logger.warning("LLM not loaded, using simple segmentation")
            return self._simple_segmentation(text, language)
        
        try:
            # Construct prompt for LLM
            prompt = self._construct_segmentation_prompt(text, speaker_info, language)
            
            # Generate segmentation
            segments = self._generate_with_llm(prompt, task='segmentation')
            
            self.logger.info(f"LLM segmented into {len(segments)} sentences")
            return segments
            
        except Exception as e:
            self.logger.error(f"LLM segmentation failed: {e}, using fallback")
            return self._simple_segmentation(text, language)
    
    def identify_speakers_in_text(
        self,
        text: str,
        audio_speaker_info: List[Dict],
        language: str = 'zh'
    ) -> List[Dict]:
        """
        Use LLM to help identify speakers in text based on context
        
        Args:
            text: Full text
            audio_speaker_info: Speaker diarization from audio analysis
            language: Language code
            
        Returns:
            Enhanced speaker information with LLM insights
        """
        self.logger.info("Using LLM to enhance speaker identification")
        
        if not self.is_loaded:
            self.logger.warning("LLM not loaded, returning audio speaker info")
            return audio_speaker_info
        
        try:
            # Construct prompt
            prompt = self._construct_speaker_identification_prompt(
                text, audio_speaker_info, language
            )
            
            # Get LLM analysis
            speaker_analysis = self._generate_with_llm(prompt, task='speaker_id')
            
            # Merge with audio analysis
            enhanced_info = self._merge_speaker_info(audio_speaker_info, speaker_analysis)
            
            return enhanced_info
            
        except Exception as e:
            self.logger.error(f"LLM speaker identification failed: {e}")
            return audio_speaker_info
    
    def adjust_translation_length(
        self,
        original_text: str,
        translation: str,
        target_duration: float,
        actual_duration: float,
        language: str = 'zh',
        target_language: str = 'en'
    ) -> str:
        """
        Adjust translation to match target duration using LLM
        
        Args:
            original_text: Original text
            translation: Initial translation
            target_duration: Target duration in seconds
            actual_duration: Actual TTS duration in seconds
            language: Source language
            target_language: Target language
            
        Returns:
            Adjusted translation
        """
        self.logger.info(
            f"Adjusting translation length: target={target_duration:.2f}s, "
            f"actual={actual_duration:.2f}s"
        )
        
        # If difference is small, no adjustment needed
        duration_ratio = actual_duration / target_duration if target_duration > 0 else 1.0
        if 0.9 <= duration_ratio <= 1.1:  # Within 10% tolerance
            self.logger.info("Duration within tolerance, no adjustment needed")
            return translation
        
        if not self.is_loaded:
            # Fallback to simple adjustment
            self.logger.warning("LLM not loaded, using simple length adjustment")
            return self._simple_length_adjustment(
                translation, duration_ratio, target_language
            )
        
        try:
            # Construct prompt
            prompt = self._construct_length_adjustment_prompt(
                original_text,
                translation,
                target_duration,
                actual_duration,
                language,
                target_language
            )
            
            # Get adjusted translation
            adjusted = self._generate_with_llm(prompt, task='translation_adjust')
            
            self.logger.info(f"Translation adjusted: {len(translation)} -> {len(adjusted)} chars")
            return adjusted
            
        except Exception as e:
            self.logger.error(f"LLM length adjustment failed: {e}")
            return self._simple_length_adjustment(translation, duration_ratio, target_language)
    
    def _construct_segmentation_prompt(
        self,
        text: str,
        speaker_info: Optional[List[Dict]],
        language: str
    ) -> str:
        """Construct prompt for sentence segmentation"""
        if language == 'zh':
            prompt = f"""请将以下文本分割成最自然的短句，考虑语义完整性和口语表达习惯。
每个短句应该是一个完整的语义单元，适合语音合成。

文本：
{text}

请以JSON格式返回，包含每个句子的text字段。"""
        else:
            prompt = f"""Please segment the following text into natural short sentences, considering semantic completeness and spoken language patterns.
Each sentence should be a complete semantic unit suitable for speech synthesis.

Text:
{text}

Return in JSON format with text field for each sentence."""
        
        return prompt
    
    def _construct_speaker_identification_prompt(
        self,
        text: str,
        audio_speaker_info: List[Dict],
        language: str
    ) -> str:
        """Construct prompt for speaker identification assistance"""
        speaker_count = len(set(s['speaker_id'] for s in audio_speaker_info))
        
        if language == 'zh':
            prompt = f"""音频分析显示有{speaker_count}个说话人。
请根据文本内容和语义，帮助确认说话人的身份和对话结构。

文本：
{text}

音频说话人时间段：
{audio_speaker_info[:5]}  # Sample

请分析可能的说话人特征和对话结构。"""
        else:
            prompt = f"""Audio analysis detected {speaker_count} speakers.
Please help identify speakers and conversation structure based on text content and semantics.

Text:
{text}

Audio speaker timeline:
{audio_speaker_info[:5]}  # Sample

Please analyze potential speaker characteristics and dialogue structure."""
        
        return prompt
    
    def _construct_length_adjustment_prompt(
        self,
        original: str,
        translation: str,
        target_duration: float,
        actual_duration: float,
        language: str,
        target_language: str
    ) -> str:
        """Construct prompt for translation length adjustment"""
        duration_ratio = actual_duration / target_duration if target_duration > 0 else 1.0
        
        if duration_ratio > 1.1:
            # Translation is too long, need to shorten
            action = "缩短" if language == 'zh' else "shorten"
        else:
            # Translation is too short, need to expand
            action = "扩展" if language == 'zh' else "expand"
        
        if language == 'zh':
            prompt = f"""请调整以下翻译，使其朗读时长更接近目标时长。

原文（{language}）：{original}
当前翻译（{target_language}）：{translation}
目标时长：{target_duration:.2f}秒
实际时长：{actual_duration:.2f}秒
需要：{action}翻译

请提供调整后的翻译，保持语义准确性和自然度。"""
        else:
            prompt = f"""Please adjust the following translation to better match the target duration when spoken.

Original ({language}): {original}
Current translation ({target_language}): {translation}
Target duration: {target_duration:.2f}s
Actual duration: {actual_duration:.2f}s
Need to: {action}

Provide adjusted translation maintaining semantic accuracy and naturalness."""
        
        return prompt
    
    def _generate_with_llm(self, prompt: str, task: str) -> Any:
        """Generate response using LLM"""
        try:
            # Encode prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if not (self.use_8bit or self.use_4bit):
                inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512 if task == 'translation_adjust' else 1024,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response based on task
            if task == 'segmentation':
                return self._parse_segmentation_response(response)
            elif task == 'speaker_id':
                return self._parse_speaker_response(response)
            elif task == 'translation_adjust':
                return self._parse_translation_response(response)
            else:
                return response
                
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise
    
    def _parse_segmentation_response(self, response: str) -> List[Dict]:
        """Parse LLM segmentation response"""
        # This is a simplified parser
        # In production, use more robust JSON parsing
        segments = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('['):
                segments.append({
                    'text': line,
                    'speaker_id': 'UNKNOWN',
                    'start_char': 0,
                    'end_char': len(line)
                })
        
        return segments if segments else [{'text': response, 'speaker_id': 'UNKNOWN', 'start_char': 0, 'end_char': len(response)}]
    
    def _parse_speaker_response(self, response: str) -> List[Dict]:
        """Parse LLM speaker identification response"""
        # Simplified parser
        return [{'analysis': response}]
    
    def _parse_translation_response(self, response: str) -> str:
        """Parse LLM translation adjustment response"""
        # Extract adjusted translation from response
        # Look for the adjusted translation in the response
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not any(keyword in line.lower() for keyword in ['original', 'current', 'target', 'actual', 'adjust']):
                return line
        
        # Fallback: return last non-empty line
        return lines[-1] if lines else response
    
    def _simple_segmentation(self, text: str, language: str) -> List[Dict]:
        """Fallback: Simple rule-based segmentation"""
        self.logger.info("Using simple rule-based segmentation")
        
        if language == 'zh':
            # Chinese: split by common punctuation
            import re
            sentences = re.split(r'[。！？；]', text)
        else:
            # English: split by period, exclamation, question mark
            import re
            sentences = re.split(r'[.!?]', text)
        
        segments = []
        char_pos = 0
        
        for sent in sentences:
            sent = sent.strip()
            if sent:
                segments.append({
                    'text': sent,
                    'speaker_id': 'UNKNOWN',
                    'start_char': char_pos,
                    'end_char': char_pos + len(sent)
                })
                char_pos += len(sent) + 1  # +1 for delimiter
        
        return segments
    
    def _simple_length_adjustment(
        self,
        translation: str,
        duration_ratio: float,
        target_language: str
    ) -> str:
        """Fallback: Simple length adjustment by truncation or padding"""
        if duration_ratio > 1.1:
            # Too long: truncate
            # Remove filler words or shorten
            words = translation.split()
            target_length = int(len(words) / duration_ratio)
            adjusted = ' '.join(words[:target_length])
            return adjusted
        elif duration_ratio < 0.9:
            # Too short: might need slight expansion
            # This is harder without LLM, so just return original
            return translation
        else:
            return translation
    
    def _merge_speaker_info(
        self,
        audio_info: List[Dict],
        llm_analysis: List[Dict]
    ) -> List[Dict]:
        """Merge audio speaker info with LLM analysis"""
        # For now, just return audio info
        # In production, use LLM analysis to enhance speaker labels
        return audio_info
