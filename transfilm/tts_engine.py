#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS引擎 - 使用Qwen3-TTS官方API
"""

import os
import logging
from typing import Optional, List, Tuple, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    """TTS引擎 - 基于Qwen3-TTS"""
    
    def __init__(
        self,
        mode: str = "custom_voice",
        custom_voice_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        voice_clone_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        default_speaker: str = "Vivian",
        default_instruct: str = "",
        x_vector_only_mode: bool = False,
    ):
        """初始化TTS引擎
        
        Args:
            mode: TTS模式 (custom_voice 或 voice_clone)
            custom_voice_model: 自定义音色模型名称
            voice_clone_model: 音色克隆模型名称
            device: 计算设备
            dtype: 数据类型
            attn_implementation: 注意力实现方式
            default_speaker: 默认说话人
            default_instruct: 默认语气指令
            x_vector_only_mode: 是否只使用x-vector模式
        """
        self.mode = mode
        self.custom_voice_model = custom_voice_model
        self.voice_clone_model = voice_clone_model
        self.device = device
        self.dtype = self._get_torch_dtype(dtype)
        self.attn_implementation = attn_implementation
        self.default_speaker = default_speaker
        self.default_instruct = default_instruct
        self.x_vector_only_mode = x_vector_only_mode
        
        self.model = None
        self.voice_clone_prompt = None
        logger.info(f"TTS引擎初始化: mode={mode}")
    
    def _get_torch_dtype(self, dtype_str: str):
        """获取torch数据类型"""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str.lower(), torch.bfloat16)
    
    def load_model(self):
        """加载TTS模型"""
        if self.model is not None:
            logger.warning("模型已加载")
            return
        
        try:
            from qwen_tts import Qwen3TTSModel
            
            # 根据模式选择模型
            if self.mode == "custom_voice":
                model_name = self.custom_voice_model
            elif self.mode == "voice_clone":
                model_name = self.voice_clone_model
            else:
                raise ValueError(f"不支持的TTS模式: {self.mode}")
            
            logger.info(f"正在加载TTS模型: {model_name}")
            
            # 使用官方API加载模型
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
            
            logger.info("TTS模型加载成功")
            
        except ImportError as e:
            logger.error("请安装 qwen-tts: pip install qwen-tts")
            raise
        except Exception as e:
            logger.error(f"加载TTS模型失败: {e}")
            raise
    
    def unload_model(self):
        """卸载模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.voice_clone_prompt is not None:
            del self.voice_clone_prompt
            self.voice_clone_prompt = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TTS模型已卸载")
    
    def create_voice_clone_prompt(
        self,
        ref_audio: str,
        ref_text: str,
    ) -> Any:
        """创建音色克隆提示
        
        Args:
            ref_audio: 参考音频路径
            ref_text: 参考音频的文本转录
        
        Returns:
            音色克隆提示对象
        """
        if self.model is None:
            self.load_model()
        
        if self.mode != "voice_clone":
            raise ValueError("create_voice_clone_prompt 只能在 voice_clone 模式下使用")
        
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"参考音频不存在: {ref_audio}")
        
        try:
            logger.info(f"创建音色克隆提示: {ref_audio}")
            
            # 使用官方API创建音色克隆提示
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=self.x_vector_only_mode,
            )
            
            self.voice_clone_prompt = prompt_items
            logger.info("音色克隆提示创建成功")
            return prompt_items
            
        except Exception as e:
            logger.error(f"创建音色克隆提示失败: {e}")
            raise
    
    def synthesize_custom_voice(
        self,
        text: str,
        language: str = "Chinese",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """使用自定义音色合成语音
        
        Args:
            text: 要合成的文本
            language: 语言
            speaker: 说话人
            instruct: 语气指令
        
        Returns:
            (音频数组, 采样率)
        """
        if self.model is None:
            self.load_model()
        
        if self.mode != "custom_voice":
            raise ValueError("synthesize_custom_voice 只能在 custom_voice 模式下使用")
        
        try:
            speaker = speaker or self.default_speaker
            instruct = instruct or self.default_instruct
            
            logger.debug(f"合成语音: text='{text[:50]}...', speaker={speaker}")
            
            # 使用官方API合成语音
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
            
            # 转换为numpy数组
            if isinstance(wavs, torch.Tensor):
                wavs = wavs.cpu().numpy()
            
            return wavs, sr
            
        except Exception as e:
            logger.error(f"合成语音失败: {e}")
            raise
    
    def synthesize_voice_clone(
        self,
        text: str,
        language: str = "Chinese",
        voice_clone_prompt: Optional[Any] = None,
    ) -> Tuple[np.ndarray, int]:
        """使用音色克隆合成语音
        
        Args:
            text: 要合成的文本
            language: 语言
            voice_clone_prompt: 音色克隆提示（如果为None则使用之前创建的）
        
        Returns:
            (音频数组, 采样率)
        """
        if self.model is None:
            self.load_model()
        
        if self.mode != "voice_clone":
            raise ValueError("synthesize_voice_clone 只能在 voice_clone 模式下使用")
        
        try:
            # 使用提供的prompt或之前创建的prompt
            prompt = voice_clone_prompt or self.voice_clone_prompt
            if prompt is None:
                raise ValueError("请先使用 create_voice_clone_prompt 创建音色克隆提示")
            
            logger.debug(f"合成语音(克隆): text='{text[:50]}...'")
            
            # 使用官方API合成语音
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
            )
            
            # 转换为numpy数组
            if isinstance(wavs, torch.Tensor):
                wavs = wavs.cpu().numpy()
            
            return wavs, sr
            
        except Exception as e:
            logger.error(f"合成语音失败: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        language: str = "Chinese",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """合成语音（自动根据模式选择方法）
        
        Args:
            text: 要合成的文本
            language: 语言
            **kwargs: 其他参数
        
        Returns:
            (音频数组, 采样率)
        """
        if self.mode == "custom_voice":
            return self.synthesize_custom_voice(
                text,
                language,
                speaker=kwargs.get("speaker"),
                instruct=kwargs.get("instruct"),
            )
        elif self.mode == "voice_clone":
            return self.synthesize_voice_clone(
                text,
                language,
                voice_clone_prompt=kwargs.get("voice_clone_prompt"),
            )
        else:
            raise ValueError(f"不支持的TTS模式: {self.mode}")
    
    def synthesize_batch(
        self,
        texts: List[str],
        language: str = "Chinese",
        **kwargs,
    ) -> List[Tuple[np.ndarray, int]]:
        """批量合成语音
        
        Args:
            texts: 文本列表
            language: 语言
            **kwargs: 其他参数
        
        Returns:
            (音频数组, 采样率) 列表
        """
        if not texts:
            return []
        
        logger.info(f"批量合成 {len(texts)} 段语音")
        
        results = []
        for i, text in enumerate(texts):
            try:
                audio, sr = self.synthesize(text, language, **kwargs)
                results.append((audio, sr))
                logger.debug(f"合成进度: {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"合成失败 (段落 {i}): {e}")
                # 失败时返回空音频
                results.append((np.array([]), 0))
        
        logger.info(f"批量合成完成")
        return results
