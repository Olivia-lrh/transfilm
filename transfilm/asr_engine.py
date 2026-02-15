#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASR引擎 - 使用Qwen3-ASR官方API
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
import torch
import numpy as np

logger = logging.getLogger(__name__)


class ASRResult:
    """ASR识别结果"""
    
    def __init__(self, language: str, text: str, time_stamps: Optional[List] = None):
        self.language = language
        self.text = text
        self.time_stamps = time_stamps or []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "language": self.language,
            "text": self.text,
            "time_stamps": [
                {
                    "text": ts.text if hasattr(ts, 'text') else str(ts),
                    "start_time": ts.start_time if hasattr(ts, 'start_time') else 0,
                    "end_time": ts.end_time if hasattr(ts, 'end_time') else 0,
                }
                for ts in self.time_stamps
            ],
        }


class ASREngine:
    """ASR引擎 - 基于Qwen3-ASR"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        forced_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
    ):
        """初始化ASR引擎
        
        Args:
            model_name: ASR模型名称
            forced_aligner: 强制对齐器模型名称
            device: 计算设备
            dtype: 数据类型
            max_inference_batch_size: 最大推理批次大小
            max_new_tokens: 最大生成token数
        """
        self.model_name = model_name
        self.forced_aligner = forced_aligner
        self.device = device
        self.dtype = self._get_torch_dtype(dtype)
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        logger.info(f"ASR引擎初始化: {model_name}")
    
    def _get_torch_dtype(self, dtype_str: str):
        """获取torch数据类型"""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str.lower(), torch.bfloat16)
    
    def load_model(self):
        """加载ASR模型"""
        if self.model is not None:
            logger.warning("模型已加载")
            return
        
        try:
            from qwen_asr import Qwen3ASRModel
            
            logger.info(f"正在加载ASR模型: {self.model_name}")
            
            # 使用官方API加载模型
            self.model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map=self.device,
                forced_aligner=self.forced_aligner,
                forced_aligner_kwargs=dict(
                    dtype=self.dtype,
                    device_map=self.device,
                ),
                max_inference_batch_size=self.max_inference_batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            
            logger.info("ASR模型加载成功")
            
        except ImportError as e:
            logger.error("请安装 qwen-asr: pip install qwen-asr")
            raise
        except Exception as e:
            logger.error(f"加载ASR模型失败: {e}")
            raise
    
    def unload_model(self):
        """卸载模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ASR模型已卸载")
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, tuple],
        language: Optional[str] = None,
        return_time_stamps: bool = True,
    ) -> List[ASRResult]:
        """转录音频
        
        Args:
            audio: 音频输入，可以是:
                   - 文件路径 (str)
                   - URL (str)
                   - base64编码 (str)
                   - (numpy数组, 采样率) 元组
            language: 语言，如果为None则自动检测
            return_time_stamps: 是否返回时间戳
        
        Returns:
            ASR结果列表
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"开始转录音频: {audio if isinstance(audio, str) else 'array'}")
            
            # 调用官方API
            results = self.model.transcribe(
                audio=audio,
                language=language,
                return_time_stamps=return_time_stamps,
            )
            
            # 转换为ASRResult对象
            asr_results = []
            for r in results:
                asr_result = ASRResult(
                    language=r.language if hasattr(r, 'language') else (language or "unknown"),
                    text=r.text if hasattr(r, 'text') else str(r),
                    time_stamps=r.time_stamps if hasattr(r, 'time_stamps') else [],
                )
                asr_results.append(asr_result)
            
            logger.info(f"转录完成，识别到 {len(asr_results)} 段文本")
            return asr_results
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise
    
    def transcribe_file(
        self,
        audio_file: str,
        language: Optional[str] = None,
    ) -> List[ASRResult]:
        """转录音频文件
        
        Args:
            audio_file: 音频文件路径
            language: 语言，如果为None则自动检测
        
        Returns:
            ASR结果列表
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")
        
        return self.transcribe(
            audio=audio_file,
            language=language,
            return_time_stamps=True,
        )
