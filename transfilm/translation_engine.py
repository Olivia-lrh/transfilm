#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
翻译引擎 - 使用MiniCPM-o官方API
"""

import logging
from typing import Optional, List
import torch

logger = logging.getLogger(__name__)


class TranslationEngine:
    """翻译引擎 - 基于MiniCPM-o"""
    
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-o-2_6",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 2048,
        sampling: bool = False,
    ):
        """初始化翻译引擎
        
        Args:
            model_name: 模型名称
            device: 计算设备
            dtype: 数据类型
            attn_implementation: 注意力实现方式
            max_new_tokens: 最大生成token数
            sampling: 是否使用采样
        """
        self.model_name = model_name
        self.device = device
        self.dtype = self._get_torch_dtype(dtype)
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens
        self.sampling = sampling
        
        self.model = None
        self.tokenizer = None
        logger.info(f"翻译引擎初始化: {model_name}")
    
    def _get_torch_dtype(self, dtype_str: str):
        """获取torch数据类型"""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str.lower(), torch.bfloat16)
    
    def load_model(self):
        """加载翻译模型"""
        if self.model is not None:
            logger.warning("模型已加载")
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"正在加载翻译模型: {self.model_name}")
            
            # 使用官方API加载模型
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.dtype,
            )
            self.model = self.model.eval().cuda()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            logger.info("翻译模型加载成功")
            
        except ImportError as e:
            logger.error("请安装 transformers: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"加载翻译模型失败: {e}")
            raise
    
    def unload_model(self):
        """卸载模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("翻译模型已卸载")
    
    def translate(
        self,
        text: str,
        target_language: str = "Chinese",
        source_language: Optional[str] = None,
    ) -> str:
        """翻译文本
        
        Args:
            text: 待翻译文本
            target_language: 目标语言
            source_language: 源语言（可选）
        
        Returns:
            翻译后的文本
        """
        if self.model is None:
            self.load_model()
        
        try:
            # 构建提示
            if source_language:
                prompt = f"Translate from {source_language} to {target_language}: {text}"
            else:
                prompt = f"Translate to {target_language}: {text}"
            
            logger.debug(f"翻译提示: {prompt}")
            
            # 使用官方API进行翻译
            msgs = [{'role': 'user', 'content': prompt}]
            
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                sampling=self.sampling,
            )
            
            logger.debug(f"翻译结果: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"翻译失败: {e}")
            raise
    
    def translate_batch(
        self,
        texts: List[str],
        target_language: str = "Chinese",
        source_language: Optional[str] = None,
    ) -> List[str]:
        """批量翻译文本
        
        Args:
            texts: 待翻译文本列表
            target_language: 目标语言
            source_language: 源语言（可选）
        
        Returns:
            翻译后的文本列表
        """
        if not texts:
            return []
        
        logger.info(f"批量翻译 {len(texts)} 段文本")
        
        results = []
        for i, text in enumerate(texts):
            try:
                translated = self.translate(text, target_language, source_language)
                results.append(translated)
                logger.debug(f"翻译进度: {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"翻译失败 (段落 {i}): {e}")
                # 失败时保留原文
                results.append(text)
        
        logger.info(f"批量翻译完成")
        return results
