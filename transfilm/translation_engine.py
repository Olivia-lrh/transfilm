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
    
    def translate_with_length_control(
        self,
        text: str,
        target_language: str,
        target_duration: float,
        source_language: Optional[str] = None,
        speaking_rate: Optional[float] = None,
        tolerance: float = 0.2,
        max_rounds: int = 2,
    ) -> str:
        """使用字符数控制的翻译（两遍法）
        
        Args:
            text: 待翻译文本
            target_language: 目标语言
            target_duration: 目标时长（秒）
            source_language: 源语言（可选）
            speaking_rate: 目标语言的语速（字符/秒），如果为None则自动设置
            tolerance: 字符数偏差容忍度（默认20%）
            max_rounds: 最大优化轮数（默认2轮）
        
        Returns:
            翻译后的文本
        """
        # 自动设置语速（字符/秒）
        if speaking_rate is None:
            speaking_rate = self._get_default_speaking_rate(target_language)
        
        # 计算目标字符数
        target_char_count = int(target_duration * speaking_rate)
        
        logger.debug(f"翻译长度控制: 目标时长={target_duration:.2f}s, 语速={speaking_rate:.1f}字符/秒, 目标字符数={target_char_count}")
        
        # Pass 1: 常规翻译
        translated = self.translate(text, target_language, source_language)
        actual_char_count = len(translated)
        
        # 检查偏差
        deviation = abs(actual_char_count - target_char_count) / max(target_char_count, 1)
        logger.debug(f"Pass 1: 实际字符数={actual_char_count}, 偏差={deviation:.1%}")
        
        if deviation <= tolerance:
            # 在容忍范围内，直接返回
            return translated
        
        # Pass 2+: 长度优化
        for round_idx in range(max_rounds):
            if actual_char_count > target_char_count * (1 + tolerance):
                # 太长，需要压缩
                prompt = (
                    f"Rewrite this translation to be approximately {target_char_count} characters "
                    f"while keeping the core meaning. Current length is {actual_char_count} characters.\n\n"
                    f"Text: {translated}\n\n"
                    f"Rewritten version:"
                )
            elif actual_char_count < target_char_count * (1 - tolerance):
                # 太短，需要扩展
                prompt = (
                    f"Expand this translation to be approximately {target_char_count} characters "
                    f"while keeping it natural. Current length is {actual_char_count} characters.\n\n"
                    f"Text: {translated}\n\n"
                    f"Expanded version:"
                )
            else:
                # 已经在容忍范围内
                break
            
            logger.debug(f"Pass {round_idx + 2}: 尝试优化长度...")
            
            try:
                # 调用LLM优化长度
                msgs = [{'role': 'user', 'content': prompt}]
                refined = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    sampling=self.sampling,
                )
                
                new_char_count = len(refined)
                new_deviation = abs(new_char_count - target_char_count) / max(target_char_count, 1)
                
                logger.debug(f"Pass {round_idx + 2}: 新字符数={new_char_count}, 新偏差={new_deviation:.1%}")
                
                # 如果更好，则采用新版本
                if new_deviation < deviation:
                    translated = refined
                    actual_char_count = new_char_count
                    deviation = new_deviation
                else:
                    # 没有改善，停止优化
                    break
            
            except Exception as e:
                logger.warning(f"长度优化失败 (round {round_idx + 2}): {e}")
                break
        
        logger.info(f"翻译完成（长度控制）: 最终字符数={len(translated)}, 目标={target_char_count}, 偏差={deviation:.1%}")
        return translated
    
    def _get_default_speaking_rate(self, language: str) -> float:
        """获取默认语速（字符/秒）
        
        Args:
            language: 语言名称
        
        Returns:
            语速（字符/秒）
        """
        # 默认语速映射
        speaking_rates = {
            "Chinese": 4.0,      # 中文约4字/秒
            "English": 12.0,     # 英文约12字符/秒
            "Japanese": 6.0,     # 日文约6字符/秒
            "Korean": 5.0,       # 韩文约5字符/秒
            "Spanish": 10.0,     # 西班牙语约10字符/秒
            "French": 10.0,      # 法语约10字符/秒
            "German": 9.0,       # 德语约9字符/秒
        }
        
        return speaking_rates.get(language, 8.0)  # 默认8字符/秒
