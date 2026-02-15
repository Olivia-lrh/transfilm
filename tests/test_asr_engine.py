#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASR引擎测试
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from transfilm.asr_engine import ASREngine, ASRResult


def test_asr_engine_init():
    """测试ASR引擎初始化"""
    engine = ASREngine(
        model_name="Qwen/Qwen3-ASR-1.7B",
        device="cuda:0",
        dtype="bfloat16",
    )
    
    assert engine.model_name == "Qwen/Qwen3-ASR-1.7B"
    assert engine.device == "cuda:0"
    assert engine.model is None  # 初始化时模型未加载


def test_asr_result():
    """测试ASR结果对象"""
    result = ASRResult(
        language="Chinese",
        text="测试文本",
        time_stamps=[],
    )
    
    assert result.language == "Chinese"
    assert result.text == "测试文本"
    assert result.time_stamps == []
    
    # 测试转换为字典
    result_dict = result.to_dict()
    assert result_dict["language"] == "Chinese"
    assert result_dict["text"] == "测试文本"


def test_asr_engine_load_model():
    """测试ASR引擎模型加载（跳过实际加载）"""
    engine = ASREngine()
    
    # 测试模型初始状态
    assert engine.model is None
    
    # 注意：实际加载需要qwen_asr包，这里只测试接口
    # 实际使用时需要安装：pip install qwen-asr


def test_asr_engine_unload_model():
    """测试ASR引擎模型卸载"""
    engine = ASREngine()
    engine.model = Mock()  # 设置一个mock模型
    
    engine.unload_model()
    
    assert engine.model is None
