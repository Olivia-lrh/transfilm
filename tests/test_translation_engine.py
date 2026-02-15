#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
翻译引擎测试
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from transfilm.translation_engine import TranslationEngine


def test_translation_engine_init():
    """测试翻译引擎初始化"""
    engine = TranslationEngine(
        model_name="openbmb/MiniCPM-o-2_6",
        device="cuda:0",
        dtype="bfloat16",
    )
    
    assert engine.model_name == "openbmb/MiniCPM-o-2_6"
    assert engine.device == "cuda:0"
    assert engine.model is None  # 初始化时模型未加载
    assert engine.tokenizer is None


def test_translation_engine_load_model():
    """测试翻译引擎模型加载（跳过实际加载）"""
    engine = TranslationEngine()
    
    # 测试模型初始状态
    assert engine.model is None
    assert engine.tokenizer is None
    
    # 注意：实际加载需要transformers包和模型，这里只测试接口
    # 实际使用时需要安装：pip install transformers


def test_translation_engine_unload_model():
    """测试翻译引擎模型卸载"""
    engine = TranslationEngine()
    engine.model = Mock()
    engine.tokenizer = Mock()
    
    engine.unload_model()
    
    assert engine.model is None
    assert engine.tokenizer is None
