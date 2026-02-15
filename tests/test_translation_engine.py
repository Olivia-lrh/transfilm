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


@patch('transfilm.translation_engine.AutoModel')
@patch('transfilm.translation_engine.AutoTokenizer')
def test_translation_engine_load_model(mock_tokenizer_class, mock_model_class):
    """测试翻译引擎模型加载"""
    # Mock模型和tokenizer
    mock_model = MagicMock()
    mock_model.eval.return_value.cuda.return_value = mock_model
    mock_model_class.from_pretrained.return_value = mock_model
    
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    
    engine = TranslationEngine()
    engine.load_model()
    
    # 验证from_pretrained被调用
    mock_model_class.from_pretrained.assert_called_once()
    mock_tokenizer_class.from_pretrained.assert_called_once()


def test_translation_engine_unload_model():
    """测试翻译引擎模型卸载"""
    engine = TranslationEngine()
    engine.model = Mock()
    engine.tokenizer = Mock()
    
    engine.unload_model()
    
    assert engine.model is None
    assert engine.tokenizer is None
