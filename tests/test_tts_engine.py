#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS引擎测试
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from transfilm.tts_engine import TTSEngine


def test_tts_engine_init():
    """测试TTS引擎初始化"""
    engine = TTSEngine(
        mode="custom_voice",
        device="cuda:0",
        dtype="bfloat16",
    )
    
    assert engine.mode == "custom_voice"
    assert engine.device == "cuda:0"
    assert engine.model is None  # 初始化时模型未加载


def test_tts_engine_init_voice_clone():
    """测试TTS引擎初始化（音色克隆模式）"""
    engine = TTSEngine(
        mode="voice_clone",
        device="cuda:0",
    )
    
    assert engine.mode == "voice_clone"
    assert engine.voice_clone_prompt is None


def test_tts_engine_load_model():
    """测试TTS引擎模型加载（跳过实际加载）"""
    engine = TTSEngine(mode="custom_voice")
    
    # 测试模型初始状态
    assert engine.model is None
    
    # 注意：实际加载需要qwen_tts包，这里只测试接口
    # 实际使用时需要安装：pip install qwen-tts


def test_tts_engine_unload_model():
    """测试TTS引擎模型卸载"""
    engine = TTSEngine()
    engine.model = Mock()
    engine.voice_clone_prompt = Mock()
    
    engine.unload_model()
    
    assert engine.model is None
    assert engine.voice_clone_prompt is None


def test_tts_engine_mode_validation():
    """测试TTS模式验证"""
    # 有效模式
    engine1 = TTSEngine(mode="custom_voice")
    assert engine1.mode == "custom_voice"
    
    engine2 = TTSEngine(mode="voice_clone")
    assert engine2.mode == "voice_clone"
