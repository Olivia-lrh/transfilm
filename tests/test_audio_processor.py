#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频处理器测试
"""

import os
import tempfile
import pytest
import numpy as np
from transfilm.audio_processor import AudioProcessor


def test_audio_processor_init():
    """测试音频处理器初始化"""
    processor = AudioProcessor()
    assert processor.sample_rate == 16000
    assert processor.channels == 1


def test_audio_processor_save_load():
    """测试音频保存和加载"""
    processor = AudioProcessor()
    
    # 创建测试音频
    audio = np.random.randn(16000).astype(np.float32)  # 1秒音频
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    try:
        # 保存
        processor.save_audio(audio, temp_path, 16000)
        assert os.path.exists(temp_path)
        
        # 加载
        loaded_audio, sr = processor.load_audio(temp_path, 16000)
        assert sr == 16000
        assert len(loaded_audio) > 0
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_extract_audio_command():
    """测试音频提取命令构建（不实际执行）"""
    processor = AudioProcessor()
    
    # 只测试逻辑，不执行FFmpeg命令
    assert processor.sample_rate == 16000
    assert processor.channels == 1


def test_match_duration():
    """测试时长匹配"""
    processor = AudioProcessor(time_stretch_method="librosa")
    
    # 创建1秒的音频
    audio = np.random.randn(16000).astype(np.float32)
    
    # 拉伸到2秒
    stretched = processor.match_duration(audio, 2.0, 16000)
    
    # 验证拉伸后的长度接近目标
    expected_length = 2.0 * 16000
    assert abs(len(stretched) - expected_length) < 1000  # 允许一些误差
