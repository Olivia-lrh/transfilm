#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
管道测试
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from transfilm.pipeline import Pipeline
from transfilm.config import Config


def test_pipeline_init():
    """测试管道初始化"""
    config = Config()
    pipeline = Pipeline(config)
    
    assert pipeline.config is not None
    assert pipeline.asr_engine is not None
    assert pipeline.translation_engine is not None
    assert pipeline.tts_engine is not None
    assert pipeline.audio_processor is not None
    assert pipeline.video_processor is not None
    assert pipeline.vad_engine is not None
    assert pipeline.speaker_diarization is not None
    assert pipeline.subtitle_generator is not None


def test_pipeline_init_without_config():
    """测试管道初始化（无配置）"""
    pipeline = Pipeline()
    
    assert pipeline.config is not None
    assert pipeline.current_stage == 0
    assert pipeline.total_stages == 8  # 从6个阶段升级到8个阶段


def test_pipeline_components():
    """测试管道组件初始化"""
    pipeline = Pipeline()
    
    # 验证所有组件都已初始化
    assert pipeline.asr_engine is not None
    assert pipeline.translation_engine is not None
    assert pipeline.tts_engine is not None
    assert pipeline.audio_processor is not None
    assert pipeline.video_processor is not None
    # 新组件
    assert pipeline.vad_engine is not None
    assert pipeline.speaker_diarization is not None
    assert pipeline.subtitle_generator is not None


def test_pipeline_progress_callback():
    """测试进度回调"""
    pipeline = Pipeline()
    
    # Mock回调函数
    callback = Mock()
    
    # 测试进度更新
    pipeline._update_progress(1, "测试阶段", callback)
    
    # 验证回调被调用
    callback.assert_called_once_with(1, 8, "测试阶段")
    assert pipeline.current_stage == 1


def test_pipeline_vad_disabled():
    """测试VAD禁用时的初始化"""
    config = Config()
    config.set("vad.enabled", False)
    
    pipeline = Pipeline(config)
    
    # VAD应该是None
    assert pipeline.vad_engine is None


def test_pipeline_speaker_diarization_disabled():
    """测试说话人分离禁用时的初始化"""
    config = Config()
    config.set("speaker_diarization.enabled", False)
    
    pipeline = Pipeline(config)
    
    # 说话人分离应该是None
    assert pipeline.speaker_diarization is None


def test_pipeline_subtitle_disabled():
    """测试字幕生成禁用时的初始化"""
    config = Config()
    config.set("subtitle.enabled", False)
    
    pipeline = Pipeline(config)
    
    # 字幕生成器应该是None
    assert pipeline.subtitle_generator is None
    
    # Mock回调函数
    callback = Mock()
    
    # 调用内部方法
    pipeline._update_progress(1, "测试消息", callback)
    
    # 验证回调被调用
    callback.assert_called_once_with(1, 6, "测试消息")
    assert pipeline.current_stage == 1
