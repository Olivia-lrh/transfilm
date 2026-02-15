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


def test_pipeline_init_without_config():
    """测试管道初始化（无配置）"""
    pipeline = Pipeline()
    
    assert pipeline.config is not None
    assert pipeline.current_stage == 0
    assert pipeline.total_stages == 6


def test_pipeline_components():
    """测试管道组件初始化"""
    pipeline = Pipeline()
    
    # 验证所有组件都已初始化
    assert pipeline.asr_engine is not None
    assert pipeline.translation_engine is not None
    assert pipeline.tts_engine is not None
    assert pipeline.audio_processor is not None
    assert pipeline.video_processor is not None


def test_pipeline_progress_callback():
    """测试进度回调"""
    pipeline = Pipeline()
    
    # Mock回调函数
    callback = Mock()
    
    # 调用内部方法
    pipeline._update_progress(1, "测试消息", callback)
    
    # 验证回调被调用
    callback.assert_called_once_with(1, 6, "测试消息")
    assert pipeline.current_stage == 1
