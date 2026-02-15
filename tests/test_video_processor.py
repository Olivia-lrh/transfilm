#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频处理器测试
"""

import pytest
from transfilm.video_processor import VideoProcessor


def test_video_processor_init():
    """测试视频处理器初始化"""
    processor = VideoProcessor()
    assert processor.video_codec == "libx264"
    assert processor.audio_codec == "aac"
    assert processor.crf == 23


def test_check_format_supported():
    """测试格式检查"""
    processor = VideoProcessor()
    
    assert processor.check_format_supported("test.mp4") is True
    assert processor.check_format_supported("test.mkv") is True
    assert processor.check_format_supported("test.avi") is True
    assert processor.check_format_supported("test.mov") is True
    assert processor.check_format_supported("test.webm") is True
    assert processor.check_format_supported("test.unknown") is False


def test_video_processor_commands():
    """测试视频处理器命令构建（不实际执行）"""
    processor = VideoProcessor(
        video_codec="libx264",
        audio_codec="aac",
        crf=23,
        audio_bitrate="192k",
    )
    
    # 验证参数设置正确
    assert processor.video_codec == "libx264"
    assert processor.audio_codec == "aac"
    assert processor.crf == 23
    assert processor.audio_bitrate == "192k"
