#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transfilm - 生产级AI视频配音系统

使用Qwen3-ASR、Qwen3-TTS和MiniCPM-o实现完整的视频翻译和配音管道
"""

__version__ = "1.0.0"
__author__ = "Olivia-lrh"

from transfilm.config import Config
from transfilm.pipeline import Pipeline

__all__ = [
    "Config",
    "Pipeline",
    "__version__",
]
