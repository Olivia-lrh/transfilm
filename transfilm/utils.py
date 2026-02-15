#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具函数
"""

import os
import logging
from typing import Optional
import torch


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, log_format: Optional[str] = None):
    """设置日志配置
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 日志文件路径，如果为None则只输出到控制台
        log_format: 日志格式字符串
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging_level,
        format=log_format,
        handlers=handlers
    )


def get_device(device: str = "auto") -> str:
    """获取计算设备
    
    Args:
        device: 设备名称 (cuda:0, cpu, auto)
    
    Returns:
        设备字符串
    """
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def get_dtype(dtype_str: str = "bfloat16"):
    """获取torch数据类型
    
    Args:
        dtype_str: 数据类型字符串 (bfloat16, float16, float32)
    
    Returns:
        torch dtype对象
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def format_time(seconds: float) -> str:
    """格式化时间显示
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串 (HH:MM:SS.mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def ensure_dir(path: str):
    """确保目录存在
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """获取文件大小(MB)
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件大小(MB)
    """
    if not os.path.exists(file_path):
        return 0.0
    return os.path.getsize(file_path) / (1024 * 1024)
