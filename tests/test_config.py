#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块测试
"""

import os
import tempfile
import pytest
from transfilm.config import Config


def test_config_init():
    """测试配置初始化"""
    config = Config()
    assert config is not None
    assert config.get("models.asr.model_name") == "Qwen/Qwen3-ASR-1.7B"


def test_config_get():
    """测试获取配置值"""
    config = Config()
    
    # 测试获取存在的配置
    asr_device = config.get("models.asr.device")
    assert asr_device is not None
    
    # 测试获取不存在的配置（返回默认值）
    value = config.get("nonexistent.key", "default")
    assert value == "default"


def test_config_set():
    """测试设置配置值"""
    config = Config()
    
    # 设置新值
    config.set("models.asr.device", "cuda:1")
    assert config.get("models.asr.device") == "cuda:1"
    
    # 设置嵌套值
    config.set("new.nested.key", "value")
    assert config.get("new.nested.key") == "value"


def test_config_load_yaml():
    """测试加载YAML配置"""
    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
models:
  asr:
    device: "cpu"
  tts:
    mode: "voice_clone"
        """)
        temp_config_path = f.name
    
    try:
        config = Config(temp_config_path)
        assert config.get("models.asr.device") == "cpu"
        assert config.get("models.tts.mode") == "voice_clone"
    finally:
        os.unlink(temp_config_path)


def test_config_env_override():
    """测试环境变量覆盖"""
    # 设置环境变量
    os.environ["TRANSFILM_MODELS_ASR_DEVICE"] = "cuda:2"
    
    config = Config()
    assert config.get("models.asr.device") == "cuda:2"
    
    # 清理
    del os.environ["TRANSFILM_MODELS_ASR_DEVICE"]


def test_config_save():
    """测试保存配置"""
    config = Config()
    config.set("test.key", "test_value")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save(temp_path)
        assert os.path.exists(temp_path)
        
        # 重新加载并验证
        new_config = Config(temp_path)
        assert new_config.get("test.key") == "test_value"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_config_get_all():
    """测试获取所有配置"""
    config = Config()
    all_config = config.get_all()
    
    assert isinstance(all_config, dict)
    assert "models" in all_config
    assert "pipeline" in all_config
