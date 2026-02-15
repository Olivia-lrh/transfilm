#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块
支持YAML配置文件、环境变量和命令行参数
"""

import os
import yaml
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """配置管理类
    
    优先级: 命令行参数 > 环境变量 > YAML配置文件 > 默认值
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置
        
        Args:
            config_path: YAML配置文件路径，如果为None则使用默认配置
        """
        self._config = {}
        self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_yaml_config(config_path)
        else:
            # 尝试加载默认配置文件
            default_paths = [
                "config.yaml",
                "config.yml",
                os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self._load_yaml_config(path)
                    break
        
        self._load_env_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        self._config = {
            "models": {
                "asr": {
                    "model_name": "Qwen/Qwen3-ASR-1.7B",
                    "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "max_inference_batch_size": 32,
                    "max_new_tokens": 256,
                },
                "tts": {
                    "mode": "custom_voice",
                    "custom_voice_model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "voice_clone_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                    "default_speaker": "Vivian",
                    "default_instruct": "",
                    "x_vector_only_mode": False,
                },
                "translation": {
                    "model_name": "Qwen/Qwen3-4B",
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "max_new_tokens": 2048,
                    "sampling": False,
                },
            },
            "download": {
                "source": "huggingface",
                "cache_dir": "",
                "show_progress": True,
                "resume_download": True,
            },
            "pipeline": {
                "cache_dir": "./cache",
                "enable_cache": True,
                "sequential_model_loading": True,
                "clear_cache_between_stages": True,
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "time_stretch_method": "librosa",
            },
            "video": {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "crf": 23,
                "audio_bitrate": "192k",
                "supported_formats": ["mp4", "mkv", "avi", "mov", "webm"],
            },
            "languages": {
                "asr_languages": ["Chinese", "English", "Japanese", "Korean"],
                "tts_languages": ["Chinese", "English"],
                "default_source": None,
                "default_target": "Chinese",
            },
            "speakers": {
                "custom_voice": [
                    {"name": "Vivian", "description": "女声-活泼", "language": "Chinese"},
                    {"name": "Serena", "description": "女声-温柔", "language": "Chinese"},
                    {"name": "Ryan", "description": "男声-沉稳", "language": "Chinese"},
                    {"name": "Emma", "description": "女声-优雅", "language": "English"},
                    {"name": "Jack", "description": "男声-阳光", "language": "English"},
                ],
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "",
            },
            "webui": {
                "host": "127.0.0.1",
                "port": 7860,
                "share": False,
                "debug": False,
            },
        }
    
    def _load_yaml_config(self, config_path: str):
        """从YAML文件加载配置
        
        Args:
            config_path: YAML配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self._merge_config(self._config, yaml_config)
                    logger.info(f"已加载配置文件: {config_path}")
        except Exception as e:
            logger.warning(f"加载配置文件失败 {config_path}: {e}")
    
    def _load_env_config(self):
        """从环境变量加载配置
        
        环境变量格式: TRANSFILM_MODELS_ASR_DEVICE=cuda:0
        """
        prefix = "TRANSFILM_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().split('_')
                self._set_nested_value(self._config, config_key, value)
    
    def _merge_config(self, base: Dict, override: Dict):
        """递归合并配置字典
        
        Args:
            base: 基础配置
            override: 覆盖配置
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, config: Dict, keys: list, value: Any):
        """设置嵌套字典的值
        
        Args:
            config: 配置字典
            keys: 键列表
            value: 要设置的值
        """
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 尝试类型转换
        final_key = keys[-1]
        if final_key in config:
            original_type = type(config[final_key])
            try:
                if original_type == bool:
                    value = value.lower() in ('true', '1', 'yes')
                elif original_type == int:
                    value = int(value)
                elif original_type == float:
                    value = float(value)
            except (ValueError, AttributeError):
                pass
        
        config[final_key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key_path: 配置键路径，用点号分隔 (例如: "models.asr.device")
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """设置配置值
        
        Args:
            key_path: 配置键路径，用点号分隔
            value: 要设置的值
        """
        keys = key_path.split('.')
        self._set_nested_value(self._config, keys, value)
    
    def get_all(self) -> Dict:
        """获取所有配置
        
        Returns:
            配置字典
        """
        return self._config.copy()
    
    def save(self, config_path: str):
        """保存配置到YAML文件
        
        Args:
            config_path: YAML配置文件路径
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"配置已保存到: {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败 {config_path}: {e}")
            raise
