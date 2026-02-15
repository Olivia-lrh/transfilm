#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载管理器
支持从HuggingFace Hub和ModelScope下载模型
"""

import os
import logging
from typing import Optional, List
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelDownloader:
    """模型下载管理器"""
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "asr": "Qwen/Qwen3-ASR-1.7B",
        "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "tts_custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "tts_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "translation": "Qwen/Qwen3-4B",
        "translation_low_vram": "Qwen/Qwen3-1.7B",
        "translation_ultra_low_vram": "Qwen/Qwen3-0.6B",
    }
    
    def __init__(
        self,
        source: str = "huggingface",
        cache_dir: Optional[str] = None,
        show_progress: bool = True,
    ):
        """初始化下载器
        
        Args:
            source: 下载源 (huggingface 或 modelscope)
            cache_dir: 本地缓存目录
            show_progress: 是否显示进度条
        """
        self.source = source.lower()
        self.cache_dir = cache_dir
        self.show_progress = show_progress
        
        if self.source not in ["huggingface", "modelscope"]:
            raise ValueError(f"不支持的下载源: {source}")
    
    def download_model(
        self,
        model_name: str,
        local_dir: Optional[str] = None,
        resume_download: bool = True,
    ) -> str:
        """下载单个模型
        
        Args:
            model_name: 模型名称或HuggingFace repo ID
            local_dir: 本地保存目录
            resume_download: 是否支持断点续传
        
        Returns:
            模型本地路径
        """
        logger.info(f"开始下载模型: {model_name}")
        
        if self.source == "huggingface":
            return self._download_from_huggingface(
                model_name, local_dir, resume_download
            )
        elif self.source == "modelscope":
            return self._download_from_modelscope(
                model_name, local_dir, resume_download
            )
        else:
            raise ValueError(f"不支持的下载源: {self.source}")
    
    def _download_from_huggingface(
        self,
        repo_id: str,
        local_dir: Optional[str],
        resume_download: bool,
    ) -> str:
        """从HuggingFace Hub下载模型
        
        Args:
            repo_id: HuggingFace仓库ID
            local_dir: 本地保存目录
            resume_download: 是否支持断点续传
        
        Returns:
            模型本地路径
        """
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"从HuggingFace Hub下载: {repo_id}")
            
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=self.cache_dir,
                local_dir=local_dir,
                resume_download=resume_download,
                local_dir_use_symlinks=False,
            )
            
            logger.info(f"模型下载完成: {model_path}")
            return model_path
            
        except ImportError:
            logger.error("请安装 huggingface_hub: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"下载模型失败 {repo_id}: {e}")
            raise
    
    def _download_from_modelscope(
        self,
        model_id: str,
        local_dir: Optional[str],
        resume_download: bool,
    ) -> str:
        """从ModelScope下载模型
        
        Args:
            model_id: ModelScope模型ID
            local_dir: 本地保存目录
            resume_download: 是否支持断点续传
        
        Returns:
            模型本地路径
        """
        try:
            from modelscope import snapshot_download as ms_snapshot_download
            
            logger.info(f"从ModelScope下载: {model_id}")
            
            # ModelScope的repo_id格式可能不同，需要转换
            # Qwen/Qwen3-ASR-1.7B -> qwen/Qwen3-ASR-1.7B
            model_id_ms = model_id.replace("/", "/").lower() if "/" in model_id else model_id
            
            kwargs = {
                "model_id": model_id_ms,
            }
            
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            
            model_path = ms_snapshot_download(**kwargs)
            
            logger.info(f"模型下载完成: {model_path}")
            return model_path
            
        except ImportError:
            logger.error("请安装 modelscope: pip install modelscope")
            raise
        except Exception as e:
            logger.warning(f"从ModelScope下载失败，尝试使用HuggingFace: {e}")
            # 如果ModelScope下载失败，尝试HuggingFace
            return self._download_from_huggingface(model_id, local_dir, resume_download)
    
    def download_all_models(
        self,
        models: Optional[List[str]] = None,
        local_dir: Optional[str] = None,
    ) -> dict:
        """下载所有需要的模型
        
        Args:
            models: 要下载的模型列表，如果为None则下载所有模型
            local_dir: 本地保存目录
        
        Returns:
            模型名称到本地路径的映射
        """
        if models is None:
            models = list(self.SUPPORTED_MODELS.keys())
        
        model_paths = {}
        
        for model_key in models:
            if model_key not in self.SUPPORTED_MODELS:
                logger.warning(f"未知的模型: {model_key}")
                continue
            
            repo_id = self.SUPPORTED_MODELS[model_key]
            try:
                model_path = self.download_model(
                    repo_id,
                    local_dir=local_dir,
                    resume_download=True,
                )
                model_paths[model_key] = model_path
            except Exception as e:
                logger.error(f"下载模型失败 {model_key}: {e}")
                model_paths[model_key] = None
        
        return model_paths
    
    def check_model_exists(self, model_name: str) -> bool:
        """检查模型是否已下载
        
        Args:
            model_name: 模型名称或路径
        
        Returns:
            是否存在
        """
        # 如果是本地路径
        if os.path.exists(model_name):
            return True
        
        # 检查缓存目录
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, model_name.replace("/", "--"))
            if os.path.exists(cache_path):
                return True
        
        return False
    
    @staticmethod
    def get_model_info(model_key: str) -> Optional[str]:
        """获取模型信息
        
        Args:
            model_key: 模型键名
        
        Returns:
            模型repo ID或None
        """
        return ModelDownloader.SUPPORTED_MODELS.get(model_key)
