#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频处理模块 - 使用FFmpeg
"""

import os
import subprocess
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理器"""
    
    SUPPORTED_FORMATS = ["mp4", "mkv", "avi", "mov", "webm"]
    
    def __init__(
        self,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23,
        audio_bitrate: str = "192k",
    ):
        """初始化视频处理器
        
        Args:
            video_codec: 视频编码器
            audio_codec: 音频编码器
            crf: 视频质量(CRF值，越低质量越高)
            audio_bitrate: 音频比特率
        """
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.crf = crf
        self.audio_bitrate = audio_bitrate
        logger.info(f"视频处理器初始化: codec={video_codec}, crf={crf}")
    
    def check_format_supported(self, video_path: str) -> bool:
        """检查视频格式是否支持
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            是否支持
        """
        ext = os.path.splitext(video_path)[1].lower().lstrip('.')
        return ext in self.SUPPORTED_FORMATS
    
    def extract_video_without_audio(
        self,
        input_video: str,
        output_video: str,
    ) -> str:
        """提取视频轨道（移除音频）
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
        
        Returns:
            输出视频路径
        """
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"视频文件不存在: {input_video}")
        
        logger.info(f"提取视频轨道: {input_video} -> {output_video}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
        
        # 构建FFmpeg命令
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-an",  # 移除音频
            "-vcodec", "copy",  # 复制视频流（不重新编码）
            "-y",  # 覆盖输出文件
            output_video,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"视频轨道提取成功: {output_video}")
            return output_video
        except subprocess.CalledProcessError as e:
            logger.error(f"视频轨道提取失败: {e.stderr}")
            raise
    
    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        sync_offset: float = 0.0,
    ) -> str:
        """合并音频和视频
        
        Args:
            video_path: 视频文件路径（无音频）
            audio_path: 音频文件路径
            output_path: 输出文件路径
            sync_offset: 音频同步偏移（秒）
        
        Returns:
            输出文件路径
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        logger.info(f"合并音视频: {video_path} + {audio_path} -> {output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # 构建FFmpeg命令
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", self.video_codec,  # 视频编码器
            "-crf", str(self.crf),  # 视频质量
            "-c:a", self.audio_codec,  # 音频编码器
            "-b:a", self.audio_bitrate,  # 音频比特率
            "-shortest",  # 使用最短流的长度
        ]
        
        # 添加音频偏移
        if sync_offset != 0.0:
            cmd.extend(["-itsoffset", str(sync_offset)])
        
        cmd.extend(["-y", output_path])  # 覆盖输出文件
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"音视频合并成功: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"音视频合并失败: {e.stderr}")
            raise
    
    def get_video_info(self, video_path: str) -> dict:
        """获取视频信息
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            视频信息字典
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            
            import json
            info = json.loads(result.stdout)
            
            # 提取关键信息
            video_info = {
                "format": info.get("format", {}).get("format_name", "unknown"),
                "duration": float(info.get("format", {}).get("duration", 0)),
                "size": int(info.get("format", {}).get("size", 0)),
                "streams": [],
            }
            
            for stream in info.get("streams", []):
                stream_info = {
                    "type": stream.get("codec_type", "unknown"),
                    "codec": stream.get("codec_name", "unknown"),
                }
                
                if stream["codec_type"] == "video":
                    stream_info.update({
                        "width": stream.get("width", 0),
                        "height": stream.get("height", 0),
                        "fps": eval(stream.get("r_frame_rate", "0/1")),
                    })
                elif stream["codec_type"] == "audio":
                    stream_info.update({
                        "sample_rate": stream.get("sample_rate", 0),
                        "channels": stream.get("channels", 0),
                    })
                
                video_info["streams"].append(stream_info)
            
            logger.debug(f"视频信息: {video_info}")
            return video_info
            
        except subprocess.CalledProcessError as e:
            logger.error(f"获取视频信息失败: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"解析视频信息失败: {e}")
            raise
    
    def check_ffmpeg_available(self) -> bool:
        """检查FFmpeg是否可用
        
        Returns:
            是否可用
        """
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
