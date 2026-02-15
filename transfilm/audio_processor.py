#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频处理模块 - 使用FFmpeg和librosa
"""

import os
import subprocess
import logging
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频处理器"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        time_stretch_method: str = "librosa",
    ):
        """初始化音频处理器
        
        Args:
            sample_rate: 目标采样率
            channels: 目标声道数
            time_stretch_method: 时间拉伸方法 (librosa 或 ffmpeg)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.time_stretch_method = time_stretch_method
        logger.info(f"音频处理器初始化: sr={sample_rate}, channels={channels}")
    
    def extract_audio_from_video(
        self,
        video_path: str,
        output_audio_path: str,
    ) -> str:
        """从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_audio_path: 输出音频文件路径
        
        Returns:
            输出音频文件路径
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        logger.info(f"从视频提取音频: {video_path} -> {output_audio_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)
        
        # 构建FFmpeg命令
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # 不包含视频
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", str(self.sample_rate),  # 采样率
            "-ac", str(self.channels),  # 声道数
            "-y",  # 覆盖输出文件
            output_audio_path,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"音频提取成功: {output_audio_path}")
            return output_audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"音频提取失败: {e.stderr}")
            raise
    
    def load_audio(
        self,
        audio_path: str,
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """加载音频文件
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率（如果为None则使用文件原始采样率）
        
        Returns:
            (音频数组, 采样率)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        try:
            import librosa
            
            target_sr = target_sr or self.sample_rate
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=(self.channels == 1))
            
            logger.debug(f"加载音频: {audio_path}, shape={audio.shape}, sr={sr}")
            return audio, sr
            
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            raise
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: int,
    ) -> str:
        """保存音频到文件
        
        Args:
            audio: 音频数组
            output_path: 输出文件路径
            sample_rate: 采样率
        
        Returns:
            输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        try:
            sf.write(output_path, audio, sample_rate)
            logger.debug(f"保存音频: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            raise
    
    def time_stretch_librosa(
        self,
        audio: np.ndarray,
        rate: float,
    ) -> np.ndarray:
        """使用librosa进行时间拉伸
        
        Args:
            audio: 音频数组
            rate: 拉伸比率 (>1 加速, <1 减速)
        
        Returns:
            拉伸后的音频数组
        """
        try:
            import librosa
            
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            return stretched
        except Exception as e:
            logger.error(f"时间拉伸失败: {e}")
            raise
    
    def time_stretch_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        rate: float,
    ) -> str:
        """使用FFmpeg进行时间拉伸
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            rate: 拉伸比率 (>1 加速, <1 减速)
        
        Returns:
            输出音频路径
        """
        # atempo滤镜的值在0.5到2.0之间
        # 如果rate超出范围，需要链式应用
        tempo = 1.0 / rate
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        filter_chain = []
        while tempo < 0.5:
            filter_chain.append("atempo=0.5")
            tempo *= 2.0
        while tempo > 2.0:
            filter_chain.append("atempo=2.0")
            tempo /= 2.0
        filter_chain.append(f"atempo={tempo}")
        
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-filter:a", ",".join(filter_chain),
            "-y",
            output_path,
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug(f"时间拉伸完成: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"时间拉伸失败: {e.stderr}")
            raise
    
    def match_duration(
        self,
        audio: np.ndarray,
        target_duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        """调整音频时长以匹配目标时长
        
        Args:
            audio: 音频数组
            target_duration: 目标时长（秒）
            sample_rate: 采样率
        
        Returns:
            调整后的音频数组
        """
        current_duration = len(audio) / sample_rate
        
        if abs(current_duration - target_duration) < 0.01:
            return audio
        
        rate = current_duration / target_duration
        
        logger.debug(f"调整音频时长: {current_duration:.2f}s -> {target_duration:.2f}s (rate={rate:.2f})")
        
        if self.time_stretch_method == "librosa":
            return self.time_stretch_librosa(audio, rate)
        else:
            # FFmpeg方法需要先保存到临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                tmp_in_path = tmp_in.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                tmp_out_path = tmp_out.name
            
            try:
                self.save_audio(audio, tmp_in_path, sample_rate)
                self.time_stretch_ffmpeg(tmp_in_path, tmp_out_path, rate)
                stretched_audio, _ = self.load_audio(tmp_out_path, sample_rate)
                return stretched_audio
            finally:
                if os.path.exists(tmp_in_path):
                    os.remove(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.remove(tmp_out_path)
    
    def concatenate_audio_segments(
        self,
        segments: list,
        output_path: str,
        sample_rate: int,
    ) -> str:
        """连接多个音频段
        
        Args:
            segments: 音频段列表，每个元素为 (audio_array, start_time, end_time)
            output_path: 输出文件路径
            sample_rate: 采样率
        
        Returns:
            输出文件路径
        """
        if not segments:
            raise ValueError("音频段列表为空")
        
        logger.info(f"连接 {len(segments)} 个音频段")
        
        # 计算总时长
        total_duration = max(seg[2] for seg in segments)  # 最大结束时间
        total_samples = int(total_duration * sample_rate)
        
        # 创建输出音频数组
        output_audio = np.zeros(total_samples, dtype=np.float32)
        
        # 填充每个段
        for audio, start_time, end_time in segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_length = end_sample - start_sample
            
            # 调整音频长度以匹配时间段
            if len(audio) != segment_length:
                duration = (end_time - start_time)
                audio = self.match_duration(audio, duration, sample_rate)
            
            # 确保不超出边界
            actual_length = min(len(audio), len(output_audio) - start_sample)
            output_audio[start_sample:start_sample + actual_length] = audio[:actual_length]
        
        # 保存
        self.save_audio(output_audio, output_path, sample_rate)
        logger.info(f"音频连接完成: {output_path}")
        return output_path
    
    def extract_audio_segment(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """从音频文件提取指定时间段
        
        Args:
            audio_path: 音频文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出路径（可选）
        
        Returns:
            (音频数组, 采样率)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 读取音频
        audio, sr = self.load_audio(audio_path)
        
        # 计算样本索引
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 确保索引在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        # 提取片段
        segment = audio[start_sample:end_sample]
        
        # 如果指定了输出路径，保存文件
        if output_path:
            self.save_audio(segment, output_path, sr)
            logger.debug(f"提取音频片段: {start_time:.2f}s - {end_time:.2f}s -> {output_path}")
        
        return segment, sr
