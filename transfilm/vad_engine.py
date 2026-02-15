#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAD引擎 - 语音活动检测
使用Silero VAD或WebRTC VAD进行语音活动检测
"""

import os
import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
import soundfile as sf

logger = logging.getLogger(__name__)


class VADSegment:
    """VAD片段"""
    
    def __init__(self, start_time: float, end_time: float, audio_array: np.ndarray):
        """初始化VAD片段
        
        Args:
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            audio_array: 音频数据
        """
        self.start_time = start_time
        self.end_time = end_time
        self.audio_array = audio_array
    
    def duration(self) -> float:
        """获取片段时长"""
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"VADSegment({self.start_time:.2f}s - {self.end_time:.2f}s, duration={self.duration():.2f}s)"


class VADEngine:
    """VAD引擎"""
    
    def __init__(
        self,
        min_silence_duration: float = 0.3,
        speech_pad_ms: int = 30,
        threshold: float = 0.5,
        use_silero: bool = True,
        sample_rate: int = 16000,
    ):
        """初始化VAD引擎
        
        Args:
            min_silence_duration: 最小静音时长（秒），超过此时长的静音将作为分段边界
            speech_pad_ms: 语音片段前后填充的毫秒数
            threshold: VAD阈值（0-1），越高越严格
            use_silero: 是否使用Silero VAD（否则使用WebRTC VAD）
            sample_rate: 音频采样率
        """
        self.min_silence_duration = min_silence_duration
        self.speech_pad_ms = speech_pad_ms
        self.threshold = threshold
        self.use_silero = use_silero
        self.sample_rate = sample_rate
        
        self.model = None
        self._silero_available = False
        self._webrtc_available = False
        
        logger.info(f"VAD引擎初始化: use_silero={use_silero}, threshold={threshold}")
    
    def load_model(self):
        """加载VAD模型"""
        if self.model is not None:
            logger.warning("VAD模型已加载")
            return
        
        try:
            if self.use_silero:
                # 尝试加载Silero VAD
                try:
                    import torch
                    # 加载Silero VAD模型
                    self.model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        onnx=False
                    )
                    self._silero_available = True
                    logger.info("Silero VAD模型加载成功")
                except Exception as e:
                    logger.warning(f"加载Silero VAD失败: {e}，尝试WebRTC VAD")
                    self.use_silero = False
            
            if not self.use_silero:
                # 使用WebRTC VAD作为备选
                try:
                    import webrtcvad
                    # 将threshold (0-1) 映射到 WebRTC aggressiveness (0-3)
                    # 使用 min/max 确保在有效范围内
                    aggressiveness = min(3, max(0, int(self.threshold * 3)))
                    self.model = webrtcvad.Vad(aggressiveness)
                    self._webrtc_available = True
                    logger.info(f"WebRTC VAD初始化成功 (aggressiveness={aggressiveness})")
                except ImportError:
                    logger.error("无法导入webrtcvad，请安装: pip install webrtcvad")
                    raise
        
        except Exception as e:
            logger.error(f"加载VAD模型失败: {e}")
            raise
    
    def unload_model(self):
        """卸载VAD模型"""
        if self.model is not None:
            self.model = None
            logger.info("VAD模型已卸载")
    
    def detect_voice_activity(
        self,
        audio_path: str,
        return_audio: bool = True,
    ) -> List[VADSegment]:
        """检测语音活动
        
        Args:
            audio_path: 音频文件路径
            return_audio: 是否在结果中包含音频数据
        
        Returns:
            VAD片段列表
        """
        if self.model is None:
            self.load_model()
        
        # 读取音频
        audio, sr = sf.read(audio_path)
        
        # 转换为单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # 重采样到目标采样率
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # 执行VAD检测
        if self._silero_available and self.use_silero:
            segments = self._detect_with_silero(audio, sr)
        elif self._webrtc_available:
            segments = self._detect_with_webrtc(audio, sr)
        else:
            # 降级：简单的能量检测
            logger.warning("使用简单能量检测作为备选方案")
            segments = self._detect_with_energy(audio, sr)
        
        # 如果需要，提取音频片段
        if return_audio:
            vad_segments = []
            for start_time, end_time in segments:
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_segment = audio[start_sample:end_sample]
                vad_segments.append(VADSegment(start_time, end_time, audio_segment))
        else:
            vad_segments = [VADSegment(start, end, np.array([])) for start, end in segments]
        
        logger.info(f"检测到 {len(vad_segments)} 个语音片段")
        return vad_segments
    
    def _detect_with_silero(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float]]:
        """使用Silero VAD检测"""
        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # 获取语音时间戳
        speech_timestamps = self.model(
            audio_tensor,
            sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=int(self.min_silence_duration * 1000),
            speech_pad_ms=self.speech_pad_ms,
        )
        
        # 转换为时间戳列表
        segments = []
        for ts in speech_timestamps:
            start_time = ts['start'] / sample_rate
            end_time = ts['end'] / sample_rate
            segments.append((start_time, end_time))
        
        return segments
    
    def _detect_with_webrtc(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float]]:
        """使用WebRTC VAD检测"""
        # WebRTC VAD需要特定的采样率和帧大小
        if sample_rate not in [8000, 16000, 32000, 48000]:
            logger.warning(f"WebRTC VAD需要特定采样率，当前: {sample_rate}")
            return self._detect_with_energy(audio, sample_rate)
        
        # 转换为16位整数
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 帧大小（10, 20, 或 30 ms）
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # 按帧检测
        is_speech = []
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) < frame_size:
                break
            try:
                speech = self.model.is_speech(frame.tobytes(), sample_rate)
                is_speech.append(speech)
            except Exception as e:
                logger.warning(f"WebRTC VAD帧检测失败: {e}")
                is_speech.append(False)
        
        # 合并连续的语音帧
        segments = []
        start_idx = None
        min_silence_frames = int(self.min_silence_duration * 1000 / frame_duration_ms)
        silence_count = 0
        
        for i, speech in enumerate(is_speech):
            if speech:
                if start_idx is None:
                    start_idx = i
                silence_count = 0
            else:
                if start_idx is not None:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        # 结束当前片段
                        start_time = start_idx * frame_duration_ms / 1000
                        end_time = (i - silence_count) * frame_duration_ms / 1000
                        segments.append((start_time, end_time))
                        start_idx = None
                        silence_count = 0
        
        # 处理最后一个片段
        if start_idx is not None:
            start_time = start_idx * frame_duration_ms / 1000
            end_time = len(is_speech) * frame_duration_ms / 1000
            segments.append((start_time, end_time))
        
        return segments
    
    def _detect_with_energy(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float]]:
        """使用简单的能量检测作为备选"""
        # 计算短时能量
        frame_length = int(sample_rate * 0.025)  # 25ms
        hop_length = int(sample_rate * 0.010)  # 10ms
        
        # 计算RMS能量
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            rms = np.sqrt(np.mean(frame**2))
            energy.append(rms)
        
        energy = np.array(energy)
        
        # 自适应阈值
        energy_threshold = np.percentile(energy, 50) * (1 + self.threshold)
        
        # 检测语音帧
        is_speech = energy > energy_threshold
        
        # 合并连续的语音帧
        segments = []
        start_idx = None
        min_silence_frames = int(self.min_silence_duration / (hop_length / sample_rate))
        silence_count = 0
        
        for i, speech in enumerate(is_speech):
            if speech:
                if start_idx is None:
                    start_idx = i
                silence_count = 0
            else:
                if start_idx is not None:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        # 结束当前片段
                        start_time = start_idx * hop_length / sample_rate
                        end_time = (i - silence_count) * hop_length / sample_rate
                        if end_time - start_time > 0.1:  # 最小片段长度0.1秒
                            segments.append((start_time, end_time))
                        start_idx = None
                        silence_count = 0
        
        # 处理最后一个片段
        if start_idx is not None:
            start_time = start_idx * hop_length / sample_rate
            end_time = len(is_speech) * hop_length / sample_rate
            if end_time - start_time > 0.1:
                segments.append((start_time, end_time))
        
        return segments
