#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
说话人分离模块
使用Resemblyzer提取说话人嵌入并进行聚类
"""

import os
import logging
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class SpeakerSegment:
    """说话人片段"""
    
    def __init__(
        self,
        speaker_id: str,
        text: str,
        start_time: float,
        end_time: float,
        language: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
    ):
        """初始化说话人片段
        
        Args:
            speaker_id: 说话人ID（如 "speaker_0"）
            text: 文本内容
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            language: 语言
            audio_array: 音频数据（可选）
        """
        self.speaker_id = speaker_id
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.language = language
        self.audio_array = audio_array
    
    def duration(self) -> float:
        """获取片段时长"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "speaker_id": self.speaker_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "language": self.language,
        }
    
    def __repr__(self):
        return f"SpeakerSegment({self.speaker_id}, {self.start_time:.2f}s-{self.end_time:.2f}s, text='{self.text[:30]}...')"


class SpeakerDiarization:
    """说话人分离"""
    
    def __init__(
        self,
        num_speakers: Optional[int] = None,
        min_segment_duration: float = 0.5,
        use_resemblyzer: bool = True,
    ):
        """初始化说话人分离
        
        Args:
            num_speakers: 预期的说话人数量（None表示自动检测）
            min_segment_duration: 最小片段时长（秒）
            use_resemblyzer: 是否使用Resemblyzer（否则使用简单的能量聚类）
        """
        self.num_speakers = num_speakers
        self.min_segment_duration = min_segment_duration
        self.use_resemblyzer = use_resemblyzer
        
        self.encoder = None
        self._resemblyzer_available = False
        
        logger.info(f"说话人分离初始化: num_speakers={num_speakers}, use_resemblyzer={use_resemblyzer}")
    
    def load_model(self):
        """加载说话人编码器模型"""
        if self.encoder is not None:
            logger.warning("说话人编码器已加载")
            return
        
        try:
            if self.use_resemblyzer:
                try:
                    from resemblyzer import VoiceEncoder
                    self.encoder = VoiceEncoder()
                    self._resemblyzer_available = True
                    logger.info("Resemblyzer说话人编码器加载成功")
                except ImportError as e:
                    logger.warning(f"无法导入Resemblyzer: {e}，将使用简单聚类方法")
                    self.use_resemblyzer = False
                    self._resemblyzer_available = False
        except Exception as e:
            logger.error(f"加载说话人编码器失败: {e}")
            self.use_resemblyzer = False
    
    def unload_model(self):
        """卸载说话人编码器模型"""
        if self.encoder is not None:
            self.encoder = None
            logger.info("说话人编码器已卸载")
    
    def diarize(
        self,
        audio_segments: List,
        transcripts: Optional[List[str]] = None,
    ) -> List[str]:
        """执行说话人分离
        
        Args:
            audio_segments: 音频片段列表（VADSegment或带有audio_array属性的对象）
            transcripts: 对应的文本转录（可选，用于LLM验证）
        
        Returns:
            说话人标签列表，与audio_segments对应
        """
        if self.encoder is None:
            self.load_model()
        
        if len(audio_segments) == 0:
            return []
        
        # 提取说话人嵌入
        embeddings = []
        valid_indices = []
        
        for i, segment in enumerate(audio_segments):
            audio = segment.audio_array if hasattr(segment, 'audio_array') else segment
            
            # 跳过太短的片段
            duration = len(audio) / 16000  # 假设16kHz采样率
            if duration < self.min_segment_duration:
                continue
            
            if self._resemblyzer_available and self.use_resemblyzer:
                try:
                    # 使用Resemblyzer提取嵌入
                    embedding = self.encoder.embed_utterance(audio)
                    embeddings.append(embedding)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"提取嵌入失败，片段 {i}: {e}")
            else:
                # 使用简单的声学特征（能量、过零率等）
                embedding = self._extract_simple_features(audio)
                embeddings.append(embedding)
                valid_indices.append(i)
        
        if len(embeddings) == 0:
            logger.warning("没有有效的音频片段用于说话人分离")
            return ["speaker_0"] * len(audio_segments)
        
        embeddings = np.array(embeddings)
        
        # 执行聚类
        speaker_labels = self._cluster_embeddings(embeddings)
        
        # 映射回所有片段
        all_labels = []
        label_idx = 0
        for i in range(len(audio_segments)):
            if i in valid_indices:
                all_labels.append(f"speaker_{speaker_labels[label_idx]}")
                label_idx += 1
            else:
                # 对于太短的片段，使用前一个或后一个标签
                if all_labels:
                    all_labels.append(all_labels[-1])
                else:
                    all_labels.append("speaker_0")
        
        logger.info(f"说话人分离完成，检测到 {len(set(speaker_labels))} 个说话人")
        return all_labels
    
    def _extract_simple_features(self, audio: np.ndarray) -> np.ndarray:
        """提取简单的声学特征作为备选"""
        features = []
        
        # 特征1: 平均能量
        energy = np.sqrt(np.mean(audio**2))
        features.append(energy)
        
        # 特征2: 过零率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)
        features.append(zero_crossings)
        
        # 特征3: 频谱质心（简化版）
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/16000)
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        features.append(spectral_centroid / 1000)  # 归一化
        
        # 特征4: 能量标准差
        frame_size = 400  # 25ms at 16kHz
        hop_size = 160    # 10ms at 16kHz
        frame_energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            frame_energy = np.sqrt(np.mean(frame**2))
            frame_energies.append(frame_energy)
        if frame_energies:
            energy_std = np.std(frame_energies)
            features.append(energy_std)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """对嵌入进行聚类"""
        from sklearn.cluster import AgglomerativeClustering, SpectralClustering
        
        n_samples = len(embeddings)
        
        # 确定聚类数量
        if self.num_speakers is not None:
            n_clusters = min(self.num_speakers, n_samples)
        else:
            # 自动检测说话人数量（使用轮廓系数）
            n_clusters = self._estimate_num_speakers(embeddings)
        
        if n_clusters <= 1:
            return np.zeros(n_samples, dtype=int)
        
        try:
            # 使用层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
            )
            labels = clustering.fit_predict(embeddings)
            logger.info(f"层次聚类完成，聚类数: {n_clusters}")
        except Exception as e:
            logger.warning(f"层次聚类失败: {e}，使用默认标签")
            labels = np.zeros(n_samples, dtype=int)
        
        return labels
    
    def _estimate_num_speakers(self, embeddings: np.ndarray) -> int:
        """估计说话人数量"""
        from sklearn.metrics import silhouette_score
        
        n_samples = len(embeddings)
        max_clusters = min(5, n_samples)  # 最多5个说话人
        
        if n_samples < 2:
            return 1
        
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except Exception as e:
                logger.debug(f"聚类数 {n_clusters} 评估失败: {e}")
        
        logger.info(f"自动检测到 {best_n_clusters} 个说话人（轮廓系数: {best_score:.3f}）")
        return best_n_clusters
    
    def verify_with_llm(
        self,
        speaker_segments: List[SpeakerSegment],
        llm_model,
    ) -> List[SpeakerSegment]:
        """使用LLM验证和优化说话人分离
        
        Args:
            speaker_segments: 初步的说话人片段列表
            llm_model: LLM模型（TranslationEngine实例）
        
        Returns:
            优化后的说话人片段列表
        """
        if len(speaker_segments) <= 1:
            return speaker_segments
        
        # 构建对话文本
        transcript_lines = []
        for seg in speaker_segments:
            transcript_lines.append(f"[{seg.speaker_id}] {seg.text}")
        transcript = "\n".join(transcript_lines)
        
        # 构建提示词
        prompt = f"""Below is a transcript with speaker labels assigned by audio analysis. 
Please verify the speaker assignments based on semantic context (dialogue patterns, 
pronoun usage, topic shifts). Correct any misassignments and merge fragments 
that belong to the same speaker turn.

Transcript:
{transcript}

Output the corrected segmentation in the same format: [speaker_id] text
Keep speaker_id format as speaker_0, speaker_1, etc.
Only output the corrected transcript, no explanations."""
        
        try:
            # 调用LLM
            logger.info("使用LLM验证说话人分离...")
            corrected_text = llm_model.translate_batch(
                [prompt],
                target_language="English",
                source_language="English",
            )[0]
            
            # 解析LLM输出
            corrected_segments = self._parse_llm_output(corrected_text, speaker_segments)
            
            if corrected_segments:
                logger.info(f"LLM验证完成，优化了 {len(corrected_segments)} 个片段")
                return corrected_segments
            else:
                logger.warning("LLM输出解析失败，使用原始分离结果")
                return speaker_segments
        
        except Exception as e:
            logger.warning(f"LLM验证失败: {e}，使用原始分离结果")
            return speaker_segments
    
    def _parse_llm_output(
        self,
        llm_output: str,
        original_segments: List[SpeakerSegment],
    ) -> Optional[List[SpeakerSegment]]:
        """解析LLM输出"""
        lines = llm_output.strip().split('\n')
        
        corrected_segments = []
        segment_idx = 0
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('['):
                continue
            
            try:
                # 解析格式: [speaker_X] text
                end_bracket = line.index(']')
                speaker_id = line[1:end_bracket].strip()
                text = line[end_bracket+1:].strip()
                
                if segment_idx < len(original_segments):
                    orig_seg = original_segments[segment_idx]
                    corrected_segments.append(SpeakerSegment(
                        speaker_id=speaker_id,
                        text=text,
                        start_time=orig_seg.start_time,
                        end_time=orig_seg.end_time,
                        language=orig_seg.language,
                        audio_array=orig_seg.audio_array,
                    ))
                    segment_idx += 1
            except Exception as e:
                logger.debug(f"解析行失败: {line}, 错误: {e}")
                continue
        
        if len(corrected_segments) >= len(original_segments) * 0.8:  # 至少保留80%
            return corrected_segments
        else:
            return None
