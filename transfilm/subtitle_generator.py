#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕生成器
生成SRT、VTT、ASS等格式的字幕文件
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """字幕生成器"""
    
    def __init__(
        self,
        include_speaker_label: bool = True,
        bilingual: bool = False,
        subtitle_format: str = "srt",
    ):
        """初始化字幕生成器
        
        Args:
            include_speaker_label: 是否包含说话人标签
            bilingual: 是否生成双语字幕
            subtitle_format: 字幕格式（srt, vtt, ass）
        """
        self.include_speaker_label = include_speaker_label
        self.bilingual = bilingual
        self.subtitle_format = subtitle_format.lower()
        
        if self.subtitle_format not in ["srt", "vtt", "ass"]:
            logger.warning(f"不支持的字幕格式: {subtitle_format}，使用srt")
            self.subtitle_format = "srt"
        
        logger.info(f"字幕生成器初始化: format={self.subtitle_format}, bilingual={bilingual}")
    
    def generate(
        self,
        segments: List,
        output_path: str,
        source_texts: Optional[List[str]] = None,
    ) -> str:
        """生成字幕文件
        
        Args:
            segments: 片段列表（SpeakerSegment或ASRResult）
            output_path: 输出路径
            source_texts: 原文文本列表（用于双语字幕）
        
        Returns:
            输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # 根据格式生成字幕
        if self.subtitle_format == "srt":
            content = self._generate_srt(segments, source_texts)
        elif self.subtitle_format == "vtt":
            content = self._generate_vtt(segments, source_texts)
        elif self.subtitle_format == "ass":
            content = self._generate_ass(segments, source_texts)
        else:
            raise ValueError(f"不支持的字幕格式: {self.subtitle_format}")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"字幕文件已生成: {output_path}")
        return output_path
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """格式化时间戳为SRT格式 (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """格式化时间戳为VTT格式 (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def _format_timestamp_ass(self, seconds: float) -> str:
        """格式化时间戳为ASS格式 (H:MM:SS.cc)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        centis = int((td.total_seconds() % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"
    
    def _get_segment_info(self, segment) -> Dict[str, Any]:
        """从segment对象提取信息"""
        info = {
            "text": "",
            "start_time": 0.0,
            "end_time": 0.0,
            "speaker_id": None,
        }
        
        # 尝试不同的属性名
        if hasattr(segment, 'text'):
            info["text"] = segment.text
        
        if hasattr(segment, 'start_time'):
            info["start_time"] = segment.start_time
        elif hasattr(segment, 'time_stamps') and segment.time_stamps:
            if hasattr(segment.time_stamps[0], 'start_time'):
                info["start_time"] = segment.time_stamps[0].start_time
        
        if hasattr(segment, 'end_time'):
            info["end_time"] = segment.end_time
        elif hasattr(segment, 'time_stamps') and segment.time_stamps:
            if hasattr(segment.time_stamps[-1], 'end_time'):
                info["end_time"] = segment.time_stamps[-1].end_time
        
        if hasattr(segment, 'speaker_id'):
            info["speaker_id"] = segment.speaker_id
        
        return info
    
    def _generate_srt(
        self,
        segments: List,
        source_texts: Optional[List[str]] = None,
    ) -> str:
        """生成SRT格式字幕"""
        lines = []
        
        for i, segment in enumerate(segments, start=1):
            info = self._get_segment_info(segment)
            
            # 序号
            lines.append(str(i))
            
            # 时间戳
            start_ts = self._format_timestamp_srt(info["start_time"])
            end_ts = self._format_timestamp_srt(info["end_time"])
            lines.append(f"{start_ts} --> {end_ts}")
            
            # 字幕文本
            text_parts = []
            
            # 添加说话人标签
            if self.include_speaker_label and info["speaker_id"]:
                text_parts.append(f"[{info['speaker_id']}]")
            
            # 添加原文（双语字幕）
            if self.bilingual and source_texts and i-1 < len(source_texts):
                text_parts.append(source_texts[i-1])
            
            # 添加译文
            text_parts.append(info["text"])
            
            lines.append(" ".join(text_parts))
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    def _generate_vtt(
        self,
        segments: List,
        source_texts: Optional[List[str]] = None,
    ) -> str:
        """生成VTT格式字幕"""
        lines = ["WEBVTT", ""]
        
        for i, segment in enumerate(segments, start=1):
            info = self._get_segment_info(segment)
            
            # 时间戳
            start_ts = self._format_timestamp_vtt(info["start_time"])
            end_ts = self._format_timestamp_vtt(info["end_time"])
            lines.append(f"{start_ts} --> {end_ts}")
            
            # 字幕文本
            text_parts = []
            
            # 添加说话人标签（VTT支持voice标签）
            if self.include_speaker_label and info["speaker_id"]:
                text_parts.append(f"<v {info['speaker_id']}>")
            
            # 添加原文（双语字幕）
            if self.bilingual and source_texts and i-1 < len(source_texts):
                text_parts.append(source_texts[i-1])
            
            # 添加译文
            text_parts.append(info["text"])
            
            if self.include_speaker_label and info["speaker_id"]:
                text_parts.append("</v>")
            
            lines.append(" ".join(text_parts))
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    def _generate_ass(
        self,
        segments: List,
        source_texts: Optional[List[str]] = None,
    ) -> str:
        """生成ASS格式字幕"""
        # ASS头部
        lines = [
            "[Script Info]",
            "Title: Generated by Transfilm",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: None",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        
        for i, segment in enumerate(segments):
            info = self._get_segment_info(segment)
            
            # 时间戳
            start_ts = self._format_timestamp_ass(info["start_time"])
            end_ts = self._format_timestamp_ass(info["end_time"])
            
            # 字幕文本
            text_parts = []
            
            # 添加说话人标签
            if self.include_speaker_label and info["speaker_id"]:
                text_parts.append(f"[{info['speaker_id']}]")
            
            # 添加原文（双语字幕，使用换行符）
            if self.bilingual and source_texts and i < len(source_texts):
                text_parts.append(source_texts[i])
                text_parts.append("\\N")  # ASS换行符
            
            # 添加译文
            text_parts.append(info["text"])
            
            text = " ".join(text_parts).replace("\n", "\\N")
            
            # Dialogue行
            speaker_name = info["speaker_id"] if info["speaker_id"] else ""
            lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,{speaker_name},0,0,0,,{text}")
        
        return "\n".join(lines)
