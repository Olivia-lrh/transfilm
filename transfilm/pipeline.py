#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主管道编排器 - 6阶段视频配音管道
"""

import os
import logging
from typing import Optional, Callable, Dict, Any, List
from transfilm.config import Config
from transfilm.asr_engine import ASREngine, ASRResult
from transfilm.translation_engine import TranslationEngine
from transfilm.tts_engine import TTSEngine
from transfilm.audio_processor import AudioProcessor
from transfilm.video_processor import VideoProcessor
from transfilm.utils import clear_gpu_cache, ensure_dir

logger = logging.getLogger(__name__)


class Pipeline:
    """视频配音管道"""
    
    def __init__(self, config: Optional[Config] = None):
        """初始化管道
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        
        # 初始化各个组件
        self._init_components()
        
        # 管道状态
        self.current_stage = 0
        self.total_stages = 6
        
        logger.info("管道初始化完成")
    
    def _init_components(self):
        """初始化所有组件"""
        # ASR引擎
        asr_config = self.config.get("models.asr", {})
        self.asr_engine = ASREngine(
            model_name=asr_config.get("model_name", "Qwen/Qwen3-ASR-1.7B"),
            forced_aligner=asr_config.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"),
            device=asr_config.get("device", "cuda:0"),
            dtype=asr_config.get("dtype", "bfloat16"),
            max_inference_batch_size=asr_config.get("max_inference_batch_size", 32),
            max_new_tokens=asr_config.get("max_new_tokens", 256),
        )
        
        # 翻译引擎
        trans_config = self.config.get("models.translation", {})
        self.translation_engine = TranslationEngine(
            model_name=trans_config.get("model_name", "openbmb/MiniCPM-o-2_6"),
            device=trans_config.get("device", "cuda:0"),
            dtype=trans_config.get("dtype", "bfloat16"),
            attn_implementation=trans_config.get("attn_implementation", "sdpa"),
            max_new_tokens=trans_config.get("max_new_tokens", 2048),
            sampling=trans_config.get("sampling", False),
        )
        
        # TTS引擎
        tts_config = self.config.get("models.tts", {})
        self.tts_engine = TTSEngine(
            mode=tts_config.get("mode", "custom_voice"),
            custom_voice_model=tts_config.get("custom_voice_model", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            voice_clone_model=tts_config.get("voice_clone_model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            device=tts_config.get("device", "cuda:0"),
            dtype=tts_config.get("dtype", "bfloat16"),
            attn_implementation=tts_config.get("attn_implementation", "sdpa"),
            default_speaker=tts_config.get("default_speaker", "Vivian"),
            default_instruct=tts_config.get("default_instruct", ""),
            x_vector_only_mode=tts_config.get("x_vector_only_mode", False),
        )
        
        # 音频处理器
        audio_config = self.config.get("audio", {})
        self.audio_processor = AudioProcessor(
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            time_stretch_method=audio_config.get("time_stretch_method", "librosa"),
        )
        
        # 视频处理器
        video_config = self.config.get("video", {})
        self.video_processor = VideoProcessor(
            video_codec=video_config.get("video_codec", "libx264"),
            audio_codec=video_config.get("audio_codec", "aac"),
            crf=video_config.get("crf", 23),
            audio_bitrate=video_config.get("audio_bitrate", "192k"),
        )
    
    def run(
        self,
        input_video: str,
        output_video: str,
        target_language: str = "Chinese",
        source_language: Optional[str] = None,
        tts_speaker: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cache_dir: Optional[str] = None,
    ) -> str:
        """运行完整的配音管道
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            target_language: 目标语言
            source_language: 源语言（None表示自动检测）
            tts_speaker: TTS说话人
            progress_callback: 进度回调函数 (current_stage, total_stages, message)
            cache_dir: 缓存目录
        
        Returns:
            输出视频路径
        """
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"输入视频不存在: {input_video}")
        
        # 准备缓存目录
        cache_dir = cache_dir or self.config.get("pipeline.cache_dir", "./cache")
        ensure_dir(cache_dir)
        
        logger.info(f"开始处理视频: {input_video}")
        logger.info(f"目标语言: {target_language}")
        
        try:
            # 阶段1: 提取音频
            self._update_progress(1, "提取音频", progress_callback)
            audio_path = os.path.join(cache_dir, "original_audio.wav")
            self.audio_processor.extract_audio_from_video(input_video, audio_path)
            
            # 阶段2: ASR转录
            self._update_progress(2, "语音识别", progress_callback)
            if self.config.get("pipeline.sequential_model_loading", True):
                self.asr_engine.load_model()
            asr_results = self.asr_engine.transcribe_file(audio_path, source_language)
            if self.config.get("pipeline.sequential_model_loading", True):
                self.asr_engine.unload_model()
                if self.config.get("pipeline.clear_cache_between_stages", True):
                    clear_gpu_cache()
            
            if not asr_results:
                raise ValueError("ASR未识别到任何文本")
            
            logger.info(f"识别到 {len(asr_results)} 段文本")
            
            # 阶段3: 翻译
            self._update_progress(3, "翻译文本", progress_callback)
            if self.config.get("pipeline.sequential_model_loading", True):
                self.translation_engine.load_model()
            
            texts_to_translate = [r.text for r in asr_results]
            translated_texts = self.translation_engine.translate_batch(
                texts_to_translate,
                target_language,
                source_language,
            )
            
            if self.config.get("pipeline.sequential_model_loading", True):
                self.translation_engine.unload_model()
                if self.config.get("pipeline.clear_cache_between_stages", True):
                    clear_gpu_cache()
            
            # 阶段4: TTS合成
            self._update_progress(4, "语音合成", progress_callback)
            if self.config.get("pipeline.sequential_model_loading", True):
                self.tts_engine.load_model()
            
            # 如果是voice_clone模式，创建音色克隆提示
            if self.tts_engine.mode == "voice_clone":
                # 使用原始音频的第一段作为参考
                if asr_results[0].text:
                    self.tts_engine.create_voice_clone_prompt(
                        ref_audio=audio_path,
                        ref_text=asr_results[0].text,
                    )
            
            # 批量合成语音
            tts_kwargs = {}
            if tts_speaker:
                tts_kwargs["speaker"] = tts_speaker
            
            synthesized_audios = self.tts_engine.synthesize_batch(
                translated_texts,
                target_language,
                **tts_kwargs,
            )
            
            if self.config.get("pipeline.sequential_model_loading", True):
                self.tts_engine.unload_model()
                if self.config.get("pipeline.clear_cache_between_stages", True):
                    clear_gpu_cache()
            
            # 阶段5: 组装音频
            self._update_progress(5, "组装音频", progress_callback)
            assembled_audio_path = os.path.join(cache_dir, "assembled_audio.wav")
            self._assemble_audio(
                asr_results,
                synthesized_audios,
                assembled_audio_path,
            )
            
            # 阶段6: 合并视频
            self._update_progress(6, "合并视频", progress_callback)
            # 提取无音频的视频
            video_no_audio = os.path.join(cache_dir, "video_no_audio.mp4")
            self.video_processor.extract_video_without_audio(input_video, video_no_audio)
            
            # 合并音视频
            self.video_processor.merge_audio_video(
                video_no_audio,
                assembled_audio_path,
                output_video,
            )
            
            logger.info(f"处理完成: {output_video}")
            return output_video
            
        except Exception as e:
            logger.error(f"管道执行失败: {e}")
            raise
    
    def _assemble_audio(
        self,
        asr_results: List[ASRResult],
        synthesized_audios: List[tuple],
        output_path: str,
    ):
        """组装音频段
        
        Args:
            asr_results: ASR结果列表
            synthesized_audios: 合成的音频列表 [(audio, sr), ...]
            output_path: 输出路径
        """
        segments = []
        
        for asr_result, (audio, sr) in zip(asr_results, synthesized_audios):
            if len(audio) == 0:
                continue
            
            # 获取时间戳
            if asr_result.time_stamps:
                # 使用ASR的时间戳
                start_time = asr_result.time_stamps[0].start_time if hasattr(asr_result.time_stamps[0], 'start_time') else 0
                end_time = asr_result.time_stamps[-1].end_time if hasattr(asr_result.time_stamps[-1], 'end_time') else (start_time + len(audio) / sr)
            else:
                # 如果没有时间戳，顺序排列
                if segments:
                    start_time = segments[-1][2]  # 上一段的结束时间
                else:
                    start_time = 0.0
                end_time = start_time + len(audio) / sr
            
            segments.append((audio, start_time, end_time))
        
        # 连接所有段
        self.audio_processor.concatenate_audio_segments(
            segments,
            output_path,
            self.audio_processor.sample_rate,
        )
    
    def _update_progress(
        self,
        stage: int,
        message: str,
        callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """更新进度
        
        Args:
            stage: 当前阶段
            message: 进度消息
            callback: 回调函数
        """
        self.current_stage = stage
        logger.info(f"[阶段 {stage}/{self.total_stages}] {message}")
        
        if callback:
            try:
                callback(stage, self.total_stages, message)
            except Exception as e:
                logger.warning(f"进度回调失败: {e}")
