#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主管道编排器 - 8阶段视频配音管道
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
from transfilm.vad_engine import VADEngine
from transfilm.speaker_diarization import SpeakerDiarization, SpeakerSegment
from transfilm.subtitle_generator import SubtitleGenerator
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
        self.total_stages = 8  # 从6个阶段升级到8个阶段
        
        logger.info("管道初始化完成（8阶段）")
    
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
        
        # VAD引擎
        vad_config = self.config.get("vad", {})
        if vad_config.get("enabled", True):
            self.vad_engine = VADEngine(
                min_silence_duration=vad_config.get("min_silence_duration", 0.3),
                speech_pad_ms=vad_config.get("speech_pad_ms", 30),
                threshold=vad_config.get("threshold", 0.5),
                use_silero=vad_config.get("use_silero", True),
                sample_rate=audio_config.get("sample_rate", 16000),
            )
        else:
            self.vad_engine = None
        
        # 说话人分离
        speaker_config = self.config.get("speaker_diarization", {})
        if speaker_config.get("enabled", True):
            self.speaker_diarization = SpeakerDiarization(
                num_speakers=speaker_config.get("num_speakers"),
                min_segment_duration=speaker_config.get("min_segment_duration", 0.5),
                use_resemblyzer=speaker_config.get("use_resemblyzer", True),
            )
        else:
            self.speaker_diarization = None
        
        # 字幕生成器
        subtitle_config = self.config.get("subtitle", {})
        if subtitle_config.get("enabled", True):
            self.subtitle_generator = SubtitleGenerator(
                include_speaker_label=subtitle_config.get("include_speaker_label", True),
                bilingual=subtitle_config.get("bilingual", False),
                subtitle_format=subtitle_config.get("format", "srt"),
            )
        else:
            self.subtitle_generator = None
    
    def run(
        self,
        input_video: str,
        output_video: str,
        target_language: str = "Chinese",
        source_language: Optional[str] = None,
        tts_speaker: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cache_dir: Optional[str] = None,
        num_speakers: Optional[int] = None,
    ) -> str:
        """运行完整的配音管道（8阶段）
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            target_language: 目标语言
            source_language: 源语言（None表示自动检测）
            tts_speaker: TTS说话人
            progress_callback: 进度回调函数 (current_stage, total_stages, message)
            cache_dir: 缓存目录
            num_speakers: 说话人数量提示（None表示自动检测）
        
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
        logger.info(f"8阶段增强管道已启用")
        
        try:
            # ===== 阶段1: 提取音频 =====
            self._update_progress(1, "提取音频", progress_callback)
            audio_path = os.path.join(cache_dir, "original_audio.wav")
            self.audio_processor.extract_audio_from_video(input_video, audio_path)
            
            # ===== 阶段2: VAD语音活动检测 =====
            vad_segments = None
            if self.vad_engine is not None:
                self._update_progress(2, "VAD语音活动检测", progress_callback)
                try:
                    if self.config.get("pipeline.sequential_model_loading", True):
                        self.vad_engine.load_model()
                    
                    vad_segments = self.vad_engine.detect_voice_activity(audio_path)
                    logger.info(f"VAD检测到 {len(vad_segments)} 个语音片段")
                    
                    if self.config.get("pipeline.sequential_model_loading", True):
                        self.vad_engine.unload_model()
                        if self.config.get("pipeline.clear_cache_between_stages", True):
                            clear_gpu_cache()
                except Exception as e:
                    logger.warning(f"VAD失败，将使用完整音频: {e}")
                    vad_segments = None
            else:
                logger.info("VAD已禁用，跳过阶段2")
            
            # ===== 阶段3: ASR转录 + 说话人分离 + LLM验证 =====
            self._update_progress(3, "ASR转录与说话人分离", progress_callback)
            
            # 加载ASR模型
            if self.config.get("pipeline.sequential_model_loading", True):
                self.asr_engine.load_model()
            
            # 对每个VAD片段进行转录（或对完整音频转录）
            if vad_segments:
                # 基于VAD片段转录
                asr_results = []
                for vad_seg in vad_segments:
                    # 保存临时音频片段
                    temp_audio_path = os.path.join(cache_dir, f"vad_seg_{len(asr_results)}.wav")
                    self.audio_processor.save_audio(
                        vad_seg.audio_array,
                        temp_audio_path,
                        self.audio_processor.sample_rate,
                    )
                    
                    # 转录
                    try:
                        results = self.asr_engine.transcribe_file(temp_audio_path, source_language)
                        # 调整时间戳
                        for r in results:
                            if r.time_stamps:
                                for ts in r.time_stamps:
                                    if hasattr(ts, 'start_time'):
                                        ts.start_time += vad_seg.start_time
                                    if hasattr(ts, 'end_time'):
                                        ts.end_time += vad_seg.start_time
                        asr_results.extend(results)
                    except Exception as e:
                        logger.warning(f"转录VAD片段失败: {e}")
                    finally:
                        # 清理临时文件
                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
            else:
                # 完整音频转录
                asr_results = self.asr_engine.transcribe_file(audio_path, source_language)
            
            # 卸载ASR模型
            if self.config.get("pipeline.sequential_model_loading", True):
                self.asr_engine.unload_model()
                if self.config.get("pipeline.clear_cache_between_stages", True):
                    clear_gpu_cache()
            
            if not asr_results:
                raise ValueError("ASR未识别到任何文本")
            
            logger.info(f"识别到 {len(asr_results)} 段文本")
            
            # 说话人分离
            speaker_segments = []
            if self.speaker_diarization is not None:
                try:
                    # 加载说话人编码器
                    if self.config.get("pipeline.sequential_model_loading", True):
                        self.speaker_diarization.load_model()
                    
                    # 如果有num_speakers参数，覆盖配置
                    if num_speakers is not None:
                        self.speaker_diarization.num_speakers = num_speakers
                    
                    # 准备音频片段
                    audio_segments = vad_segments if vad_segments else []
                    if not audio_segments:
                        # 如果没有VAD片段，从ASR结果创建片段
                        audio, sr = self.audio_processor.load_audio(audio_path)
                        for asr_result in asr_results:
                            if asr_result.time_stamps:
                                start_time = asr_result.time_stamps[0].start_time if hasattr(asr_result.time_stamps[0], 'start_time') else 0
                                end_time = asr_result.time_stamps[-1].end_time if hasattr(asr_result.time_stamps[-1], 'end_time') else len(audio) / sr
                                start_sample = int(start_time * sr)
                                end_sample = int(end_time * sr)
                                audio_segment = audio[start_sample:end_sample]
                                
                                from transfilm.vad_engine import VADSegment
                                audio_segments.append(VADSegment(start_time, end_time, audio_segment))
                    
                    # 执行说话人分离
                    transcripts = [r.text for r in asr_results]
                    speaker_labels = self.speaker_diarization.diarize(audio_segments, transcripts)
                    
                    # 创建说话人片段
                    for asr_result, speaker_label, audio_seg in zip(asr_results, speaker_labels, audio_segments):
                        start_time = audio_seg.start_time if hasattr(audio_seg, 'start_time') else 0
                        end_time = audio_seg.end_time if hasattr(audio_seg, 'end_time') else 0
                        
                        speaker_seg = SpeakerSegment(
                            speaker_id=speaker_label,
                            text=asr_result.text,
                            start_time=start_time,
                            end_time=end_time,
                            language=asr_result.language,
                            audio_array=audio_seg.audio_array if hasattr(audio_seg, 'audio_array') else None,
                        )
                        speaker_segments.append(speaker_seg)
                    
                    # 卸载说话人编码器
                    if self.config.get("pipeline.sequential_model_loading", True):
                        self.speaker_diarization.unload_model()
                        if self.config.get("pipeline.clear_cache_between_stages", True):
                            clear_gpu_cache()
                    
                    logger.info(f"说话人分离完成，检测到 {len(set(speaker_labels))} 个说话人")
                    
                    # LLM验证（使用翻译模型）
                    if self.config.get("speaker_diarization.use_llm_verification", True):
                        logger.info("使用LLM验证说话人分离...")
                        try:
                            # 加载翻译模型用于验证
                            if self.config.get("pipeline.sequential_model_loading", True):
                                self.translation_engine.load_model()
                            
                            speaker_segments = self.speaker_diarization.verify_with_llm(
                                speaker_segments,
                                self.translation_engine,
                            )
                            
                            # 注意：不要在这里卸载翻译模型，因为下一阶段还要用
                        except Exception as e:
                            logger.warning(f"LLM验证失败: {e}")
                    
                except Exception as e:
                    logger.warning(f"说话人分离失败: {e}，将所有片段标记为speaker_0")
                    # 创建默认说话人片段
                    for asr_result in asr_results:
                        start_time = asr_result.time_stamps[0].start_time if asr_result.time_stamps and hasattr(asr_result.time_stamps[0], 'start_time') else 0
                        end_time = asr_result.time_stamps[-1].end_time if asr_result.time_stamps and hasattr(asr_result.time_stamps[-1], 'end_time') else 0
                        
                        speaker_seg = SpeakerSegment(
                            speaker_id="speaker_0",
                            text=asr_result.text,
                            start_time=start_time,
                            end_time=end_time,
                            language=asr_result.language,
                        )
                        speaker_segments.append(speaker_seg)
            else:
                # 说话人分离已禁用，创建默认片段
                logger.info("说话人分离已禁用")
                for asr_result in asr_results:
                    start_time = asr_result.time_stamps[0].start_time if asr_result.time_stamps and hasattr(asr_result.time_stamps[0], 'start_time') else 0
                    end_time = asr_result.time_stamps[-1].end_time if asr_result.time_stamps and hasattr(asr_result.time_stamps[-1], 'end_time') else 0
                    
                    speaker_seg = SpeakerSegment(
                        speaker_id="speaker_0",
                        text=asr_result.text,
                        start_time=start_time,
                        end_time=end_time,
                        language=asr_result.language,
                    )
                    speaker_segments.append(speaker_seg)
            
            # ===== 阶段4: 智能翻译（字符数控制） =====
            self._update_progress(4, "智能翻译", progress_callback)
            
            # 如果翻译模型还没加载（LLM验证被跳过），现在加载
            if self.translation_engine.model is None and self.config.get("pipeline.sequential_model_loading", True):
                self.translation_engine.load_model()
            
            # 翻译配置
            enable_character_control = self.config.get("translation.enable_character_control", True)
            
            translated_segments = []
            for seg in speaker_segments:
                try:
                    if enable_character_control:
                        # 使用字符数控制的翻译
                        translated = self.translation_engine.translate_with_length_control(
                            text=seg.text,
                            target_language=target_language,
                            target_duration=seg.duration(),
                            source_language=source_language,
                            tolerance=self.config.get("translation.character_count_tolerance", 0.2),
                            max_rounds=self.config.get("translation.max_refinement_rounds", 2),
                        )
                    else:
                        # 常规翻译
                        translated = self.translation_engine.translate(
                            seg.text,
                            target_language,
                            source_language,
                        )
                    
                    translated_segments.append((seg, translated))
                except Exception as e:
                    logger.error(f"翻译失败: {e}")
                    translated_segments.append((seg, seg.text))  # 保留原文
            
            # 卸载翻译模型
            if self.config.get("pipeline.sequential_model_loading", True):
                self.translation_engine.unload_model()
                if self.config.get("pipeline.clear_cache_between_stages", True):
                    clear_gpu_cache()
            
            logger.info(f"翻译完成 {len(translated_segments)} 段")
            
            # ===== 阶段5: 每说话人音色克隆TTS =====
            self._update_progress(5, "语音合成（每说话人音色克隆）", progress_callback)
            
            # 加载TTS模型
            if self.config.get("pipeline.sequential_model_loading", True):
                self.tts_engine.load_model()
            
            # 检查是否启用每说话人克隆
            per_speaker_clone = self.config.get("models.tts.per_speaker_clone", True) and self.tts_engine.mode == "voice_clone"
            
            # 为每个说话人准备音色克隆提示
            speaker_prompts = {}
            if per_speaker_clone:
                # 按说话人分组片段
                speaker_to_segments = {}
                for seg, _ in translated_segments:
                    speaker_id = seg.speaker_id or "speaker_0"
                    if speaker_id not in speaker_to_segments:
                        speaker_to_segments[speaker_id] = []
                    speaker_to_segments[speaker_id].append(seg)
                
                # 为每个说话人选择参考音频
                for speaker_id, segments in speaker_to_segments.items():
                    try:
                        ref_audio, ref_start, ref_end = self.tts_engine.select_reference_audio(
                            segments,
                            min_duration=self.config.get("models.tts.ref_audio_min_duration", 3.0),
                            max_duration=self.config.get("models.tts.ref_audio_max_duration", 10.0),
                        )
                        
                        # 保存参考音频到临时文件
                        ref_audio_path = os.path.join(cache_dir, f"ref_audio_{speaker_id}.wav")
                        self.audio_processor.save_audio(
                            ref_audio,
                            ref_audio_path,
                            self.audio_processor.sample_rate,
                        )
                        
                        # 获取参考文本（使用原始ASR文本）
                        ref_text = ""
                        for seg in segments:
                            if seg.start_time <= ref_start < seg.end_time or seg.start_time < ref_end <= seg.end_time:
                                ref_text = seg.text
                                break
                        if not ref_text and segments:
                            ref_text = segments[0].text
                        
                        # 创建音色克隆提示
                        prompt = self.tts_engine.create_voice_clone_prompt(
                            ref_audio=ref_audio_path,
                            ref_text=ref_text,
                        )
                        speaker_prompts[speaker_id] = prompt
                        
                        logger.info(f"为 {speaker_id} 创建音色克隆提示")
                    except Exception as e:
                        logger.warning(f"为 {speaker_id} 创建音色克隆提示失败: {e}")
            
            # 合成语音
            synthesized_audios = []
            for seg, translated_text in translated_segments:
                try:
                    speaker_id = seg.speaker_id or "speaker_0"
                    
                    if per_speaker_clone and speaker_id in speaker_prompts:
                        # 使用每说话人的音色克隆
                        audio, sr = self.tts_engine.synthesize_voice_clone(
                            text=translated_text,
                            language=target_language,
                            voice_clone_prompt=speaker_prompts[speaker_id],
                        )
                    elif tts_speaker:
                        # 使用指定的自定义音色
                        audio, sr = self.tts_engine.synthesize_custom_voice(
                            text=translated_text,
                            language=target_language,
                            speaker=tts_speaker,
                        )
                    else:
                        # 使用默认合成
                        audio, sr = self.tts_engine.synthesize(
                            text=translated_text,
                            language=target_language,
                        )
                    
                    synthesized_audios.append((audio, sr))
                except Exception as e:
                    logger.error(f"合成失败: {e}")
                    # 返回空音频
                    import numpy as np
                    synthesized_audios.append((np.array([]), 0))
            
            # 卸载TTS模型
            if self.config.get("pipeline.sequential_model_loading", True):
                self.tts_engine.unload_model()
                if self.config.get("pipeline.clear_cache_between_stages", True):
                    clear_gpu_cache()
            
            logger.info(f"语音合成完成 {len(synthesized_audios)} 段")
            
            # ===== 阶段6: 组装音频 =====
            self._update_progress(6, "组装音频", progress_callback)
            assembled_audio_path = os.path.join(cache_dir, "assembled_audio.wav")
            self._assemble_audio_from_speaker_segments(
                translated_segments,
                synthesized_audios,
                assembled_audio_path,
            )
            
            # ===== 阶段7: 合并视频 =====
            self._update_progress(7, "合并视频", progress_callback)
            # 提取无音频的视频
            video_no_audio = os.path.join(cache_dir, "video_no_audio.mp4")
            self.video_processor.extract_video_without_audio(input_video, video_no_audio)
            
            # 合并音视频
            self.video_processor.merge_audio_video(
                video_no_audio,
                assembled_audio_path,
                output_video,
            )
            
            # ===== 阶段8: 生成字幕（可选） =====
            if self.subtitle_generator is not None:
                self._update_progress(8, "生成字幕", progress_callback)
                try:
                    # 准备字幕片段
                    subtitle_segments = []
                    source_texts = []
                    
                    for (seg, translated_text) in translated_segments:
                        # 创建带有翻译文本的片段
                        subtitle_seg = SpeakerSegment(
                            speaker_id=seg.speaker_id,
                            text=translated_text,
                            start_time=seg.start_time,
                            end_time=seg.end_time,
                            language=target_language,
                        )
                        subtitle_segments.append(subtitle_seg)
                        source_texts.append(seg.text)
                    
                    # 生成字幕文件
                    subtitle_path = os.path.splitext(output_video)[0] + ".srt"
                    self.subtitle_generator.generate(
                        subtitle_segments,
                        subtitle_path,
                        source_texts=source_texts if self.subtitle_generator.bilingual else None,
                    )
                    
                    logger.info(f"字幕文件已生成: {subtitle_path}")
                except Exception as e:
                    logger.warning(f"字幕生成失败: {e}")
            else:
                logger.info("字幕生成已禁用，跳过阶段8")
            
            logger.info(f"处理完成: {output_video}")
            return output_video
            
        except Exception as e:
            logger.error(f"管道执行失败: {e}")
            raise
    
    def _assemble_audio_from_speaker_segments(
        self,
        translated_segments: List[tuple],
        synthesized_audios: List[tuple],
        output_path: str,
    ):
        """从说话人片段组装音频
        
        Args:
            translated_segments: (SpeakerSegment, translated_text) 列表
            synthesized_audios: (audio, sr) 列表
            output_path: 输出路径
        """
        segments = []
        
        for (speaker_seg, _), (audio, sr) in zip(translated_segments, synthesized_audios):
            if len(audio) == 0:
                continue
            
            start_time = speaker_seg.start_time
            end_time = speaker_seg.end_time
            
            segments.append((audio, start_time, end_time))
        
        # 连接所有段
        if segments:
            self.audio_processor.concatenate_audio_segments(
                segments,
                output_path,
                self.audio_processor.sample_rate,
            )
        else:
            logger.warning("没有有效的音频片段可组装")
    
    def _assemble_audio(
        self,
        asr_results: List[ASRResult],
        synthesized_audios: List[tuple],
        output_path: str,
    ):
        """组装音频段（旧版本兼容方法）
        
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
