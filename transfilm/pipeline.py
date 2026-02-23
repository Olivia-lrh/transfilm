"""
Main processing pipeline for TransFilm
Orchestrates the entire video dubbing workflow with multiple Qwen models
"""
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import tempfile
import shutil

from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from .asr_engine import ASREngine
from .translation_engine import TranslationEngine
from .forced_aligner_engine import ForcedAlignerEngine
from .tts_engine import TTSEngine
from .speaker_analyzer import SpeakerAnalyzer
from .llm_engine import LLMEngine
from .utils import setup_logger, format_time, ensure_dir
import config

logger = setup_logger(__name__)


class VideoDubbingPipeline:
    """Complete pipeline for AI video dubbing"""
    
    def __init__(
        self,
        asr_model: str = None,
        translation_model: str = None,
        forced_aligner_model: str = None,
        tts_model: str = None,
        llm_model: str = None,
        device: str = None,
        chunk_size: float = None,
        source_language: str = None,
        target_language: str = None,
        sample_rate: int = None,
        enable_speaker_diarization: bool = None,
        enable_llm_features: bool = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize video dubbing pipeline with enhanced workflow
        
        Args:
            asr_model: ASR model name or path (Qwen3-ASR-1.7B)
            translation_model: Translation model name or path (Qwen3-0.6B)
            forced_aligner_model: Forced aligner model name or path (Qwen3-ForcedAligner-0.6B)
            tts_model: TTS model name or path (Qwen3-TTS)
            llm_model: LLM model name or path (for segmentation and analysis)
            device: Device to run on
            chunk_size: Audio chunk size in seconds
            source_language: Source language code
            target_language: Target language code
            sample_rate: Audio sample rate
            enable_speaker_diarization: Enable speaker identification and multi-speaker cloning
            enable_llm_features: Enable LLM-based features (segmentation, length adjustment)
            progress_callback: Function(message, progress_percent) for progress updates
        """
        self.asr_model = asr_model or config.DEFAULT_ASR_MODEL
        self.translation_model = translation_model or config.DEFAULT_TRANSLATION_MODEL
        self.forced_aligner_model = forced_aligner_model or config.DEFAULT_FORCED_ALIGNER_MODEL
        self.tts_model = tts_model or config.DEFAULT_TTS_MODEL
        self.llm_model = llm_model or getattr(config, 'DEFAULT_LLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
        self.device = device or config.DEFAULT_DEVICE
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.source_language = source_language or config.DEFAULT_LANGUAGE
        self.target_language = target_language or config.DEFAULT_TARGET_LANGUAGE
        self.sample_rate = sample_rate or config.DEFAULT_SAMPLE_RATE
        self.enable_speaker_diarization = (
            enable_speaker_diarization if enable_speaker_diarization is not None
            else getattr(config, 'ENABLE_SPEAKER_DIARIZATION', True)
        )
        self.enable_llm_features = (
            enable_llm_features if enable_llm_features is not None
            else getattr(config, 'ENABLE_LLM_SEGMENTATION', True)
        )
        self.progress_callback = progress_callback
        
        self.logger = logger
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)
        
        # Engines will be initialized lazily
        self.asr_engine = None
        self.translation_engine = None
        self.forced_aligner_engine = None
        self.tts_engine = None
        self.speaker_analyzer = None
        self.llm_engine = None
        
        # Working directory for temporary files
        self.work_dir = None
    
    def _report_progress(self, message: str, progress: float = 0.0) -> None:
        """Report progress to callback and logger"""
        self.logger.info(f"Progress {progress:.1f}%: {message}")
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def process_video(
        self,
        input_video: Path,
        output_video: Path,
        keep_temp: bool = False
    ) -> Path:
        """
        Process video file with enhanced multi-speaker workflow:
        1. Audio extraction and speaker analysis
        2. ASR transcription
        3. LLM-based sentence segmentation and speaker assignment
        4. Translation with LLM length adjustment
        5. Timestamp alignment
        6. Multi-speaker voice cloning and TTS
        7. Audio concatenation and video assembly
        
        Args:
            input_video: Path to input video file
            output_video: Path to output video file
            keep_temp: Whether to keep temporary files
            
        Returns:
            Path to output video file
        """
        start_time = time.time()
        input_video = Path(input_video)
        output_video = Path(output_video)
        
        self._report_progress(f"Starting enhanced video dubbing: {input_video.name}", 0)
        
        # Create temporary working directory
        self.work_dir = Path(tempfile.mkdtemp(prefix="transfilm_"))
        self.logger.info(f"Working directory: {self.work_dir}")
        
        try:
            # === Stage 1: Extract audio and analyze speakers (0-15%) ===
            self._report_progress("Stage 1: Extracting audio from video", 2)
            audio_file = self.work_dir / "original_audio.wav"
            video_only_file = self.work_dir / "video_only.mp4"
            
            self.video_processor.extract_audio(
                input_video, audio_file, self.sample_rate
            )
            self.video_processor.extract_video_without_audio(
                input_video, video_only_file
            )
            
            # Load full audio
            audio_data, sr = self.audio_processor.load_audio(audio_file)
            audio_duration = len(audio_data) / sr
            self._report_progress(f"Audio extracted: {audio_duration:.2f}s", 8)
            
            # Analyze speakers if enabled
            speaker_segments = None
            voice_samples = {}
            
            if self.enable_speaker_diarization:
                self._report_progress("Analyzing speakers in audio", 10)
                speaker_segments = self._analyze_speakers(audio_data, sr)
                
                # Extract voice samples for each speaker (5 seconds each)
                voice_samples = self._extract_voice_samples(
                    audio_data, sr, speaker_segments
                )
                num_speakers = len(voice_samples)
                self._report_progress(
                    f"Speaker analysis complete: {num_speakers} speakers detected",
                    15
                )
            else:
                self._report_progress("Speaker analysis skipped", 15)
            
            # === Stage 2: ASR Transcription (15-30%) ===
            self._report_progress("Stage 2: Transcribing audio with Qwen3-ASR-1.7B", 17)
            transcription = self._transcribe_audio(audio_data)
            self._report_progress(f"Transcription complete: {len(transcription)} chars", 30)
            
            # === Stage 3: LLM-based Segmentation and Speaker Assignment (30-45%) ===
            if self.enable_llm_features:
                self._report_progress(
                    "Stage 3: LLM-based sentence segmentation and speaker assignment",
                    32
                )
                text_segments = self._llm_segment_and_assign_speakers(
                    transcription, speaker_segments
                )
            else:
                self._report_progress(
                    "Stage 3: Rule-based segmentation and speaker assignment",
                    32
                )
                text_segments = self._simple_segment_and_assign(
                    transcription, speaker_segments
                )
            
            self._report_progress(
                f"Segmentation complete: {len(text_segments)} segments",
                45
            )
            
            # === Stage 4: Translation with LLM Length Adjustment (45-60%) ===
            self._report_progress("Stage 4: Translating segments", 47)
            translated_segments = self._translate_segments(text_segments)
            self._report_progress(
                f"Translation complete: {len(translated_segments)} segments",
                60
            )
            
            # === Stage 5: Timestamp Alignment (60-70%) ===
            self._report_progress(
                "Stage 5: Generating timestamps with Qwen3-ForcedAligner-0.6B",
                62
            )
            
            # Align timestamps for each segment
            segments_with_timestamps = self._align_segment_timestamps(
                audio_data, translated_segments
            )
            
            # Refine to match total duration
            segments_with_timestamps = self._refine_segment_timestamps(
                segments_with_timestamps, audio_duration
            )
            
            self._report_progress(
                f"Timestamps generated for {len(segments_with_timestamps)} segments",
                70
            )
            
            # === Stage 6: Multi-Speaker Voice Cloning and TTS (70-90%) ===
            self._report_progress("Stage 6: Generating speech with multi-speaker TTS", 72)
            
            if self.enable_speaker_diarization and voice_samples:
                # Multi-speaker synthesis
                new_audio_segments = self._synthesize_multi_speaker(
                    segments_with_timestamps,
                    voice_samples
                )
            else:
                # Single voice synthesis
                voice_features = self.audio_processor.extract_voice_features(audio_data)
                new_audio_segments = self._synthesize_single_speaker(
                    segments_with_timestamps,
                    voice_features
                )
            
            self._report_progress("TTS generation complete", 90)
            
            # === Stage 7: Audio Concatenation (90-95%) ===
            self._report_progress("Stage 7: Concatenating audio segments", 91)
            final_audio = self._concatenate_by_timestamps(
                new_audio_segments,
                segments_with_timestamps,
                audio_duration
            )
            
            # Final validation
            final_duration = len(final_audio) / self.sample_rate
            self.logger.info(
                f"Final audio duration: {final_duration:.2f}s "
                f"(target: {audio_duration:.2f}s, "
                f"difference: {abs(final_duration - audio_duration):.3f}s)"
            )
            self._report_progress("Audio concatenation complete", 95)
            
            # === Stage 8: Save and Combine (95-100%) ===
            self._report_progress("Stage 8: Combining video and audio", 96)
            new_audio_file = self.work_dir / "dubbed_audio.wav"
            self.audio_processor.save_audio(final_audio, new_audio_file)
            
            self.video_processor.combine_audio_video(
                video_only_file,
                new_audio_file,
                output_video,
                video_codec='libx264',
                audio_codec='aac'
            )
            self._report_progress("Video assembly complete", 98)
            
            # Cleanup
            if not keep_temp:
                self._report_progress("Cleaning up temporary files", 99)
                shutil.rmtree(self.work_dir)
            else:
                self.logger.info(f"Temporary files kept in: {self.work_dir}")
            
            elapsed_time = time.time() - start_time
            self._report_progress(
                f"Enhanced multi-speaker video dubbing complete in {format_time(elapsed_time)}",
                100
            )
            
            return output_video
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            # Cleanup on error
            if self.work_dir and self.work_dir.exists() and not keep_temp:
                shutil.rmtree(self.work_dir)
            raise
        
        finally:
            # Unload models to free memory
            if self.asr_engine:
                self.asr_engine.unload_model()
            if self.translation_engine:
                self.translation_engine.unload_model()
            if self.forced_aligner_engine:
                self.forced_aligner_engine.unload_model()
            if self.tts_engine:
                self.tts_engine.unload_model()
            if self.speaker_analyzer:
                self.speaker_analyzer.unload_model()
            if self.llm_engine:
                self.llm_engine.unload_model()
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe full audio with Qwen3-ASR-1.7B
        
        Args:
            audio_data: Full audio data
            
        Returns:
            Complete transcription
        """
        self.logger.info("Transcribing audio with Qwen3-ASR-1.7B")
        
        # Initialize ASR engine
        if not self.asr_engine:
            self.asr_engine = ASREngine(
                model_name=self.asr_model,
                device=self.device
            )
        
        with self.asr_engine:
            transcription = self.asr_engine.transcribe(
                audio_data,
                self.sample_rate,
                self.source_language
            )
        
        self.logger.info(f"Transcription: {transcription[:100]}...")
        return transcription
    
    def _segment_and_translate(
        self,
        transcription: str
    ) -> List[Dict[str, Any]]:
        """
        Segment and translate text with Qwen3-0.6B
        
        Args:
            transcription: Full transcription text
            
        Returns:
            List of translation results with character matching
        """
        self.logger.info("Segmenting and translating with Qwen3-0.6B")
        
        # Initialize translation engine
        if not self.translation_engine:
            self.translation_engine = TranslationEngine(
                model_name=self.translation_model,
                device=self.device
            )
        
        with self.translation_engine:
            results = self.translation_engine.segment_and_translate(
                transcription,
                self.source_language,
                self.target_language
            )
        
        # Log statistics
        total_original = sum(r['original_length'] for r in results)
        total_translation = sum(r['translation_length'] for r in results)
        avg_ratio = total_translation / total_original if total_original > 0 else 1.0
        
        self.logger.info(
            f"Translated {len(results)} segments. "
            f"Character ratio: {avg_ratio:.2f}"
        )
        
        return results
    
    def _align_timestamps(
        self,
        audio_data: np.ndarray,
        text_segments: List[str]
    ) -> List[Dict[str, float]]:
        """
        Generate timestamps with Qwen3-ForcedAligner-0.6B
        
        Args:
            audio_data: Full audio data
            text_segments: List of text segments to align
            
        Returns:
            List of alignments with timestamps
        """
        self.logger.info("Generating timestamps with Qwen3-ForcedAligner-0.6B")
        
        # Initialize forced aligner engine
        if not self.forced_aligner_engine:
            self.forced_aligner_engine = ForcedAlignerEngine(
                model_name=self.forced_aligner_model,
                device=self.device
            )
        
        with self.forced_aligner_engine:
            alignments = self.forced_aligner_engine.align_text_to_audio(
                audio_data,
                text_segments,
                self.sample_rate
            )
        
        # Log alignment statistics
        total_duration = sum(a['duration'] for a in alignments)
        self.logger.info(
            f"Generated {len(alignments)} alignments, "
            f"total duration: {total_duration:.2f}s"
        )
        
        return alignments
    
    def _refine_alignments(
        self,
        alignments: List[Dict[str, Any]],
        target_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Refine alignments to match target duration exactly
        
        Args:
            alignments: Initial alignments
            target_duration: Target total duration
            
        Returns:
            Refined alignments
        """
        self.logger.info(f"Refining alignments to match {target_duration:.2f}s")
        
        if not self.forced_aligner_engine:
            self.forced_aligner_engine = ForcedAlignerEngine(
                model_name=self.forced_aligner_model,
                device=self.device
            )
        
        refined = self.forced_aligner_engine.refine_alignments(
            alignments,
            target_duration
        )
        
        return refined
    
    def _synthesize_with_timestamps(
        self,
        translation_results: List[Dict[str, Any]],
        alignments: List[Dict[str, float]],
        reference_audio: np.ndarray,
        voice_features: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Synthesize audio segments with Qwen3-TTS matching timestamps
        
        Args:
            translation_results: Translation results with character counts
            alignments: Timestamp alignments
            reference_audio: Original audio for voice cloning
            voice_features: Extracted voice features
            
        Returns:
            List of synthesized audio segments
        """
        self.logger.info("Synthesizing audio with Qwen3-TTS")
        
        if len(translation_results) != len(alignments):
            raise ValueError(
                f"Mismatch: {len(translation_results)} translations but "
                f"{len(alignments)} alignments"
            )
        
        # Initialize TTS engine
        if not self.tts_engine:
            self.tts_engine = TTSEngine(
                model_name=self.tts_model,
                device=self.device
            )
        
        audio_segments = []
        total_segments = len(translation_results)
        
        with self.tts_engine:
            for i, (trans_result, alignment) in enumerate(
                zip(translation_results, alignments)
            ):
                progress = 62 + (23 * (i + 1) / total_segments)
                self._report_progress(
                    f"Synthesizing segment {i+1}/{total_segments}",
                    progress
                )
                
                # Get translated text and target duration
                translated_text = trans_result['translation']
                target_duration = alignment['duration']
                
                self.logger.info(
                    f"Segment {i+1}: '{translated_text[:50]}...' "
                    f"target duration: {target_duration:.2f}s"
                )
                
                # Extract reference audio segment for voice cloning
                start_sample = int(alignment['start'] * self.sample_rate)
                end_sample = int(alignment['end'] * self.sample_rate)
                ref_segment = reference_audio[start_sample:end_sample]
                
                # Synthesize with precise duration matching
                audio_segment = self.tts_engine.synthesize(
                    translated_text,
                    sample_rate=self.sample_rate,
                    voice_features=voice_features,
                    target_duration=target_duration,
                    reference_audio=ref_segment
                )
                
                audio_segments.append(audio_segment)
                
                # Validate duration
                actual_duration = len(audio_segment) / self.sample_rate
                duration_diff = abs(actual_duration - target_duration)
                
                if duration_diff > 0.1:
                    self.logger.warning(
                        f"Duration mismatch in segment {i+1}: "
                        f"target {target_duration:.2f}s, actual {actual_duration:.2f}s, "
                        f"diff {duration_diff:.2f}s"
                    )
        
        return audio_segments
    
    def _analyze_speakers(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict]:
        """Analyze audio and identify speakers"""
        if self.speaker_analyzer is None:
            self.speaker_analyzer = SpeakerAnalyzer(
                device=self.device,
                min_speakers=getattr(config, 'MIN_SPEAKERS', 1),
                max_speakers=getattr(config, 'MAX_SPEAKERS', 10),
                voice_sample_duration=getattr(config, 'VOICE_SAMPLE_DURATION', 5.0)
            )
        
        speaker_segments = self.speaker_analyzer.analyze_speakers(audio_data, sample_rate)
        return speaker_segments
    
    def _extract_voice_samples(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_segments: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """Extract 5-second voice samples for each speaker"""
        if self.speaker_analyzer is None:
            self.speaker_analyzer = SpeakerAnalyzer(
                device=self.device,
                voice_sample_duration=getattr(config, 'VOICE_SAMPLE_DURATION', 5.0)
            )
        
        voice_samples = self.speaker_analyzer.extract_voice_samples(
            audio_data, sample_rate, speaker_segments
        )
        
        # Save voice samples for debugging if needed
        if self.work_dir:
            for speaker_id, sample in voice_samples.items():
                sample_file = self.work_dir / f"voice_sample_{speaker_id}.wav"
                self.audio_processor.save_audio(sample, sample_file, sample_rate)
        
        return voice_samples
    
    def _llm_segment_and_assign_speakers(
        self,
        transcription: str,
        speaker_segments: Optional[List[Dict]]
    ) -> List[Dict]:
        """Use LLM to segment text and assign speakers"""
        if self.llm_engine is None:
            self.llm_engine = LLMEngine(
                model_name=self.llm_model,
                device=self.device
            )
        
        # LLM-based sentence segmentation
        text_segments = self.llm_engine.segment_sentences(
            transcription,
            speaker_info=speaker_segments,
            language=self.source_language
        )
        
        # If we have speaker analysis, assign speakers to segments
        if speaker_segments and self.speaker_analyzer:
            # First, we need timestamps for text segments
            # This is a chicken-and-egg problem - we need rough timestamps first
            # For now, distribute evenly (will be refined by forced aligner)
            total_chars = sum(len(seg['text']) for seg in text_segments)
            current_time = 0.0
            
            for seg in text_segments:
                # Rough estimate based on character proportion
                char_ratio = len(seg['text']) / total_chars if total_chars > 0 else 0
                duration = char_ratio * len(speaker_segments) * 5.0  # Rough estimate
                seg['start'] = current_time
                seg['end'] = current_time + duration
                current_time += duration
            
            # Assign speakers based on rough timestamps
            text_segments = self.speaker_analyzer.assign_speaker_to_text(
                text_segments, speaker_segments
            )
        
        return text_segments
    
    def _simple_segment_and_assign(
        self,
        transcription: str,
        speaker_segments: Optional[List[Dict]]
    ) -> List[Dict]:
        """Fallback: simple rule-based segmentation"""
        # Use LLMEngine's fallback method
        if self.llm_engine is None:
            self.llm_engine = LLMEngine(model_name=self.llm_model, device=self.device)
        
        text_segments = self.llm_engine._simple_segmentation(
            transcription, self.source_language
        )
        
        # Assign speakers if available
        if speaker_segments and self.speaker_analyzer:
            # Add rough timestamps
            total_chars = sum(len(seg['text']) for seg in text_segments)
            current_time = 0.0
            
            for seg in text_segments:
                char_ratio = len(seg['text']) / total_chars if total_chars > 0 else 0
                duration = char_ratio * 60.0  # Rough 1 minute total
                seg['start'] = current_time
                seg['end'] = current_time + duration
                current_time += duration
            
            text_segments = self.speaker_analyzer.assign_speaker_to_text(
                text_segments, speaker_segments
            )
        
        return text_segments
    
    def _translate_segments(
        self,
        text_segments: List[Dict]
    ) -> List[Dict]:
        """Translate each segment with LLM length adjustment"""
        if self.translation_engine is None:
            self.translation_engine = TranslationEngine(
                model_name=self.translation_model,
                device=self.device
            )
        
        translated_segments = []
        
        for i, segment in enumerate(text_segments):
            text = segment['text']
            
            # Initial translation
            translation = self.translation_engine.translate(
                text,
                source_lang=self.source_language,
                target_lang=self.target_language
            )
            
            # Store translated segment
            translated_segment = {
                'original': text,
                'translation': translation,
                'speaker_id': segment.get('speaker_id', 'UNKNOWN'),
                'start_char': segment.get('start_char', 0),
                'end_char': segment.get('end_char', len(text))
            }
            
            translated_segments.append(translated_segment)
        
        return translated_segments
    
    def _align_segment_timestamps(
        self,
        audio_data: np.ndarray,
        translated_segments: List[Dict]
    ) -> List[Dict]:
        """Generate timestamps for each segment using forced aligner"""
        if self.forced_aligner_engine is None:
            self.forced_aligner_engine = ForcedAlignerEngine(
                model_name=self.forced_aligner_model,
                device=self.device
            )
        
        # Extract original texts for alignment
        original_texts = [seg['original'] for seg in translated_segments]
        
        # Get timestamps
        alignments = self.forced_aligner_engine.align(
            audio_data,
            original_texts,
            language=self.source_language
        )
        
        # Combine with translated segments
        segments_with_timestamps = []
        for segment, alignment in zip(translated_segments, alignments):
            combined = {
                **segment,
                'start': alignment['start'],
                'end': alignment['end'],
                'duration': alignment['duration']
            }
            segments_with_timestamps.append(combined)
        
        return segments_with_timestamps
    
    def _refine_segment_timestamps(
        self,
        segments: List[Dict],
        total_duration: float
    ) -> List[Dict]:
        """Refine timestamps to match total duration"""
        if not segments:
            return segments
        
        # Calculate current total
        current_total = max(seg['end'] for seg in segments)
        
        # Scale if needed
        if abs(current_total - total_duration) > 0.1:
            scale_factor = total_duration / current_total
            self.logger.info(
                f"Scaling timestamps: {current_total:.2f}s -> {total_duration:.2f}s "
                f"(factor: {scale_factor:.3f})"
            )
            
            for seg in segments:
                seg['start'] *= scale_factor
                seg['end'] *= scale_factor
                seg['duration'] = seg['end'] - seg['start']
        
        return segments
    
    def _synthesize_multi_speaker(
        self,
        segments: List[Dict],
        voice_samples: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """Synthesize audio for multiple speakers"""
        if self.tts_engine is None:
            self.tts_engine = TTSEngine(
                model_name=self.tts_model,
                device=self.device
            )
        
        # Use TTS engine's multi-speaker method
        audio_segments = self.tts_engine.synthesize_multi_speaker(
            segments, voice_samples, self.sample_rate
        )
        
        # Apply LLM length adjustment if needed and enabled
        if self.enable_llm_features and self.llm_engine:
            audio_segments = self._adjust_segments_with_llm(
                segments, audio_segments
            )
        
        return audio_segments
    
    def _synthesize_single_speaker(
        self,
        segments: List[Dict],
        voice_features: Dict
    ) -> List[np.ndarray]:
        """Synthesize audio with single voice"""
        if self.tts_engine is None:
            self.tts_engine = TTSEngine(
                model_name=self.tts_model,
                device=self.device
            )
        
        audio_segments = []
        
        for i, segment in enumerate(segments):
            text = segment['translation']
            target_duration = segment.get('duration', 5.0)
            
            audio = self.tts_engine.synthesize(
                text,
                sample_rate=self.sample_rate,
                voice_features=voice_features,
                target_duration=target_duration
            )
            
            audio_segments.append(audio)
        
        return audio_segments
    
    def _adjust_segments_with_llm(
        self,
        segments: List[Dict],
        audio_segments: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Use LLM to adjust translation if duration mismatch is significant"""
        adjusted_segments = []
        
        for i, (segment, audio) in enumerate(zip(segments, audio_segments)):
            target_duration = segment.get('duration', 5.0)
            actual_duration = len(audio) / self.sample_rate
            
            # If significant mismatch, try to adjust translation
            if abs(actual_duration - target_duration) / target_duration > 0.15:  # >15% diff
                self.logger.info(
                    f"Segment {i+1}: Duration mismatch {actual_duration:.2f}s vs "
                    f"{target_duration:.2f}s, attempting LLM adjustment"
                )
                
                # Get adjusted translation
                adjusted_translation = self.llm_engine.adjust_translation_length(
                    original_text=segment['original'],
                    translation=segment['translation'],
                    target_duration=target_duration,
                    actual_duration=actual_duration,
                    language=self.source_language,
                    target_language=self.target_language
                )
                
                # Re-synthesize with adjusted translation
                if adjusted_translation != segment['translation']:
                    self.logger.info(f"Re-synthesizing with adjusted translation")
                    
                    # Get voice sample for this speaker
                    speaker_id = segment.get('speaker_id')
                    reference = None  # Would need to pass voice_samples here
                    
                    audio = self.tts_engine.synthesize(
                        adjusted_translation,
                        sample_rate=self.sample_rate,
                        target_duration=target_duration,
                        reference_audio=reference,
                        speaker_id=speaker_id
                    )
                    adjusted_segments.append(audio)
                else:
                    adjusted_segments.append(audio)
            else:
                adjusted_segments.append(audio)
        
        return adjusted_segments
    
    def _concatenate_by_timestamps(
        self,
        audio_segments: List[np.ndarray],
        segments: List[Dict],
        total_duration: float
    ) -> np.ndarray:
        """Concatenate audio segments according to timestamps"""
        # Use audio processor's timestamp-aware concatenation
        return self.audio_processor.concatenate_audio_with_timestamps(
            audio_segments,
            segments,
            total_duration
        )
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information"""
        return self.video_processor.get_video_info(video_path)
