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
        device: str = None,
        chunk_size: float = None,
        source_language: str = None,
        target_language: str = None,
        sample_rate: int = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize video dubbing pipeline with enhanced workflow
        
        Args:
            asr_model: ASR model name or path (Qwen3-ASR-1.7B)
            translation_model: Translation model name or path (Qwen3-0.6B)
            forced_aligner_model: Forced aligner model name or path (Qwen3-ForcedAligner-0.6B)
            tts_model: TTS model name or path (Qwen3-TTS)
            device: Device to run on
            chunk_size: Audio chunk size in seconds
            source_language: Source language code
            target_language: Target language code
            sample_rate: Audio sample rate
            progress_callback: Function(message, progress_percent) for progress updates
        """
        self.asr_model = asr_model or config.DEFAULT_ASR_MODEL
        self.translation_model = translation_model or config.DEFAULT_TRANSLATION_MODEL
        self.forced_aligner_model = forced_aligner_model or config.DEFAULT_FORCED_ALIGNER_MODEL
        self.tts_model = tts_model or config.DEFAULT_TTS_MODEL
        self.device = device or config.DEFAULT_DEVICE
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.source_language = source_language or config.DEFAULT_LANGUAGE
        self.target_language = target_language or config.DEFAULT_TARGET_LANGUAGE
        self.sample_rate = sample_rate or config.DEFAULT_SAMPLE_RATE
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
        Process video file with enhanced workflow:
        1. ASR transcription (Qwen3-ASR-1.7B)
        2. Segmentation and translation (Qwen3-0.6B)
        3. Timestamp alignment (Qwen3-ForcedAligner-0.6B)
        4. Voice cloning and TTS (Qwen3-TTS)
        5. Audio concatenation and video assembly
        
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
            # === Stage 1: Extract and prepare audio (0-10%) ===
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
            self._report_progress(f"Audio extracted: {audio_duration:.2f}s", 10)
            
            # === Stage 2: ASR Transcription (10-25%) ===
            self._report_progress("Stage 2: Transcribing audio with Qwen3-ASR-1.7B", 12)
            transcription = self._transcribe_audio(audio_data)
            self._report_progress(f"Transcription complete: {len(transcription)} chars", 25)
            
            # === Stage 3: Segmentation and Translation (25-45%) ===
            self._report_progress("Stage 3: Segmenting and translating with Qwen3-0.6B", 27)
            translation_results = self._segment_and_translate(transcription)
            self._report_progress(
                f"Translation complete: {len(translation_results)} segments",
                45
            )
            
            # === Stage 4: Timestamp Alignment (45-60%) ===
            self._report_progress("Stage 4: Generating timestamps with Qwen3-ForcedAligner-0.6B", 47)
            original_sentences = [r['original'] for r in translation_results]
            alignments = self._align_timestamps(audio_data, original_sentences)
            
            # Refine alignments to match total duration
            alignments = self._refine_alignments(alignments, audio_duration)
            self._report_progress(f"Timestamps generated for {len(alignments)} segments", 60)
            
            # === Stage 5: Voice Cloning and TTS (60-85%) ===
            self._report_progress("Stage 5: Generating speech with Qwen3-TTS", 62)
            
            # Extract voice features from original audio for cloning
            voice_features = self.audio_processor.extract_voice_features(audio_data)
            
            # Generate new audio segments with precise duration matching
            new_audio_segments = self._synthesize_with_timestamps(
                translation_results,
                alignments,
                audio_data,
                voice_features
            )
            self._report_progress("TTS generation complete", 85)
            
            # === Stage 6: Audio Concatenation (85-92%) ===
            self._report_progress("Stage 6: Concatenating audio segments", 87)
            final_audio = self.audio_processor.concatenate_audio_with_timestamps(
                new_audio_segments,
                alignments,
                audio_duration
            )
            
            # Final validation
            final_duration = len(final_audio) / self.sample_rate
            self.logger.info(
                f"Final audio duration: {final_duration:.2f}s "
                f"(target: {audio_duration:.2f}s, "
                f"difference: {abs(final_duration - audio_duration):.3f}s)"
            )
            self._report_progress("Audio concatenation complete", 92)
            
            # === Stage 7: Save and Combine (92-100%) ===
            self._report_progress("Stage 7: Combining video and audio", 94)
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
                f"Enhanced video dubbing complete in {format_time(elapsed_time)}",
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
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information"""
        return self.video_processor.get_video_info(video_path)
