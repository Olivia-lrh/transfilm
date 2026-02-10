"""
Main processing pipeline for TransFilm
Orchestrates the entire video dubbing workflow
"""
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import tempfile
import shutil

from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from .asr_engine import ASREngine
from .tts_engine import TTSEngine
from .utils import setup_logger, format_time, ensure_dir
import config

logger = setup_logger(__name__)


class VideoDubbingPipeline:
    """Complete pipeline for AI video dubbing"""
    
    def __init__(
        self,
        asr_model: str = None,
        tts_model: str = None,
        device: str = None,
        chunk_size: float = None,
        language: str = None,
        sample_rate: int = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize video dubbing pipeline
        
        Args:
            asr_model: ASR model name or path
            tts_model: TTS model name or path
            device: Device to run on
            chunk_size: Audio chunk size in seconds
            language: Language code
            sample_rate: Audio sample rate
            progress_callback: Function(message, progress_percent) for progress updates
        """
        self.asr_model = asr_model or config.DEFAULT_ASR_MODEL
        self.tts_model = tts_model or config.DEFAULT_TTS_MODEL
        self.device = device or config.DEFAULT_DEVICE
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.language = language or config.DEFAULT_LANGUAGE
        self.sample_rate = sample_rate or config.DEFAULT_SAMPLE_RATE
        self.progress_callback = progress_callback
        
        self.logger = logger
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)
        
        # Engines will be initialized lazily
        self.asr_engine = None
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
        Process video file to create dubbed version
        
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
        
        self._report_progress(f"Starting video dubbing: {input_video.name}", 0)
        
        # Create temporary working directory
        self.work_dir = Path(tempfile.mkdtemp(prefix="transfilm_"))
        self.logger.info(f"Working directory: {self.work_dir}")
        
        try:
            # Step 1: Extract audio from video (10%)
            self._report_progress("Extracting audio from video", 5)
            audio_file = self.work_dir / "original_audio.wav"
            video_only_file = self.work_dir / "video_only.mp4"
            
            self.video_processor.extract_audio(
                input_video, audio_file, self.sample_rate
            )
            self.video_processor.extract_video_without_audio(
                input_video, video_only_file
            )
            self._report_progress("Audio extracted", 10)
            
            # Step 2: Load and chunk audio (15%)
            self._report_progress("Loading and chunking audio", 12)
            audio_data, sr = self.audio_processor.load_audio(audio_file)
            audio_chunks = self.audio_processor.chunk_audio(
                audio_data,
                self.chunk_size,
                overlap_seconds=0.5
            )
            self._report_progress(f"Created {len(audio_chunks)} audio chunks", 15)
            
            # Step 3: Transcribe audio with ASR (40%)
            self._report_progress("Starting speech recognition", 17)
            transcriptions = self._transcribe_chunks(audio_chunks)
            self._report_progress("Speech recognition complete", 40)
            
            # Step 4: Extract voice features from original audio (45%)
            self._report_progress("Extracting voice features", 42)
            voice_features = self.audio_processor.extract_voice_features(audio_data)
            self._report_progress("Voice features extracted", 45)
            
            # Step 5: Synthesize new audio with TTS (75%)
            self._report_progress("Starting speech synthesis", 47)
            new_audio_chunks = self._synthesize_chunks(
                transcriptions,
                audio_chunks,
                voice_features
            )
            self._report_progress("Speech synthesis complete", 75)
            
            # Step 6: Concatenate audio chunks (80%)
            self._report_progress("Concatenating audio", 77)
            final_audio = self.audio_processor.concatenate_audio(new_audio_chunks)
            
            # Adjust to match original video duration
            video_info = self.video_processor.get_video_info(input_video)
            target_samples = int(video_info['duration'] * self.sample_rate)
            final_audio = self.audio_processor.adjust_audio_length(
                final_audio, target_samples, method='stretch'
            )
            self._report_progress("Audio concatenated", 80)
            
            # Step 7: Save new audio (85%)
            self._report_progress("Saving new audio", 82)
            new_audio_file = self.work_dir / "new_audio.wav"
            self.audio_processor.save_audio(final_audio, new_audio_file)
            self._report_progress("New audio saved", 85)
            
            # Step 8: Combine video and audio (95%)
            self._report_progress("Combining video and audio", 87)
            self.video_processor.combine_audio_video(
                video_only_file,
                new_audio_file,
                output_video,
                video_codec='libx264',
                audio_codec='aac'
            )
            self._report_progress("Video combined", 95)
            
            # Cleanup
            if not keep_temp:
                self._report_progress("Cleaning up temporary files", 97)
                shutil.rmtree(self.work_dir)
            
            elapsed_time = time.time() - start_time
            self._report_progress(
                f"Video dubbing complete in {format_time(elapsed_time)}",
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
            if self.tts_engine:
                self.tts_engine.unload_model()
    
    def _transcribe_chunks(self, audio_chunks: list) -> list:
        """Transcribe audio chunks with ASR"""
        self.logger.info(f"Transcribing {len(audio_chunks)} chunks")
        
        # Initialize ASR engine
        if not self.asr_engine:
            self.asr_engine = ASREngine(
                model_name=self.asr_model,
                device=self.device
            )
        
        transcriptions = []
        total_chunks = len(audio_chunks)
        
        with self.asr_engine:
            for i, (chunk_data, start_time, end_time) in enumerate(audio_chunks):
                progress = 17 + (23 * (i + 1) / total_chunks)
                self._report_progress(
                    f"Transcribing chunk {i+1}/{total_chunks} "
                    f"({format_time(start_time)} - {format_time(end_time)})",
                    progress
                )
                
                transcription = self.asr_engine.transcribe(
                    chunk_data,
                    self.sample_rate,
                    self.language
                )
                transcriptions.append(transcription)
                
                self.logger.info(f"Chunk {i+1} transcription: {transcription[:50]}...")
        
        return transcriptions
    
    def _synthesize_chunks(
        self,
        transcriptions: list,
        audio_chunks: list,
        voice_features: Dict[str, Any]
    ) -> list:
        """Synthesize new audio chunks with TTS"""
        self.logger.info(f"Synthesizing {len(transcriptions)} chunks")
        
        # Initialize TTS engine
        if not self.tts_engine:
            self.tts_engine = TTSEngine(
                model_name=self.tts_model,
                device=self.device
            )
        
        new_audio_chunks = []
        total_chunks = len(transcriptions)
        
        with self.tts_engine:
            for i, (text, (chunk_data, start_time, end_time)) in enumerate(
                zip(transcriptions, audio_chunks)
            ):
                progress = 47 + (28 * (i + 1) / total_chunks)
                self._report_progress(
                    f"Synthesizing chunk {i+1}/{total_chunks}",
                    progress
                )
                
                # Calculate target duration
                target_duration = end_time - start_time
                
                # Synthesize audio
                new_audio = self.tts_engine.synthesize(
                    text,
                    sample_rate=self.sample_rate,
                    voice_features=voice_features if config.ENABLE_VOICE_CLONING else None,
                    target_duration=target_duration
                )
                
                new_audio_chunks.append(new_audio)
                
                self.logger.info(
                    f"Chunk {i+1} synthesized: {len(new_audio)/self.sample_rate:.2f}s"
                )
        
        return new_audio_chunks
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information"""
        return self.video_processor.get_video_info(video_path)
