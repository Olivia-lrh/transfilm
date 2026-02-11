#!/usr/bin/env python3
"""
Web UI for TransFilm using Gradio
"""
import gradio as gr
from pathlib import Path
import tempfile
import os

from transfilm import VideoDubbingPipeline
from transfilm.utils import setup_logger, get_video_info
import config

logger = setup_logger("webui")


class TransFilmWebUI:
    """Web UI wrapper for TransFilm"""
    
    def __init__(self):
        self.pipeline = None
        self.progress_message = ""
        self.progress_value = 0.0
    
    def progress_callback(self, message: str, progress: float):
        """Update progress"""
        self.progress_message = message
        self.progress_value = progress
    
    def process_video(
        self,
        input_video,
        asr_model: str,
        translation_model: str,
        forced_aligner_model: str,
        tts_model: str,
        device: str,
        chunk_size: float,
        source_language: str,
        target_language: str,
        progress=gr.Progress()
    ):
        """Process video with progress tracking"""
        if input_video is None:
            return None, "❌ Please upload a video file"
        
        try:
            # Get input path
            input_path = Path(input_video)
            
            # Create output path
            output_dir = Path(tempfile.mkdtemp(prefix="transfilm_output_"))
            output_path = output_dir / f"dubbed_{input_path.name}"
            
            logger.info(f"Processing video: {input_path}")
            
            # Create pipeline with progress callback
            def gradio_progress_callback(message: str, prog: float):
                self.progress_callback(message, prog)
                progress(prog / 100, desc=message)
            
            self.pipeline = VideoDubbingPipeline(
                asr_model=asr_model,
                translation_model=translation_model,
                forced_aligner_model=forced_aligner_model,
                tts_model=tts_model,
                device=device,
                chunk_size=chunk_size,
                source_language=source_language,
                target_language=target_language,
                progress_callback=gradio_progress_callback
            )
            
            # Process video
            result = self.pipeline.process_video(
                input_path,
                output_path,
                keep_temp=False
            )
            
            success_msg = f"✅ Video dubbing complete! Duration: {self.progress_message}"
            logger.info(success_msg)
            
            return str(result), success_msg
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    def get_video_info_ui(self, input_video):
        """Get video information for display"""
        if input_video is None:
            return "No video uploaded"
        
        try:
            info = get_video_info(Path(input_video))
            
            info_text = f"""
**Video Information:**
- Duration: {info['duration']:.2f} seconds
- Format: {info['format']}
- Size: {info['size'] / (1024*1024):.2f} MB
- Has Video: {info['has_video']}
- Has Audio: {info['has_audio']}
            """
            return info_text
        except Exception as e:
            return f"Failed to get video info: {e}"


def create_ui():
    """Create Gradio UI"""
    ui = TransFilmWebUI()
    
    with gr.Blocks(
        title="TransFilm - AI Video Dubbing",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 🎬 TransFilm - AI视频二次配音工具
            
            使用 Qwen3-ASR 和 Qwen3-TTS 为视频生成新的配音
            
            Upload a video and generate new dubbing using Qwen3-ASR and Qwen3-TTS
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input / 输入")
                
                input_video = gr.Video(
                    label="Upload Video / 上传视频",
                    height=300
                )
                
                video_info = gr.Markdown(
                    label="Video Info / 视频信息",
                    value="Upload a video to see information"
                )
                
                input_video.change(
                    fn=ui.get_video_info_ui,
                    inputs=[input_video],
                    outputs=[video_info]
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output / 输出")
                
                output_video = gr.Video(
                    label="Dubbed Video / 配音后视频",
                    height=300
                )
                
                status_text = gr.Textbox(
                    label="Status / 状态",
                    value="Ready",
                    interactive=False
                )
        
        with gr.Row():
            gr.Markdown("### ⚙️ Settings / 设置")
        
        with gr.Row():
            with gr.Column():
                asr_model = gr.Textbox(
                    label="ASR Model / ASR模型",
                    value=config.DEFAULT_ASR_MODEL,
                    placeholder="Qwen/Qwen3-ASR-1.7B"
                )
                
                translation_model = gr.Textbox(
                    label="Translation Model / 翻译模型",
                    value=config.DEFAULT_TRANSLATION_MODEL,
                    placeholder="Qwen/Qwen3-0.6B"
                )
            
            with gr.Column():
                forced_aligner_model = gr.Textbox(
                    label="Forced Aligner Model / 时间对齐模型",
                    value=config.DEFAULT_FORCED_ALIGNER_MODEL,
                    placeholder="Qwen/Qwen3-ForcedAligner-0.6B"
                )
                
                tts_model = gr.Textbox(
                    label="TTS Model / TTS模型",
                    value=config.DEFAULT_TTS_MODEL,
                    placeholder="Qwen/Qwen3-TTS"
                )
            
            with gr.Column():
                device = gr.Radio(
                    label="Device / 设备",
                    choices=["cuda", "cpu"],
                    value=config.DEFAULT_DEVICE
                )
                
                source_language = gr.Radio(
                    label="Source Language / 源语言",
                    choices=["zh", "en"],
                    value=config.DEFAULT_LANGUAGE
                )
                
                target_language = gr.Radio(
                    label="Target Language / 目标语言",
                    choices=["zh", "en"],
                    value=config.DEFAULT_TARGET_LANGUAGE
                )
            
            with gr.Column():
                chunk_size = gr.Slider(
                    label="Chunk Size (seconds) / 分块大小（秒）",
                    minimum=5,
                    maximum=60,
                    value=config.DEFAULT_CHUNK_SIZE,
                    step=5
                )
        
        with gr.Row():
            process_btn = gr.Button(
                "🚀 Start Dubbing / 开始配音",
                variant="primary",
                size="lg"
            )
        
        # Process button click
        process_btn.click(
            fn=ui.process_video,
            inputs=[
                input_video,
                asr_model,
                translation_model,
                forced_aligner_model,
                tts_model,
                device,
                chunk_size,
                source_language,
                target_language
            ],
            outputs=[output_video, status_text]
        )
        
        gr.Markdown(
            """
            ---
            ### 📖 Usage Instructions / 使用说明
            
            1. **Upload Video / 上传视频**: Click the video upload area to select a video file
            2. **Configure Models / 配置模型**: Adjust ASR, Translation, Aligner and TTS models
            3. **Set Languages / 设置语言**: Select source and target languages
            4. **Start Processing / 开始处理**: Click "Start Dubbing" button to begin
            5. **Download Result / 下载结果**: Download the dubbed video when processing is complete
            
            **Enhanced Workflow / 增强工作流:**
            - Stage 1: ASR transcription with Qwen3-ASR-1.7B
            - Stage 2: Segmentation & translation with Qwen3-0.6B
            - Stage 3: Timestamp alignment with Qwen3-ForcedAligner-0.6B
            - Stage 4: Voice cloning & TTS with Qwen3-TTS
            - Stage 5: Audio assembly with timestamp synchronization
            
            **Tips / 提示:**
            - Use CPU mode if you have limited GPU memory / 显存不足时使用CPU模式
            - Smaller chunk sizes use less memory but take longer / 更小的分块使用更少显存但耗时更长
            - First run will download models (may take time) / 首次运行需下载模型（可能需要时间）
            - Translation tries to match character counts for better duration / 翻译会尝试匹配字符数以获得更好的时长
            """
        )
    
    return app


def main():
    """Main entry point"""
    logger.info("Starting TransFilm Web UI")
    
    app = create_ui()
    
    app.launch(
        server_name=config.WEBUI_HOST,
        server_port=config.WEBUI_PORT,
        share=config.WEBUI_SHARE,
        show_error=True
    )


if __name__ == "__main__":
    main()
