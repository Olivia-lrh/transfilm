#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio Web UI
"""

import os
import sys
import logging
import gradio as gr
from transfilm.config import Config
from transfilm.pipeline import Pipeline
from transfilm.model_downloader import ModelDownloader
from transfilm.utils import setup_logging

logger = logging.getLogger(__name__)

# 全局变量
config = None
pipeline = None
downloader = None


def init_app(config_path=None):
    """初始化应用"""
    global config, pipeline, downloader
    
    config = Config(config_path)
    setup_logging(level=config.get("logging.level", "INFO"))
    
    pipeline = Pipeline(config)
    downloader = ModelDownloader(
        source=config.get("download.source", "huggingface"),
        cache_dir=config.get("download.cache_dir", ""),
        show_progress=config.get("download.show_progress", True),
    )
    
    logger.info("Web UI 初始化完成")


def process_video(
    input_video,
    target_language,
    source_language,
    tts_mode,
    tts_speaker,
    num_speakers,
    per_speaker_clone,
    generate_subtitle,
    bilingual_subtitle,
    progress=gr.Progress(),
):
    """处理视频"""
    if input_video is None:
        return None, "请上传视频文件"
    
    try:
        # 生成输出路径
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        input_name = os.path.splitext(os.path.basename(input_video))[0]
        output_video = os.path.join(output_dir, f"{input_name}_dubbed.mp4")
        
        # 更新配置
        config.set("models.tts.mode", tts_mode)
        config.set("models.tts.per_speaker_clone", per_speaker_clone)
        config.set("subtitle.enabled", generate_subtitle)
        config.set("subtitle.bilingual", bilingual_subtitle)
        
        # 重新初始化组件
        pipeline._init_components()
        
        # 进度回调
        def progress_callback(stage, total, message):
            progress((stage, total), desc=message)
        
        # 运行管道
        result = pipeline.run(
            input_video=input_video,
            output_video=output_video,
            target_language=target_language,
            source_language=source_language if source_language != "自动检测" else None,
            tts_speaker=tts_speaker if tts_speaker != "默认" else None,
            progress_callback=progress_callback,
            num_speakers=num_speakers if num_speakers > 0 else None,
        )
        
        # 检查字幕文件
        subtitle_path = os.path.splitext(result)[0] + ".srt"
        subtitle_msg = ""
        if os.path.exists(subtitle_path):
            subtitle_msg = f"\n字幕文件: {subtitle_path}"
        
        return result, f"✅ 处理完成！\n输出: {result}{subtitle_msg}"
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return None, f"❌ 处理失败: {str(e)}"


def download_model(model_key):
    """下载模型"""
    try:
        if model_key == "全部模型":
            models = None
        else:
            # 从显示名称映射到模型键
            model_map = {
                "ASR模型": "asr",
                "对齐器模型": "forced_aligner",
                "TTS自定义音色": "tts_custom",
                "TTS音色克隆": "tts_clone",
                "翻译模型": "translation",
            }
            models = [model_map.get(model_key, model_key)]
        
        model_paths = downloader.download_all_models(models)
        
        result_text = "下载结果:\n"
        for key, path in model_paths.items():
            if path:
                result_text += f"✅ {key}: {path}\n"
            else:
                result_text += f"❌ {key}: 下载失败\n"
        
        return result_text
        
    except Exception as e:
        logger.error(f"下载失败: {e}", exc_info=True)
        return f"❌ 下载失败: {str(e)}"


def create_ui():
    """创建UI"""
    with gr.Blocks(title="Transfilm - AI视频配音系统") as app:
        gr.Markdown("# 🎬 Transfilm - AI视频配音系统 (8阶段增强版)")
        gr.Markdown("使用Qwen3-ASR、Qwen3-TTS和MiniCPM-o进行高质量视频翻译配音，支持说话人分离和智能翻译")
        
        with gr.Tabs():
            # 主要功能标签页
            with gr.Tab("视频配音"):
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(label="输入视频")
                        
                        with gr.Row():
                            source_language = gr.Dropdown(
                                choices=["自动检测"] + config.get("languages.asr_languages", []),
                                value="自动检测",
                                label="源语言",
                            )
                            target_language = gr.Dropdown(
                                choices=config.get("languages.tts_languages", []),
                                value=config.get("languages.default_target", "Chinese"),
                                label="目标语言",
                            )
                        
                        with gr.Row():
                            tts_mode = gr.Radio(
                                choices=["custom_voice", "voice_clone"],
                                value=config.get("models.tts.mode", "custom_voice"),
                                label="TTS模式",
                                info="custom_voice: 自定义音色 | voice_clone: 音色克隆",
                            )
                        
                        tts_speaker = gr.Dropdown(
                            choices=["默认"] + [s["name"] for s in config.get("speakers.custom_voice", [])],
                            value="默认",
                            label="说话人",
                        )
                        
                        # 新增控件
                        with gr.Accordion("高级选项", open=False):
                            num_speakers = gr.Number(
                                label="说话人数量提示",
                                value=0,
                                precision=0,
                                info="0表示自动检测",
                            )
                            per_speaker_clone = gr.Checkbox(
                                label="每说话人音色克隆",
                                value=config.get("models.tts.per_speaker_clone", True),
                                info="为每个说话人克隆独立的音色",
                            )
                            generate_subtitle = gr.Checkbox(
                                label="生成字幕",
                                value=config.get("subtitle.enabled", True),
                                info="生成SRT字幕文件",
                            )
                            bilingual_subtitle = gr.Checkbox(
                                label="双语字幕",
                                value=config.get("subtitle.bilingual", False),
                                info="包含源语言和目标语言",
                            )
                        
                        process_btn = gr.Button("开始处理", variant="primary", size="lg")
                    
                    with gr.Column():
                        output_video = gr.Video(label="输出视频")
                        output_message = gr.Textbox(label="处理信息", lines=5)
                
                process_btn.click(
                    fn=process_video,
                    inputs=[
                        input_video, 
                        target_language, 
                        source_language, 
                        tts_mode, 
                        tts_speaker,
                        num_speakers,
                        per_speaker_clone,
                        generate_subtitle,
                        bilingual_subtitle,
                    ],
                    outputs=[output_video, output_message],
                )
            
            # 模型下载标签页
            with gr.Tab("模型下载"):
                gr.Markdown("### 下载所需模型")
                gr.Markdown("首次使用前，请下载所需的AI模型")
                
                model_select = gr.Dropdown(
                    choices=[
                        "全部模型",
                        "ASR模型",
                        "对齐器模型",
                        "TTS自定义音色",
                        "TTS音色克隆",
                        "翻译模型",
                    ],
                    value="全部模型",
                    label="选择模型",
                )
                
                download_btn = gr.Button("开始下载", variant="primary")
                download_output = gr.Textbox(label="下载状态", lines=10)
                
                download_btn.click(
                    fn=download_model,
                    inputs=[model_select],
                    outputs=[download_output],
                )
            
            # 信息标签页
            with gr.Tab("说明信息"):
                gr.Markdown("""
                ## 使用说明
                
                ### 1. 下载模型
                首次使用前，请在"模型下载"标签页下载所需模型。
                
                ### 2. 上传视频
                支持常见视频格式：mp4, mkv, avi, mov, webm
                
                ### 3. 选择语言
                - 源语言：可选择自动检测，或指定源语言以提高准确度
                - 目标语言：选择配音的目标语言
                
                ### 4. TTS模式
                - **自定义音色 (custom_voice)**: 使用预设的说话人音色
                - **音色克隆 (voice_clone)**: 克隆原视频中的音色
                
                ### 5. 高级选项 (新功能)
                - **说话人数量提示**: 可以指定视频中的说话人数量，或设为0自动检测
                - **每说话人音色克隆**: 为每个说话人分别克隆音色，保持说话人差异
                - **生成字幕**: 自动生成SRT字幕文件
                - **双语字幕**: 在字幕中同时显示原文和译文
                
                ## 8阶段增强管道
                
                1. **提取音频**: 从视频中提取音频轨道
                2. **VAD语音活动检测**: 检测和分割语音片段
                3. **ASR转录与说话人分离**: 语音识别 + 基于音色的说话人分类 + LLM语义验证
                4. **智能翻译**: 考虑语速的字符数控制翻译，确保译文长度与原音频匹配
                5. **每说话人音色克隆**: 为每个说话人选择最佳参考音频并克隆音色
                6. **组装音频**: 将合成的音频片段组装回完整音轨
                7. **合并视频**: 将新音轨与原视频合并
                8. **生成字幕**: 可选的字幕文件生成
                
                ## 特点
                
                - ✅ 精准的说话人分离
                - ✅ 语义验证的说话人边界优化
                - ✅ 字符数控制的智能翻译
                - ✅ 每说话人独立音色克隆
                - ✅ 自动字幕生成（支持双语）
                - ✅ 适配8GB显存的渐进式模型加载
                
                ## 技术栈
                
                - ASR: Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B
                - Translation: MiniCPM-o-2_6
                - TTS: Qwen3-TTS-12Hz-1.7B (CustomVoice + Base)
                - VAD: Silero VAD
                - Speaker Diarization: Resemblyzer
                """)
        
        gr.Markdown("---")
        gr.Markdown("© 2026 Transfilm | [GitHub](https://github.com/Olivia-lrh/transfilm)")
        
        return app
    
    return app


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transfilm Web UI")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--host", type=str, help="主机地址")
    parser.add_argument("--port", type=int, help="端口号")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    # 初始化
    init_app(args.config)
    
    # 获取配置
    host = args.host or config.get("webui.host", "127.0.0.1")
    port = args.port or config.get("webui.port", 7860)
    share = args.share or config.get("webui.share", False)
    debug = args.debug or config.get("webui.debug", False)
    
    # 创建UI
    app = create_ui()
    
    # 启动
    logger.info(f"启动 Web UI: http://{host}:{port}")
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
    )


if __name__ == "__main__":
    main()
