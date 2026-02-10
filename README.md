# TransFilm - AI视频二次配音工具 / AI Video Dubbing Tool

## 项目简介 / Project Description

TransFilm 是一个AI驱动的视频二次配音工具，整合了 QwenLM/Qwen3-ASR、Qwen3-ASR Toolkit 和 QwenLM/Qwen3-TTS 三个项目，实现对视频文件的智能配音转换。

TransFilm is an AI-powered video dubbing tool that integrates QwenLM/Qwen3-ASR, Qwen3-ASR Toolkit, and QwenLM/Qwen3-TTS to enable intelligent audio dubbing conversion for video files.

## 核心功能 / Core Features

- ✅ **双运行模式** / **Dual Running Modes**
  - Web UI（基于Gradio） / Web UI (based on Gradio)
  - 命令行模式 / Command-line Interface

- ✅ **音视频处理** / **Audio-Video Processing**
  - 音视频分离 / Audio-video separation
  - 音频分块处理 / Chunked audio processing
  - 音视频重新拼接 / Audio-video reassembly

- ✅ **AI语音处理** / **AI Voice Processing**
  - 使用 Qwen3-ASR 进行语音识别 / Speech recognition with Qwen3-ASR
  - 音色特征提取和记录 / Voice characteristic extraction
  - 使用 Qwen3-TTS 生成新音频 / Audio generation with Qwen3-TTS

- ✅ **性能优化** / **Performance Optimization**
  - 低显存优化运行 / Low VRAM optimization
  - 批处理支持 / Batch processing support
  - 模型懒加载和卸载 / Model lazy loading and offloading

## 安装 / Installation

### 环境要求 / Requirements

- Python 3.8+
- CUDA 11.0+ (推荐 / recommended) or CPU
- FFmpeg

### 安装步骤 / Installation Steps

1. 克隆仓库 / Clone the repository:
```bash
git clone https://github.com/Olivia-lrh/transfilm.git
cd transfilm
```

2. 安装依赖 / Install dependencies:
```bash
pip install -r requirements.txt
```

3. 安装 FFmpeg（如果尚未安装）/ Install FFmpeg (if not already installed):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## 使用方法 / Usage

### Web UI 模式 / Web UI Mode

启动 Web 界面 / Launch Web interface:
```bash
python webui.py
```

然后在浏览器中打开显示的URL（通常是 http://127.0.0.1:7860）
Then open the displayed URL in your browser (usually http://127.0.0.1:7860)

### 命令行模式 / Command-line Mode

基本用法 / Basic usage:
```bash
python cli.py --input video.mp4 --output output.mp4
```

完整参数 / Full parameters:
```bash
python cli.py \
  --input input_video.mp4 \
  --output output_video.mp4 \
  --asr-model Qwen/Qwen3-ASR \
  --tts-model Qwen/Qwen3-TTS \
  --device cuda \
  --chunk-size 30 \
  --language zh
```

参数说明 / Parameter descriptions:
- `--input`: 输入视频文件路径 / Input video file path
- `--output`: 输出视频文件路径 / Output video file path
- `--asr-model`: ASR模型名称或路径 / ASR model name or path (default: Qwen/Qwen3-ASR)
- `--tts-model`: TTS模型名称或路径 / TTS model name or path (default: Qwen/Qwen3-TTS)
- `--device`: 运行设备 / Device to run on (cuda/cpu, default: cuda)
- `--chunk-size`: 音频分块大小（秒）/ Audio chunk size in seconds (default: 30)
- `--language`: 语言代码 / Language code (zh/en, default: zh)

## 项目结构 / Project Structure

```
transfilm/
├── README.md                   # 项目说明 / Project documentation
├── requirements.txt            # Python依赖 / Python dependencies
├── setup.py                    # 安装配置 / Installation config
├── cli.py                      # 命令行入口 / CLI entry point
├── webui.py                    # Web界面入口 / WebUI entry point
├── config.py                   # 配置文件 / Configuration
├── transfilm/                  # 核心包 / Core package
│   ├── __init__.py
│   ├── video_processor.py      # 视频处理 / Video processing
│   ├── audio_processor.py      # 音频处理 / Audio processing
│   ├── asr_engine.py          # ASR引擎 / ASR engine
│   ├── tts_engine.py          # TTS引擎 / TTS engine
│   ├── pipeline.py            # 处理流程 / Processing pipeline
│   └── utils.py               # 工具函数 / Utility functions
└── examples/                   # 示例 / Examples
    └── example_config.yaml     # 配置示例 / Config example
```

## 工作流程 / Workflow

1. **视频输入** / Video Input
   - 加载视频文件 / Load video file
   - 分离音频和视频轨道 / Separate audio and video tracks

2. **音频识别** / Audio Recognition
   - 将音频分块 / Chunk audio
   - 使用 Qwen3-ASR 识别语音 / Recognize speech with Qwen3-ASR
   - 提取音色特征 / Extract voice characteristics

3. **语音合成** / Speech Synthesis
   - 使用 Qwen3-TTS 生成新音频 / Generate new audio with Qwen3-TTS
   - 保持原始音色特征 / Maintain original voice characteristics
   - 对齐音频长度 / Align audio duration

4. **视频输出** / Video Output
   - 合并新音频和原视频 / Merge new audio with original video
   - 输出最终视频文件 / Output final video file

## 性能优化建议 / Performance Optimization Tips

### 低显存环境 / Low VRAM Environment

1. 使用 CPU 模式 / Use CPU mode:
```bash
python cli.py --input video.mp4 --output output.mp4 --device cpu
```

2. 减小批处理大小 / Reduce batch size:
```bash
python cli.py --input video.mp4 --output output.mp4 --chunk-size 15
```

3. 启用模型卸载 / Enable model offloading:
编辑 `config.py` 设置 `ENABLE_MODEL_OFFLOADING = True`
Edit `config.py` and set `ENABLE_MODEL_OFFLOADING = True`

## 依赖项目 / Dependencies

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - 语音识别 / Speech Recognition
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - 语音合成 / Text-to-Speech
- [Qwen3-ASR Toolkit](https://github.com/QwenLM/Qwen3-ASR) - ASR工具集 / ASR Toolkit

## 许可证 / License

MIT License - 详见 [LICENSE](LICENSE) 文件 / See [LICENSE](LICENSE) file for details

## 贡献 / Contributing

欢迎提交问题和拉取请求！
Issues and pull requests are welcome!

## 致谢 / Acknowledgments

感谢 Alibaba Cloud 的 Qwen 团队开发的优秀语音模型。
Thanks to the Qwen team at Alibaba Cloud for their excellent speech models.
