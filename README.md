# TransFilm - AI视频二次配音工具 / AI Video Dubbing Tool

## 项目简介 / Project Description

TransFilm 是一个AI驱动的视频二次配音工具，整合了多个 Qwen 模型，实现智能化的视频配音转换：

- **Qwen3-ASR-1.7B**: 高精度语音识别 / High-precision speech recognition
- **Qwen3-0.6B**: 智能翻译和分句 / Intelligent translation and segmentation
- **Qwen3-ForcedAligner-0.6B**: 精确时间戳对齐 / Precise timestamp alignment
- **Qwen3-TTS**: 音色克隆和语音合成 / Voice cloning and speech synthesis

TransFilm is an AI-powered video dubbing tool that integrates multiple Qwen models to enable intelligent audio dubbing conversion for video files.

## 核心功能 / Core Features

- ✅ **双运行模式** / **Dual Running Modes**
  - Web UI（基于Gradio） / Web UI (based on Gradio)
  - 命令行模式 / Command-line Interface

- ✅ **增强型AI处理流程** / **Enhanced AI Processing Pipeline**
  - 阶段1: 高精度语音识别 (Qwen3-ASR-1.7B) / Stage 1: High-precision ASR
  - 阶段2: 智能分句与翻译 (Qwen3-0.6B) / Stage 2: Intelligent segmentation & translation
  - 阶段3: 时间戳精确对齐 (Qwen3-ForcedAligner-0.6B) / Stage 3: Precise timestamp alignment
  - 阶段4: 音色克隆与合成 (Qwen3-TTS) / Stage 4: Voice cloning & synthesis
  - 阶段5: 时间同步音频拼接 / Stage 5: Time-synchronized audio assembly

- ✅ **智能翻译优化** / **Intelligent Translation Optimization**
  - 字符数量匹配，确保相似音频时长 / Character count matching for similar duration
  - 最短自然断句方式 / Shortest natural sentence breaks
  - 上下文感知翻译 / Context-aware translation

- ✅ **精确时长控制** / **Precise Duration Control**
  - 时间戳级别的音频对齐 / Timestamp-level audio alignment
  - TTS生成时长匹配 / TTS duration matching
  - 最终音频与源视频时长完全一致 / Final audio matches source duration exactly

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
  --input video.mp4 \
  --output output.mp4 \
  --asr-model Qwen/Qwen3-ASR-1.7B \
  --translation-model Qwen/Qwen3-0.6B \
  --forced-aligner-model Qwen/Qwen3-ForcedAligner-0.6B \
  --tts-model Qwen/Qwen3-TTS \
  --device cuda \
  --chunk-size 30 \
  --language zh \
  --target-language en
```

参数说明 / Parameter descriptions:
- `--input`: 输入视频文件路径 / Input video file path
- `--output`: 输出视频文件路径 / Output video file path
- `--asr-model`: ASR模型名称或路径 / ASR model name or path (default: Qwen/Qwen3-ASR-1.7B)
- `--translation-model`: 翻译模型名称或路径 / Translation model name or path (default: Qwen/Qwen3-0.6B)
- `--forced-aligner-model`: 时间对齐模型名称或路径 / Aligner model name or path (default: Qwen/Qwen3-ForcedAligner-0.6B)
- `--tts-model`: TTS模型名称或路径 / TTS model name or path (default: Qwen/Qwen3-TTS)
- `--device`: 运行设备 / Device to run on (cuda/cpu, default: cuda)
- `--chunk-size`: 音频分块大小（秒）/ Audio chunk size in seconds (default: 30)
- `--language`: 源语言代码 / Source language code (zh/en, default: zh)
- `--target-language`: 目标语言代码 / Target language code (zh/en, default: en)

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

### 增强型5阶段处理流程 / Enhanced 5-Stage Processing Pipeline

1. **阶段1: 语音识别** / **Stage 1: Speech Recognition**
   - 使用 Qwen3-ASR-1.7B 提取完整音频内容 / Extract full audio content with Qwen3-ASR-1.7B
   - 高精度识别，支持多语言 / High-precision recognition, multi-language support

2. **阶段2: 智能分句与翻译** / **Stage 2: Intelligent Segmentation & Translation**
   - 使用 Qwen3-0.6B 进行智能分句 / Intelligent segmentation with Qwen3-0.6B
   - 采用最短说话方式断句 / Shortest natural speaking breaks
   - 翻译时考虑字符数匹配，确保相似时长 / Translation considers character count for similar duration

3. **阶段3: 时间戳对齐** / **Stage 3: Timestamp Alignment**
   - 使用 Qwen3-ForcedAligner-0.6B 生成精确时间戳 / Generate precise timestamps with Qwen3-ForcedAligner-0.6B
   - 为每个句子匹配对应的时间段 / Match time segments to each sentence
   - 微调对齐以匹配总时长 / Fine-tune alignment to match total duration

4. **阶段4: 音色克隆与语音合成** / **Stage 4: Voice Cloning & Synthesis**
   - 提取原始音频的音色特征 / Extract voice characteristics from original audio
   - 使用 Qwen3-TTS 生成新音频 / Generate new audio with Qwen3-TTS
   - 克隆原始音色 / Clone original voice
   - 精确匹配时间戳时长 / Precisely match timestamp durations

5. **阶段5: 音频拼接与视频合成** / **Stage 5: Audio Assembly & Video Synthesis**
   - 按时间戳拼接音频段 / Concatenate audio segments by timestamps
   - 确保总时长与源视频一致 / Ensure total duration matches source video
   - 合成最终带新音轨的视频 / Synthesize final video with new audio track

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

- [QwenLM/Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) - 高精度语音识别 / High-precision Speech Recognition
- [QwenLM/Qwen3-0.6B](https://github.com/QwenLM/Qwen) - 智能翻译 / Intelligent Translation
- [QwenLM/Qwen3-ForcedAligner-0.6B](https://github.com/QwenLM/Qwen3-ASR) - 时间戳对齐 / Timestamp Alignment
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - 语音合成 / Text-to-Speech

## 许可证 / License

MIT License - 详见 [LICENSE](LICENSE) 文件 / See [LICENSE](LICENSE) file for details

## 贡献 / Contributing

欢迎提交问题和拉取请求！
Issues and pull requests are welcome!

## 致谢 / Acknowledgments

感谢 Alibaba Cloud 的 Qwen 团队开发的优秀语音模型。
Thanks to the Qwen team at Alibaba Cloud for their excellent speech models.
