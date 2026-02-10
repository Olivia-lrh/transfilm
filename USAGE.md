# TransFilm 使用指南 / Usage Guide

## 快速开始 / Quick Start

### 方法 1: 使用命令行 / Method 1: Using CLI

最简单的用法 / Simplest usage:

```bash
python cli.py --input your_video.mp4 --output dubbed_video.mp4
```

### 方法 2: 使用 Web 界面 / Method 2: Using Web UI

启动 Web 界面 / Start web interface:

```bash
python webui.py
```

然后在浏览器中打开 http://127.0.0.1:7860

Then open http://127.0.0.1:7860 in your browser

## 详细使用说明 / Detailed Instructions

### 命令行参数 / Command Line Arguments

#### 基本参数 / Basic Arguments

- `--input`, `-i`: 输入视频文件路径 / Input video file path (required)
- `--output`, `-o`: 输出视频文件路径 / Output video file path (required)

#### 模型配置 / Model Configuration

```bash
python cli.py \
  --input video.mp4 \
  --output output.mp4 \
  --asr-model Qwen/Qwen3-ASR \
  --tts-model Qwen/Qwen3-TTS
```

可以使用本地模型路径 / Can use local model paths:

```bash
python cli.py \
  --input video.mp4 \
  --output output.mp4 \
  --asr-model /path/to/local/asr/model \
  --tts-model /path/to/local/tts/model
```

#### 设备选择 / Device Selection

使用 GPU / Use GPU:
```bash
python cli.py --input video.mp4 --output output.mp4 --device cuda
```

使用 CPU（低显存环境）/ Use CPU (low VRAM):
```bash
python cli.py --input video.mp4 --output output.mp4 --device cpu
```

#### 音频分块设置 / Audio Chunking Settings

调整分块大小以优化内存使用 / Adjust chunk size to optimize memory:

```bash
# 小分块（使用更少内存）/ Smaller chunks (less memory)
python cli.py --input video.mp4 --output output.mp4 --chunk-size 15

# 大分块（更快处理）/ Larger chunks (faster processing)
python cli.py --input video.mp4 --output output.mp4 --chunk-size 60
```

#### 语言设置 / Language Settings

```bash
# 中文 / Chinese
python cli.py --input video.mp4 --output output.mp4 --language zh

# 英文 / English
python cli.py --input video.mp4 --output output.mp4 --language en
```

#### 其他选项 / Other Options

显示视频信息 / Show video info:
```bash
python cli.py --input video.mp4 --info
```

保留临时文件（用于调试）/ Keep temp files (for debugging):
```bash
python cli.py --input video.mp4 --output output.mp4 --keep-temp
```

启用详细日志 / Enable verbose logging:
```bash
python cli.py --input video.mp4 --output output.mp4 --verbose
```

## 使用场景 / Use Cases

### 场景 1: 标准配音 / Scenario 1: Standard Dubbing

对普通视频进行配音转换 / Standard dubbing for regular videos:

```bash
python cli.py \
  --input original.mp4 \
  --output dubbed.mp4 \
  --language zh \
  --chunk-size 30
```

### 场景 2: 低显存环境 / Scenario 2: Low VRAM Environment

在显存有限的环境中运行 / Run in limited VRAM environment:

```bash
python cli.py \
  --input video.mp4 \
  --output output.mp4 \
  --device cpu \
  --chunk-size 15
```

或者启用量化 / Or enable quantization (edit config.py):
```python
USE_8BIT = True  # 或 USE_4BIT = True / Or USE_4BIT = True
```

### 场景 3: 批量处理 / Scenario 3: Batch Processing

批量处理多个视频 / Process multiple videos:

```bash
#!/bin/bash
for video in *.mp4; do
  echo "Processing $video"
  python cli.py --input "$video" --output "dubbed_$video"
done
```

### 场景 4: 自定义配置 / Scenario 4: Custom Configuration

使用自定义配置文件 / Using custom configuration:

1. 复制配置示例 / Copy config example:
```bash
cp examples/example_config.yaml my_config.yaml
```

2. 编辑配置 / Edit configuration:
```yaml
processing:
  device: "cuda"
  chunk_size: 20
  language: "zh"
```

3. 修改 config.py 加载自定义配置 / Modify config.py to load custom config

## Python API 使用 / Python API Usage

### 基本使用 / Basic Usage

```python
from transfilm import VideoDubbingPipeline

# 创建管道 / Create pipeline
pipeline = VideoDubbingPipeline(
    asr_model="Qwen/Qwen3-ASR",
    tts_model="Qwen/Qwen3-TTS",
    device="cuda",
    chunk_size=30,
    language="zh"
)

# 处理视频 / Process video
output_path = pipeline.process_video(
    input_video="input.mp4",
    output_video="output.mp4"
)

print(f"Dubbed video saved to: {output_path}")
```

### 带进度回调 / With Progress Callback

```python
from transfilm import VideoDubbingPipeline

def my_progress_callback(message, progress):
    print(f"[{progress:.1f}%] {message}")

pipeline = VideoDubbingPipeline(
    progress_callback=my_progress_callback
)

pipeline.process_video("input.mp4", "output.mp4")
```

### 仅使用特定模块 / Using Specific Modules

#### 视频处理 / Video Processing

```python
from transfilm import VideoProcessor

processor = VideoProcessor()

# 提取音频 / Extract audio
processor.extract_audio("video.mp4", "audio.wav")

# 合并音视频 / Combine audio and video
processor.combine_audio_video(
    "video_only.mp4",
    "new_audio.wav",
    "final.mp4"
)
```

#### 音频处理 / Audio Processing

```python
from transfilm import AudioProcessor
import numpy as np

processor = AudioProcessor(sample_rate=16000)

# 加载音频 / Load audio
audio, sr = processor.load_audio("audio.wav")

# 分块 / Chunk audio
chunks = processor.chunk_audio(audio, chunk_size_seconds=30)

# 提取特征 / Extract features
features = processor.extract_voice_features(audio)
```

## 性能优化 / Performance Optimization

### 内存优化 / Memory Optimization

1. **减小分块大小** / Reduce chunk size:
   - 更小的分块使用更少内存 / Smaller chunks use less memory
   - 推荐: 15-30秒 / Recommended: 15-30 seconds

2. **启用模型卸载** / Enable model offloading:
   ```python
   # 在 config.py 中 / In config.py
   ENABLE_MODEL_OFFLOADING = True
   ```

3. **使用量化** / Use quantization:
   ```python
   # 在 config.py 中 / In config.py
   USE_8BIT = True  # 8-bit quantization
   # 或 / Or
   USE_4BIT = True  # 4-bit quantization (更激进 / more aggressive)
   ```

4. **限制 GPU 内存** / Limit GPU memory:
   ```python
   # 在 config.py 中 / In config.py
   MAX_MEMORY_PER_GPU = "8GB"
   ```

### 速度优化 / Speed Optimization

1. **使用更大的分块** / Use larger chunks:
   ```bash
   python cli.py --chunk-size 60  # 更快但需要更多内存 / Faster but needs more memory
   ```

2. **使用 GPU** / Use GPU:
   ```bash
   python cli.py --device cuda
   ```

3. **使用 float16** / Use float16:
   ```python
   # 在 config.py 中 / In config.py
   DEFAULT_DTYPE = "float16"  # 比 float32 快 / Faster than float32
   ```

## 故障排除 / Troubleshooting

### 常见问题 / Common Issues

#### 1. 内存不足 / Out of Memory

**症状** / Symptoms:
```
RuntimeError: CUDA out of memory
```

**解决方案** / Solutions:
- 减小分块大小: `--chunk-size 15`
- 使用 CPU: `--device cpu`
- 启用量化: 设置 `USE_8BIT = True`

#### 2. FFmpeg 未找到 / FFmpeg Not Found

**症状** / Symptoms:
```
FileNotFoundError: ffmpeg not found
```

**解决方案** / Solutions:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: 下载并安装 / Download and install from ffmpeg.org

#### 3. 模型下载失败 / Model Download Failed

**症状** / Symptoms:
```
Error downloading model from HuggingFace
```

**解决方案** / Solutions:
- 检查网络连接 / Check network connection
- 使用镜像站 / Use mirror site
- 手动下载并指定本地路径 / Download manually and specify local path

#### 4. 音视频不同步 / Audio-Video Desync

**症状** / Symptoms: 音频和视频不匹配 / Audio and video don't match

**解决方案** / Solutions:
- 检查输入视频质量 / Check input video quality
- 尝试不同的分块大小 / Try different chunk sizes
- 使用 `--keep-temp` 检查中间文件 / Use `--keep-temp` to inspect intermediate files

## 更多资源 / More Resources

- 📖 完整文档 / Full documentation: [README.md](README.md)
- 🤝 贡献指南 / Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- 🐛 问题报告 / Issue reporting: [GitHub Issues](https://github.com/Olivia-lrh/transfilm/issues)
- 💬 讨论 / Discussions: [GitHub Discussions](https://github.com/Olivia-lrh/transfilm/discussions)

---

如有其他问题，请在 GitHub 上提出 Issue。

For other questions, please open an issue on GitHub.
