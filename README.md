# Transfilm - AI视频配音系统

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)

生产级AI视频翻译和配音管道，使用Qwen3-ASR、Qwen3-TTS和Qwen3

</div>

## 📋 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [硬件要求](#硬件要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用文档](#使用文档)
- [API文档](#api文档)
- [常见问题](#常见问题)
- [许可证](#许可证)

## ✨ 功能特性

- 🎯 **端到端管道**: 完整的6阶段视频配音流程
  - 视频预处理（音视频分离）
  - ASR语音识别（带时间戳）
  - 高质量文本翻译
  - TTS语音合成
  - 音频组装（时长匹配）
  - 视频合并

- 🤖 **官方API**: 使用官方模型API
  - **Qwen3-ASR**: 准确的语音识别和强制对齐
  - **Qwen3**: 高质量多语言翻译 (支持119种语言)
  - **Qwen3-TTS**: 自然的语音合成（支持音色克隆）

- 🎨 **双TTS模式**:
  - **自定义音色**: 使用预设说话人（Vivian、Ryan等）
  - **音色克隆**: 克隆原视频中的音色

- 🚀 **性能优化**:
  - 低显存优化（支持8GB+ VRAM）
  - 统一Qwen生态系统（ASR、TTS、翻译全部使用Qwen）
  - 按需模型加载/卸载
  - Flash Attention 2支持
  - 批处理优化

- 💻 **多种接口**:
  - 命令行界面（CLI）
  - Gradio Web UI
  - Python API

- 🐳 **部署友好**:
  - Docker支持
  - 一键安装脚本
  - 灵活的配置系统

## 🏗️ 系统架构

```
输入视频
   ↓
[1. 视频预处理] → 提取音频
   ↓
[2. ASR识别] → 转录文本 + 时间戳
   ↓
[3. 文本翻译] → 翻译到目标语言
   ↓
[4. TTS合成] → 生成新语音
   ↓
[5. 音频组装] → 基于时间戳拼接
   ↓
[6. 视频合并] → 合并新音频和原视频
   ↓
输出视频
```

## 💾 硬件要求

### 推荐配置
- **GPU**: NVIDIA GPU, 12GB+ VRAM (RTX 4070, A4000等)
- **CPU**: 8核+
- **内存**: 32GB+
- **存储**: 100GB+ (模型约40GB)

### 最低配置 (使用Qwen3-4B)
- **GPU**: NVIDIA GPU, 8GB VRAM (RTX 3060等)
- **CPU**: 4核+
- **内存**: 16GB+
- **存储**: 100GB+

### 超低显存配置 (使用Qwen3-1.7B/0.6B)
- **GPU**: NVIDIA GPU, 4GB+ VRAM
- **CPU**: 4核+
- **内存**: 16GB+
- **存储**: 80GB+

### CPU模式
- 支持纯CPU运行，但速度较慢（不推荐）
- **内存**: 32GB+

## 📦 安装指南

### 方法1: 一键安装脚本（推荐）

```bash
# 克隆仓库
git clone https://github.com/Olivia-lrh/transfilm.git
cd transfilm

# 运行安装脚本
chmod +x install.sh
./install.sh
```

### 方法2: 手动安装

#### 前置要求
- Python 3.8+
- FFmpeg
- CUDA 11.8+ (GPU模式)

```bash
# 1. 克隆仓库
git clone https://github.com/Olivia-lrh/transfilm.git
cd transfilm

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装项目
pip install -e .

# 5. 下载模型
python cli.py download --models all
```

### 方法3: Docker安装

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 访问Web UI
# http://localhost:7860
```

### 安装FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
从 [FFmpeg官网](https://ffmpeg.org/download.html) 下载并添加到PATH

## 🚀 快速开始

### 命令行使用

```bash
# 基本用法
python cli.py dub input.mp4 --output output.mp4 --target-lang Chinese

# 指定源语言
python cli.py dub input.mp4 -o output.mp4 --source-lang English --target-lang Chinese

# 使用特定说话人
python cli.py dub input.mp4 -o output.mp4 --target-lang Chinese --speaker Vivian

# 查看可用说话人
python cli.py info --speakers

# 查看支持的语言
python cli.py info --languages
```

### Web UI使用

```bash
# 启动Web UI
python webui.py

# 自定义端口
python webui.py --port 8080

# 创建公开链接
python webui.py --share
```

然后在浏览器中打开 http://localhost:7860

### Python API使用

```python
from transfilm import Config, Pipeline

# 加载配置
config = Config("config.yaml")

# 创建管道
pipeline = Pipeline(config)

# 处理视频
output = pipeline.run(
    input_video="input.mp4",
    output_video="output.mp4",
    target_language="Chinese",
    source_language=None,  # 自动检测
    tts_speaker="Vivian",
)

print(f"处理完成: {output}")
```

## ⚙️ 配置说明

### 配置文件结构

配置文件 `config.yaml` 包含以下部分：

```yaml
models:
  asr:
    model_name: "Qwen/Qwen3-ASR-1.7B"
    device: "cuda:0"
    dtype: "bfloat16"
  
  tts:
    mode: "custom_voice"  # custom_voice 或 voice_clone
    custom_voice_model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    device: "cuda:0"
  
  translation:
    model_name: "Qwen/Qwen3-4B"  # 或 Qwen/Qwen3-1.7B, Qwen/Qwen3-0.6B
    device: "cuda:0"

pipeline:
  sequential_model_loading: true  # 按需加载模型
  clear_cache_between_stages: true  # 清理GPU缓存

audio:
  sample_rate: 16000
  time_stretch_method: "librosa"

video:
  video_codec: "libx264"
  crf: 23
```

### 环境变量

可以通过环境变量覆盖配置：

```bash
# 设置设备
export TRANSFILM_MODELS_ASR_DEVICE=cuda:0
export TRANSFILM_MODELS_TTS_DEVICE=cuda:1

# 设置数据类型
export TRANSFILM_MODELS_ASR_DTYPE=float16
```

### 低显存优化

对于8GB显存的GPU：

```yaml
models:
  asr:
    dtype: "float16"  # 使用float16
    max_inference_batch_size: 16  # 减小批次大小
  
  tts:
    dtype: "float16"
    attn_implementation: "sdpa"  # 使用SDPA而非Flash Attention
  
  translation:
    model_name: "Qwen/Qwen3-1.7B"  # 使用较小的模型
    dtype: "float16"

pipeline:
  sequential_model_loading: true  # 必须启用
  clear_cache_between_stages: true  # 必须启用
```

对于4GB显存的GPU：

```yaml
models:
  translation:
    model_name: "Qwen/Qwen3-0.6B"  # 使用超小模型
```

## 📖 使用文档

### 模型下载

```bash
# 下载所有模型
python cli.py download --models all

# 下载特定模型
python cli.py download --models asr tts_custom translation

# 从ModelScope下载（国内更快）
python cli.py download --models all --source modelscope
```

### 支持的语言

**ASR（语音识别）**:
- Chinese（中文）
- English（英语）
- Japanese（日语）
- Korean（韩语）

**TTS（语音合成）**:
- Chinese（中文）
- English（英语）

### 可用说话人

**自定义音色模式**:
- **Vivian**: 女声-活泼（中文）
- **Serena**: 女声-温柔（中文）
- **Ryan**: 男声-沉稳（中文）
- **Emma**: 女声-优雅（英语）
- **Jack**: 男声-阳光（英语）

**音色克隆模式**:
自动从原视频中提取和克隆音色

### 支持的视频格式

- MP4
- MKV
- AVI
- MOV
- WebM

## 🔧 API文档

### Pipeline类

```python
class Pipeline:
    def __init__(self, config: Optional[Config] = None)
    
    def run(
        self,
        input_video: str,
        output_video: str,
        target_language: str = "Chinese",
        source_language: Optional[str] = None,
        tts_speaker: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ) -> str
```

### ASREngine类

```python
class ASREngine:
    def transcribe(
        self,
        audio: Union[str, np.ndarray, tuple],
        language: Optional[str] = None,
        return_time_stamps: bool = True,
    ) -> List[ASRResult]
```

### TranslationEngine类

```python
class TranslationEngine:
    def translate(
        self,
        text: str,
        target_language: str = "Chinese",
        source_language: Optional[str] = None,
    ) -> str
```

### TTSEngine类

```python
class TTSEngine:
    def synthesize(
        self,
        text: str,
        language: str = "Chinese",
        **kwargs,
    ) -> Tuple[np.ndarray, int]
```

## ❓ 常见问题

### 1. 显存不足错误

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 启用按需模型加载: `pipeline.sequential_model_loading: true`
- 使用float16: `dtype: "float16"`
- 减小批次大小: `max_inference_batch_size: 16`

### 2. FFmpeg未找到

**问题**: `FileNotFoundError: ffmpeg`

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并添加到PATH
```

### 3. 模型下载速度慢

**问题**: HuggingFace下载缓慢

**解决方案**:
```bash
# 使用ModelScope（国内更快）
python cli.py download --models all --source modelscope
```

### 4. 音频不同步

**问题**: 翻译后的音频与视频不同步

**解决方案**:
- 检查时间戳是否正确: ASR必须启用 `return_time_stamps=True`
- 调整时间拉伸方法: `audio.time_stretch_method: "librosa"`

### 5. TTS音质不佳

**问题**: 合成的语音不自然

**解决方案**:
- 尝试不同的说话人
- 使用音色克隆模式: `tts.mode: "voice_clone"`
- 添加语气指令: `tts.default_instruct: "用专业的语气说"`

## 🛠️ 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black transfilm/
```

### 类型检查

```bash
mypy transfilm/
```

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

本项目使用以下开源项目：

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - 语音识别
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - 语音合成
- [Qwen3](https://github.com/QwenLM/Qwen3) - 文本翻译
- [FFmpeg](https://ffmpeg.org/) - 音视频处理
- [Gradio](https://gradio.app/) - Web UI

## 📧 联系方式

- GitHub Issues: [提交问题](https://github.com/Olivia-lrh/transfilm/issues)
- 作者: Olivia-lrh

---

<div align="center">
如果这个项目对您有帮助，请给个⭐️吧！
</div>
