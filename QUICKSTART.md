# TransFilm Quick Start / 快速开始

## 1. 安装 / Installation

### 使用安装脚本 / Using Installation Script

```bash
chmod +x install.sh
./install.sh
```

### 手动安装 / Manual Installation

```bash
# 创建虚拟环境 / Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖 / Install dependencies
pip install -r requirements.txt

# 安装 FFmpeg / Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# 安装包 / Install package
pip install -e .
```

## 2. 验证安装 / Verify Installation

```bash
python test_structure.py
```

应该看到 "All Structure Tests Passed!" / Should see "All Structure Tests Passed!"

## 3. 使用 / Usage

### 方法 A: Web 界面（推荐新手）/ Web UI (Recommended for Beginners)

```bash
python webui.py
```

然后打开浏览器访问: http://127.0.0.1:7860

Then open browser at: http://127.0.0.1:7860

### 方法 B: 命令行 / Command Line

```bash
python cli.py --input your_video.mp4 --output dubbed_video.mp4
```

### 方法 C: Docker

```bash
# 启动服务 / Start service
docker-compose up transfilm

# 访问 / Access at: http://localhost:7860
```

## 4. 第一次使用注意事项 / First Time Usage Notes

⚠️ **重要 / Important:**

1. **模型下载 / Model Download**: 首次运行会自动下载 Qwen 模型（可能需要较长时间）
   First run will download Qwen models (may take some time)

2. **显存要求 / VRAM Requirements**: 
   - GPU 模式 / GPU mode: 至少 4GB 显存 / Minimum 4GB VRAM
   - CPU 模式 / CPU mode: 无 GPU 要求，但速度较慢 / No GPU required, but slower

3. **模型适配 / Model Adaptation**: 
   当前版本使用通用接口，需要根据实际 Qwen API 调整
   Current version uses generic interface, needs adaptation for actual Qwen API
   详见 / See: MODEL_INTEGRATION.md

## 5. 常用命令 / Common Commands

### 查看视频信息 / View Video Info

```bash
python cli.py --input video.mp4 --info
```

### 使用 CPU（低显存）/ Use CPU (Low VRAM)

```bash
python cli.py --input video.mp4 --output output.mp4 --device cpu
```

### 减小内存使用 / Reduce Memory Usage

```bash
python cli.py --input video.mp4 --output output.mp4 --chunk-size 15
```

### 显示详细日志 / Verbose Logging

```bash
python cli.py --input video.mp4 --output output.mp4 --verbose
```

## 6. 故障排除 / Troubleshooting

### 问题: 找不到 FFmpeg / Problem: FFmpeg not found

**解决 / Solution:**
```bash
# 检查 FFmpeg / Check FFmpeg
ffmpeg -version

# 如果未安装 / If not installed:
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg
```

### 问题: 内存不足 / Problem: Out of Memory

**解决 / Solution:**
```bash
# 使用 CPU 模式 / Use CPU mode
python cli.py --input video.mp4 --output output.mp4 --device cpu

# 或减小分块大小 / Or reduce chunk size
python cli.py --input video.mp4 --output output.mp4 --chunk-size 10
```

### 问题: 模型下载失败 / Problem: Model Download Failed

**解决 / Solution:**
- 检查网络连接 / Check network connection
- 使用国内镜像 / Use China mirror (if in China)
- 手动下载模型并配置本地路径 / Manually download and configure local path

## 7. 更多帮助 / More Help

- 📖 完整文档 / Full docs: [README.md](README.md)
- 📚 使用指南 / Usage guide: [USAGE.md](USAGE.md)
- 🤝 贡献指南 / Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- 🔧 模型集成 / Model integration: [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md)
- 📊 项目状态 / Project status: [PROJECT_STATUS.md](PROJECT_STATUS.md)

## 8. 示例工作流 / Example Workflow

```bash
# 1. 检查视频 / Check video
python cli.py --input my_video.mp4 --info

# 2. 处理视频 / Process video
python cli.py \
  --input my_video.mp4 \
  --output dubbed_video.mp4 \
  --language zh \
  --chunk-size 30 \
  --device cuda

# 3. 查看结果 / View result
# 使用任意视频播放器打开 dubbed_video.mp4
# Open dubbed_video.mp4 with any video player
```

## 9. 性能建议 / Performance Tips

### 快速模式 / Fast Mode (需要更多显存 / Needs more VRAM)
```bash
python cli.py --input video.mp4 --output output.mp4 --chunk-size 60 --device cuda
```

### 省内存模式 / Memory Saving Mode
```bash
python cli.py --input video.mp4 --output output.mp4 --chunk-size 15 --device cpu
```

### 平衡模式 / Balanced Mode (推荐 / Recommended)
```bash
python cli.py --input video.mp4 --output output.mp4 --chunk-size 30 --device cuda
```

---

**准备好了吗？开始你的第一个视频配音！**

**Ready? Start your first video dubbing!**

```bash
python webui.py
```

或 / Or

```bash
python cli.py --input your_video.mp4 --output result.mp4
```

🎉 祝你使用愉快！/ Enjoy using TransFilm!
