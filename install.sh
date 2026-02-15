#!/bin/bash
# Transfilm 一键安装脚本

set -e

echo "========================================="
echo "  Transfilm AI视频配音系统 安装脚本"
echo "========================================="
echo ""

# 检查Python版本
echo "检查Python版本..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python 3"
    echo "请先安装Python 3.8或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Python版本: $PYTHON_VERSION"

# 检查FFmpeg
echo "检查FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "警告: 未找到FFmpeg"
    echo "FFmpeg是必需的，请先安装："
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  Windows: 从 https://ffmpeg.org/download.html 下载"
    exit 1
fi

echo "FFmpeg版本: $(ffmpeg -version | head -n1)"

# 创建虚拟环境（可选）
read -p "是否创建Python虚拟环境? (推荐) [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "虚拟环境已创建并激活"
fi

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 安装项目
echo "安装Transfilm..."
pip install -e .

# 创建必要的目录
echo "创建目录..."
mkdir -p cache outputs models

# 下载模型（可选）
read -p "是否立即下载所需模型? (需要较大存储空间和时间) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始下载模型..."
    python3 cli.py download --models all
fi

echo ""
echo "========================================="
echo "  安装完成！"
echo "========================================="
echo ""
echo "使用方法："
echo "  1. 命令行: python cli.py dub input.mp4 --output output.mp4 --target-lang Chinese"
echo "  2. Web UI: python webui.py"
echo ""
echo "更多帮助: python cli.py --help"
echo ""
