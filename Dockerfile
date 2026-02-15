FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置Python环境
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 复制requirements.txt
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 安装项目
RUN pip3 install -e .

# 创建输出和缓存目录
RUN mkdir -p /app/outputs /app/cache

# 暴露端口
EXPOSE 7860

# 设置启动命令
CMD ["python3", "webui.py", "--host", "0.0.0.0"]
