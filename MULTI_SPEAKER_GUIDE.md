# TransFilm Multi-Speaker Voice Cloning Guide

## 概述 / Overview

TransFilm现已支持多说话人识别和音色克隆功能。本指南介绍如何使用这些增强功能。

TransFilm now supports multi-speaker identification and voice cloning. This guide explains how to use these enhanced features.

## 新增功能 / New Features

### 1. 音频声音分析 / Audio Voice Analysis

系统自动分析音频，识别不同的说话人。

The system automatically analyzes audio to identify different speakers.

**特性 / Features:**
- 自动说话人分离（Speaker Diarization）
- 说话人数量自动检测（1-10人）
- 每个说话人的时间戳标记

**实现 / Implementation:**
- 使用 `SpeakerAnalyzer` 模块
- 基于音频特征聚类识别说话人
- 为每个说话人段分配置信度

### 2. LLM智能断句和说话人判定 / LLM-Based Segmentation and Speaker Assignment

使用大语言模型(LLM)结合音频分析结果，进行智能断句和说话人识别。

Uses Large Language Model (LLM) combined with audio analysis for intelligent sentence segmentation and speaker identification.

**特性 / Features:**
- 基于语义的自然断句
- 上下文感知的说话人分配
- 对话结构理解

**实现 / Implementation:**
- 使用 `LLMEngine` 模块
- Qwen2.5-7B-Instruct 或更大模型
- 可选：禁用LLM使用规则方法

### 3. LLM翻译长度调整 / LLM Translation Length Adjustment

根据时间戳自动调整翻译内容，确保TTS输出的音频长度与原始音频匹配。

Automatically adjusts translation based on timestamps to ensure TTS output matches original audio length.

**特性 / Features:**
- 智能缩短或扩展翻译
- 保持语义准确性
- 自适应时长控制

**实现 / Implementation:**
- 当实际时长与目标时长差异>15%时触发
- LLM生成调整后的翻译
- 重新合成音频

### 4. 5秒音色样本提取 / 5-Second Voice Sample Extraction

为每个识别的说话人提取5秒钟的音频样本，用于音色克隆。

Extracts 5-second audio samples for each identified speaker for voice cloning.

**特性 / Features:**
- 自动选择最佳片段
- 优先选择长且高置信度的片段
- 支持多片段组合

**实现 / Implementation:**
- 在说话人分析后自动执行
- 保存到临时目录供调试
- 传递给TTS引擎用于克隆

### 5. 多说话人TTS音色克隆 / Multi-Speaker TTS Voice Cloning

使用提取的音色样本，为每个说话人生成对应的配音。

Generates dubbing for each speaker using extracted voice samples.

**特性 / Features:**
- 基于真实音色的克隆
- 保持原始音色特征
- 支持任意数量的说话人

**实现 / Implementation:**
- TTS引擎的 `synthesize_multi_speaker` 方法
- 每个片段使用对应说话人的音色样本
- 精确时长匹配

### 6. 时间戳精确拼接 / Timestamp-Based Accurate Concatenation

按时间戳顺序拼接音频，确保与原视频完全同步。

Concatenates audio segments by timestamps, ensuring perfect sync with original video.

**特性 / Features:**
- 精确到采样点的时间戳对齐
- 自动填充静音间隙
- 总时长验证

**实现 / Implementation:**
- 使用 `AudioProcessor.concatenate_audio_with_timestamps`
- 确保总时长与原始音频一致（±0.01秒）
- 处理重叠和间隙

## 使用方法 / Usage

### 命令行 / Command Line

#### 基本使用（启用所有功能）/ Basic Usage (All Features Enabled)

```bash
python cli.py --input input.mp4 --output output.mp4
```

这将启用：
- ✅ 说话人识别
- ✅ LLM智能断句
- ✅ LLM翻译调整
- ✅ 多说话人音色克隆
- ✅ 时间戳拼接

#### 单说话人模式 / Single Speaker Mode

如果视频只有一个说话人，可以禁用说话人识别以提高速度：

```bash
python cli.py --input input.mp4 --output output.mp4 \
  --disable-speaker-diarization
```

#### 规则方法模式 / Rule-Based Mode

如果想节省GPU内存，可以禁用LLM功能：

```bash
python cli.py --input input.mp4 --output output.mp4 \
  --disable-llm-features
```

#### 自定义模型 / Custom Models

```bash
python cli.py --input input.mp4 --output output.mp4 \
  --llm-model Qwen/Qwen2.5-14B-Instruct \
  --asr-model Qwen/Qwen3-ASR-1.7B \
  --tts-model Qwen/Qwen3-TTS
```

#### 完整参数示例 / Full Parameter Example

```bash
python cli.py \
  --input chinese_interview.mp4 \
  --output english_interview.mp4 \
  --language zh \
  --target-language en \
  --device cuda \
  --enable-speaker-diarization \
  --enable-llm-features \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --keep-temp \
  --verbose
```

### Python API

```python
from transfilm import VideoDubbingPipeline

# 创建管道 / Create pipeline
pipeline = VideoDubbingPipeline(
    source_language="zh",
    target_language="en",
    enable_speaker_diarization=True,  # 启用说话人识别
    enable_llm_features=True,          # 启用LLM功能
    llm_model="Qwen/Qwen2.5-7B-Instruct"
)

# 处理视频 / Process video
pipeline.process_video(
    input_video="input.mp4",
    output_video="output.mp4",
    keep_temp=True  # 保留临时文件用于调试
)
```

## 工作流程 / Workflow

### 8阶段处理流程 / 8-Stage Processing Pipeline

1. **阶段1 (0-15%)**: 音频提取和说话人分析
   - 从视频提取音频
   - 识别说话人（1-10人）
   - 提取每个说话人的5秒音色样本

2. **阶段2 (15-30%)**: ASR转录
   - 使用Qwen3-ASR-1.7B进行完整转录

3. **阶段3 (30-45%)**: LLM断句和说话人分配
   - LLM智能断句
   - 根据音频分析分配说话人

4. **阶段4 (45-60%)**: 翻译
   - 翻译每个片段
   - 准备长度调整

5. **阶段5 (60-70%)**: 时间戳对齐
   - 使用Forced Aligner生成精确时间戳
   - 微调以匹配总时长

6. **阶段6 (70-90%)**: 多说话人TTS
   - 使用说话人特定音色样本合成
   - 应用LLM长度调整（如需要）

7. **阶段7 (90-95%)**: 音频拼接
   - 按时间戳拼接片段
   - 确保总时长匹配

8. **阶段8 (95-100%)**: 视频组装
   - 合并新音频和视频
   - 保存最终输出

## 配置选项 / Configuration Options

### config.py 中的新设置 / New Settings in config.py

```python
# LLM模型
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 说话人分析设置
ENABLE_SPEAKER_DIARIZATION = True
MIN_SPEAKERS = 1
MAX_SPEAKERS = 10
VOICE_SAMPLE_DURATION = 5.0  # 秒

# LLM增强功能
ENABLE_LLM_SEGMENTATION = True
ENABLE_LLM_SPEAKER_ANALYSIS = True
ENABLE_LLM_LENGTH_ADJUSTMENT = True
```

## 性能优化 / Performance Optimization

### GPU内存使用 / GPU Memory Usage

多说话人功能需要更多内存。推荐配置：

**最低配置 / Minimum:**
- GPU: 8GB VRAM
- CPU: 16GB RAM
- 使用8位量化

**推荐配置 / Recommended:**
- GPU: 24GB VRAM (RTX 3090/4090)
- CPU: 32GB RAM
- 使用FP16

**优化建议 / Optimization Tips:**

1. 启用模型卸载：
```python
config.ENABLE_MODEL_OFFLOADING = True
```

2. 使用量化：
```python
config.USE_8BIT = True  # 或 USE_4BIT = True
```

3. 禁用不需要的功能：
```bash
# 单说话人视频
--disable-speaker-diarization

# 简单场景
--disable-llm-features
```

## 故障排除 / Troubleshooting

### 常见问题 / Common Issues

**1. 说话人识别不准确**

解决方案：
- 确保音频质量良好
- 调整 `MIN_SPEAKERS` 和 `MAX_SPEAKERS`
- 检查是否有背景噪音

**2. GPU内存不足**

解决方案：
```bash
# 使用量化
export USE_8BIT=1

# 或禁用部分功能
--disable-llm-features
--disable-speaker-diarization

# 或使用CPU
--device cpu
```

**3. 音频时长不匹配**

解决方案：
- 启用LLM长度调整
- 检查 `DURATION_TOLERANCE` 设置
- 查看日志了解具体片段的问题

**4. LLM功能无法使用**

解决方案：
- 确认已安装transformers库
- 检查LLM模型是否已下载
- 查看日志中的错误信息

## 高级用法 / Advanced Usage

### 自定义说话人数量 / Custom Speaker Count

```python
from transfilm import VideoDubbingPipeline

pipeline = VideoDubbingPipeline(
    enable_speaker_diarization=True
)

# 修改说话人范围
import config
config.MIN_SPEAKERS = 2  # 至少2个说话人
config.MAX_SPEAKERS = 5  # 最多5个说话人

pipeline.process_video("input.mp4", "output.mp4")
```

### 调试模式 / Debug Mode

保留临时文件以检查中间结果：

```bash
python cli.py --input input.mp4 --output output.mp4 \
  --keep-temp --verbose
```

临时文件包括：
- `original_audio.wav` - 原始音频
- `voice_sample_SPEAKER_XX.wav` - 每个说话人的音色样本
- `dubbed_audio.wav` - 最终合成音频

## 最佳实践 / Best Practices

1. **视频准备**
   - 使用清晰的音频
   - 避免背景音乐过强
   - 说话人声音应有明显区别

2. **参数选择**
   - 对话视频：启用所有功能
   - 单人讲座：禁用说话人识别
   - 快速测试：禁用LLM功能

3. **质量检查**
   - 使用 `--keep-temp` 检查中间结果
   - 验证说话人识别准确性
   - 检查音频时长匹配

4. **性能优化**
   - 根据GPU内存选择模型大小
   - 使用量化减少内存占用
   - 考虑批处理多个视频

## 示例场景 / Example Scenarios

### 场景1：双人对话访谈 / Scenario 1: Two-Person Interview

```bash
python cli.py \
  --input interview.mp4 \
  --output interview_en.mp4 \
  --language zh \
  --target-language en \
  --enable-speaker-diarization \
  --enable-llm-features
```

### 场景2：多人会议 / Scenario 2: Multi-Person Meeting

```python
import config
config.MAX_SPEAKERS = 8

from transfilm import VideoDubbingPipeline

pipeline = VideoDubbingPipeline(
    source_language="zh",
    target_language="en",
    enable_speaker_diarization=True,
    enable_llm_features=True
)

pipeline.process_video("meeting.mp4", "meeting_en.mp4")
```

### 场景3：单人演讲 / Scenario 3: Single Speaker Presentation

```bash
python cli.py \
  --input presentation.mp4 \
  --output presentation_en.mp4 \
  --language zh \
  --target-language en \
  --disable-speaker-diarization
```

## 技术细节 / Technical Details

### 说话人分析算法 / Speaker Analysis Algorithm

1. 音频特征提取
2. 语音活动检测(VAD)
3. 说话人嵌入提取
4. 聚类分析
5. 后处理优化

### LLM断句策略 / LLM Segmentation Strategy

1. 全文语义理解
2. 自然断句点识别
3. 说话人上下文分析
4. 时长约束考虑

### 音色克隆技术 / Voice Cloning Technology

1. 提取5秒音色样本
2. 声学特征编码
3. TTS模型适配
4. 音质优化处理

## 参考资料 / References

- [Qwen3-ASR Documentation](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-TTS Documentation](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen2.5 Documentation](https://github.com/QwenLM/Qwen2.5)
- [Speaker Diarization Basics](https://en.wikipedia.org/wiki/Speaker_diarisation)

## 更新日志 / Changelog

### v0.2.0 (Current)
- ✅ 添加多说话人识别和分析
- ✅ 集成LLM智能断句
- ✅ 实现LLM翻译长度调整
- ✅ 支持多说话人音色克隆
- ✅ 优化时间戳对齐和拼接

### v0.1.0
- 基础视频配音功能
- 单说话人处理
- ASR + Translation + TTS流程

## 贡献 / Contributing

欢迎提交问题和拉取请求！

Contributions are welcome! Please submit issues and pull requests.
