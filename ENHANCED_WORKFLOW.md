# Enhanced Workflow Implementation Guide / 增强型工作流实现指南

## Overview / 概述

TransFilm 现已实现增强型5阶段工作流，整合4个 Qwen 模型，实现高精度视频配音转换。

TransFilm now features an enhanced 5-stage workflow integrating 4 Qwen models for high-precision video dubbing.

## Architecture / 架构

### Model Integration / 模型集成

```
┌─────────────────────────────────────────────────────────────────┐
│                      Enhanced Workflow                          │
└─────────────────────────────────────────────────────────────────┘

┌────────────────┐
│  Input Video   │
└───────┬────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 1: Speech Recognition (Qwen3-ASR-1.7B)                   │
│ • Extract audio from video                                      │
│ • Transcribe full audio content                                 │
│ • High-precision recognition                                    │
└───────┬────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 2: Segmentation & Translation (Qwen3-0.6B)              │
│ • Segment text using shortest natural breaks                   │
│ • Translate each segment                                        │
│ • Match character counts for similar duration                   │
└───────┬────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 3: Timestamp Alignment (Qwen3-ForcedAligner-0.6B)       │
│ • Generate precise timestamps for each segment                  │
│ • Align text to audio                                          │
│ • Refine to match total duration                               │
└───────┬────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 4: Voice Cloning & TTS (Qwen3-TTS)                      │
│ • Extract voice features from original                          │
│ • Clone voice characteristics                                   │
│ • Generate audio matching timestamps exactly                    │
└───────┬────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 5: Audio Assembly                                        │
│ • Concatenate segments by timestamps                           │
│ • Ensure total duration matches source                         │
│ • Combine with video                                           │
└───────┬────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────┐
│  Output Video  │
└────────────────┘
```

## Key Features / 核心特性

### 1. Character Count Matching / 字符数匹配

**Purpose / 目的**: Ensure translated text generates audio of similar duration to original
确保翻译后的文本生成相似时长的音频

**Implementation / 实现**:
```python
# In translation_engine.py
def translate_with_character_matching(text, source_lang, target_lang):
    source_char_count = len(text)
    
    # Prompt includes character count hint
    prompt = f"""Translate keeping length similar to {source_char_count} characters
    
    Original: {text}
    Translation:"""
    
    translation = model.generate(prompt)
    char_ratio = len(translation) / source_char_count
    
    return translation, char_ratio
```

**Configuration / 配置**:
- `ENABLE_CHARACTER_COUNT_MATCHING = True` in config.py
- `CHARACTER_COUNT_TOLERANCE = 0.2` (允许20%差异)

### 2. Precise Duration Control / 精确时长控制

**Purpose / 目的**: Match TTS audio duration to timestamps exactly
使TTS生成的音频精确匹配时间戳

**Implementation / 实现**:
```python
# In tts_engine.py
def synthesize(text, target_duration=None):
    audio = generate_audio(text)
    
    if target_duration:
        # Time-stretch to exact duration
        current_duration = len(audio) / sample_rate
        rate = current_duration / target_duration
        audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Trim/pad to exact samples
        target_samples = int(target_duration * sample_rate)
        audio = adjust_to_exact_length(audio, target_samples)
    
    return audio
```

**Configuration / 配置**:
- `ENABLE_DURATION_MATCHING = True` in config.py
- `DURATION_TOLERANCE = 0.1` (允许10%差异)

### 3. Timestamp-Based Assembly / 基于时间戳的拼接

**Purpose / 目的**: Concatenate audio segments at precise timestamps
在精确时间戳位置拼接音频段

**Implementation / 实现**:
```python
# In audio_processor.py
def concatenate_audio_with_timestamps(segments, timestamps, total_duration):
    # Create output buffer with exact duration
    total_samples = int(total_duration * sample_rate)
    output_audio = np.zeros(total_samples)
    
    for audio, timestamp in zip(segments, timestamps):
        start_sample = int(timestamp['start'] * sample_rate)
        # Adjust audio to match timestamp duration
        adjusted = adjust_audio_length(audio, timestamp['duration'])
        # Place in output buffer at exact position
        output_audio[start_sample:start_sample + len(adjusted)] = adjusted
    
    return output_audio
```

## Usage Examples / 使用示例

### Command Line / 命令行

```bash
# Full workflow with Chinese to English translation
python cli.py \
  --input chinese_video.mp4 \
  --output english_video.mp4 \
  --language zh \
  --target-language en \
  --device cuda

# English to Chinese translation
python cli.py \
  --input english_video.mp4 \
  --output chinese_video.mp4 \
  --language en \
  --target-language zh \
  --device cuda

# Use custom models
python cli.py \
  --input video.mp4 \
  --output output.mp4 \
  --asr-model /path/to/asr/model \
  --translation-model /path/to/translation/model \
  --forced-aligner-model /path/to/aligner/model \
  --tts-model /path/to/tts/model

# Low VRAM mode
python cli.py \
  --input video.mp4 \
  --output output.mp4 \
  --device cpu \
  --chunk-size 15
```

### Python API / Python接口

```python
from transfilm import VideoDubbingPipeline

# Create pipeline with enhanced workflow
pipeline = VideoDubbingPipeline(
    asr_model="Qwen/Qwen3-ASR-1.7B",
    translation_model="Qwen/Qwen3-0.6B",
    forced_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
    tts_model="Qwen/Qwen3-TTS",
    source_language="zh",
    target_language="en",
    device="cuda"
)

# Process video with progress callback
def progress_callback(message, progress):
    print(f"[{progress:.1f}%] {message}")

pipeline.progress_callback = progress_callback
output = pipeline.process_video("input.mp4", "output.mp4")
print(f"Complete: {output}")
```

## Configuration / 配置

### config.py Settings / 配置设置

```python
# Model configurations
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_TRANSLATION_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS"

# Processing parameters
DEFAULT_CHUNK_SIZE = 30
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "zh"
DEFAULT_TARGET_LANGUAGE = "en"

# Translation settings
ENABLE_CHARACTER_COUNT_MATCHING = True
CHARACTER_COUNT_TOLERANCE = 0.2
ENABLE_DURATION_MATCHING = True
DURATION_TOLERANCE = 0.1

# Memory optimization
ENABLE_MODEL_OFFLOADING = True
USE_8BIT = False
USE_4BIT = False
```

## Performance Optimization / 性能优化

### Memory Management / 内存管理

1. **Model Offloading / 模型卸载**
   - Models are automatically loaded/unloaded between stages
   - 模型在各阶段间自动加载/卸载
   - Reduces peak memory usage / 降低峰值内存使用

2. **Quantization / 量化**
   - 8-bit: ~50% memory reduction
   - 4-bit: ~75% memory reduction
   - Trade-off: Slight quality reduction / 权衡：略微降低质量

3. **Chunk Size Tuning / 分块大小调优**
   - Larger chunks: Faster but more memory
   - Smaller chunks: Slower but less memory
   - Recommended: 15-30 seconds / 推荐：15-30秒

### Duration Matching Optimization / 时长匹配优化

```python
# Three critical checkpoints for duration matching
# 时长匹配的三个关键检查点

# 1. Translation character count matching
translation_char_ratio = len(translation) / len(original)
if abs(translation_char_ratio - 1.0) > CHARACTER_COUNT_TOLERANCE:
    logger.warning("Character ratio outside tolerance")

# 2. TTS audio duration matching
audio_duration = len(audio) / sample_rate
if abs(audio_duration - target_duration) > DURATION_TOLERANCE:
    audio = match_duration_precisely(audio, target_duration)

# 3. Final audio total duration matching
final_duration = len(final_audio) / sample_rate
if abs(final_duration - source_duration) > 0.01:
    final_audio = adjust_to_exact_duration(final_audio, source_duration)
```

## Workflow Details / 工作流详情

### Stage 1: ASR Transcription / 语音识别

**Input / 输入**: Raw audio from video
**Output / 输出**: Complete transcription text
**Time / 时间**: ~10-25% of workflow

**Process / 过程**:
1. Extract audio from video at 16kHz
2. Load Qwen3-ASR-1.7B model
3. Transcribe full audio in one pass
4. Unload ASR model to free memory

### Stage 2: Translation / 翻译

**Input / 输入**: Transcription text
**Output / 输出**: Segmented and translated text
**Time / 时间**: ~25-45% of workflow

**Process / 过程**:
1. Load Qwen3-0.6B translation model
2. Segment text using shortest natural breaks
3. Translate each segment with character count awareness
4. Log character ratios for monitoring
5. Unload translation model

### Stage 3: Alignment / 对齐

**Input / 输入**: Original audio + segmented text
**Output / 输出**: Timestamps for each segment
**Time / 时间**: ~45-60% of workflow

**Process / 过程**:
1. Load Qwen3-ForcedAligner-0.6B model
2. Align each text segment to audio
3. Generate start/end timestamps
4. Refine alignments to match total duration
5. Unload aligner model

### Stage 4: TTS Synthesis / 语音合成

**Input / 输入**: Translated text + timestamps
**Output / 输出**: Audio segments matching timestamps
**Time / 时间**: ~60-85% of workflow

**Process / 过程**:
1. Extract voice features from original audio
2. Load Qwen3-TTS model
3. For each segment:
   - Extract reference audio for voice cloning
   - Synthesize with target duration
   - Validate duration match
4. Unload TTS model

### Stage 5: Assembly / 拼接

**Input / 输入**: Audio segments + timestamps
**Output / 输出**: Final video with new audio
**Time / 时间**: ~85-100% of workflow

**Process / 过程**:
1. Concatenate segments at timestamp positions
2. Validate total duration
3. Save combined audio
4. Merge with original video
5. Clean up temporary files

## Troubleshooting / 故障排查

### Duration Mismatch / 时长不匹配

**Problem / 问题**: Final audio doesn't match source duration
**Solutions / 解决方案**:
1. Check `DURATION_TOLERANCE` setting
2. Enable `ENABLE_DURATION_MATCHING`
3. Use smaller chunk sizes for better precision
4. Check log warnings for specific segments

### Character Ratio Issues / 字符比率问题

**Problem / 问题**: Translation too long/short
**Solutions / 解决方案**:
1. Adjust `CHARACTER_COUNT_TOLERANCE`
2. Use different translation prompts
3. Post-process translations manually
4. Try different language pairs

### Memory Issues / 内存问题

**Problem / 问题**: Out of memory errors
**Solutions / 解决方案**:
1. Enable `ENABLE_MODEL_OFFLOADING = True`
2. Use 8-bit or 4-bit quantization
3. Reduce chunk size
4. Use CPU mode
5. Process shorter videos

## Future Enhancements / 未来增强

1. **Multi-speaker Support / 多说话人支持**
   - Detect and preserve multiple speakers
   - 检测并保留多个说话人

2. **Emotion Preservation / 情感保留**
   - Analyze and transfer emotional tone
   - 分析并转移情感语调

3. **Background Music Handling / 背景音乐处理**
   - Separate and preserve background audio
   - 分离并保留背景音频

4. **Real-time Processing / 实时处理**
   - Stream processing for live videos
   - 流式处理实时视频

## References / 参考资料

- [Qwen3-ASR Documentation](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-TTS Documentation](https://github.com/QwenLM/Qwen3-TTS)
- [Forced Alignment Techniques](https://en.wikipedia.org/wiki/Forced_alignment)
- [Time-stretching Algorithms](https://librosa.org/doc/latest/generated/librosa.effects.time_stretch.html)

---

**Last Updated / 最后更新**: 2026-02-10
**Version / 版本**: 0.2.0 (Enhanced Workflow)
