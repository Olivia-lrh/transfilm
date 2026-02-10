# Model Integration Notes / 模型集成说明

## Current Status / 当前状态

TransFilm 项目已建立完整的框架，但实际的 Qwen3-ASR 和 Qwen3-TTS 模型集成需要根据官方 API 进行调整。

The TransFilm project has established a complete framework, but actual Qwen3-ASR and Qwen3-TTS model integration needs to be adjusted according to official APIs.

## Qwen3-ASR Integration / Qwen3-ASR 集成

### Current Implementation / 当前实现

文件: `transfilm/asr_engine.py`

File: `transfilm/asr_engine.py`

当前使用标准的 Transformers API:
```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
```

Currently uses standard Transformers API.

### Official Qwen3-ASR API / 官方 Qwen3-ASR API

根据 Qwen 官方文档，实际 API 可能为:

According to Qwen official documentation, the actual API might be:

```python
# Example - adjust based on official documentation
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='qwen/Qwen3-ASR'
)

result = asr_pipeline(audio_input)
```

### TODO for ASR Integration / ASR 集成待办

1. 查阅 Qwen3-ASR 官方文档 / Check Qwen3-ASR official docs
2. 更新 `ASREngine` 类以使用正确的 API / Update `ASREngine` class with correct API
3. 测试不同的输入格式 / Test different input formats
4. 优化批处理 / Optimize batch processing

## Qwen3-TTS Integration / Qwen3-TTS 集成

### Current Implementation / 当前实现

文件: `transfilm/tts_engine.py`

File: `transfilm/tts_engine.py`

当前使用占位符 API:
```python
from transformers import AutoModelForTextToWaveform, AutoTokenizer
```

Currently uses placeholder API.

### Official Qwen3-TTS API / 官方 Qwen3-TTS API

根据 Qwen 官方文档，实际 API 可能为:

According to Qwen official documentation, the actual API might be:

```python
# Example - adjust based on official documentation
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

tts_pipeline = pipeline(
    task=Tasks.text_to_speech,
    model='qwen/Qwen3-TTS'
)

result = tts_pipeline(
    text="要合成的文本",
    voice_features=voice_params
)
```

### TODO for TTS Integration / TTS 集成待办

1. 查阅 Qwen3-TTS 官方文档 / Check Qwen3-TTS official docs
2. 更新 `TTSEngine` 类以使用正确的 API / Update `TTSEngine` class with correct API
3. 实现音色克隆功能 / Implement voice cloning feature
4. 优化音频长度控制 / Optimize audio duration control

## Voice Cloning / 音色克隆

### Current Approach / 当前方法

音色特征提取在 `audio_processor.py` 中:

Voice feature extraction in `audio_processor.py`:

```python
def extract_voice_features(self, audio_data: np.ndarray) -> dict:
    # Extracts pitch, energy, spectral features, tempo
    ...
```

### Enhancement Needed / 需要增强

1. 使用更高级的音色提取方法 / Use advanced voice extraction methods
2. 集成声纹识别模型 / Integrate speaker recognition models
3. 实现音色转换 / Implement voice conversion
4. 测试音色保真度 / Test voice fidelity

## Model Sources / 模型来源

### HuggingFace Hub

```python
# If models are on HuggingFace
model = "Qwen/Qwen3-ASR"
```

### ModelScope

```python
# If models are on ModelScope
model = "qwen/Qwen3-ASR"
```

### Local Models / 本地模型

```python
# Use local path
model = "/path/to/local/model"
```

## Performance Optimization / 性能优化

### Current Optimizations / 当前优化

1. ✅ 模型懒加载 / Model lazy loading
2. ✅ 模型卸载 / Model offloading
3. ✅ 量化支持 (8-bit, 4-bit) / Quantization support
4. ✅ CPU 回退 / CPU fallback
5. ✅ 批处理 / Batch processing

### Additional Optimizations / 额外优化

1. ⬜ Flash Attention 支持 / Flash Attention support
2. ⬜ 模型蒸馏 / Model distillation
3. ⬜ ONNX 导出 / ONNX export
4. ⬜ TensorRT 加速 / TensorRT acceleration

## Testing with Actual Models / 使用实际模型测试

### Step 1: Install ModelScope / 安装 ModelScope

```bash
pip install modelscope
```

### Step 2: Download Models / 下载模型

```python
from modelscope.hub.snapshot_download import snapshot_download

# Download ASR model
asr_model_dir = snapshot_download('qwen/Qwen3-ASR')

# Download TTS model
tts_model_dir = snapshot_download('qwen/Qwen3-TTS')
```

### Step 3: Update Configuration / 更新配置

```python
# In config.py
DEFAULT_ASR_MODEL = "/path/to/downloaded/asr/model"
DEFAULT_TTS_MODEL = "/path/to/downloaded/tts/model"
```

### Step 4: Test / 测试

```bash
python cli.py --input test_video.mp4 --output output.mp4
```

## Contributing Model Integration / 贡献模型集成

如果您熟悉 Qwen3-ASR 或 Qwen3-TTS 的官方 API，欢迎贡献！

If you're familiar with official Qwen3-ASR or Qwen3-TTS APIs, contributions are welcome!

### How to Contribute / 如何贡献

1. Fork 本仓库 / Fork this repository
2. 创建新分支 / Create new branch: `git checkout -b feature/qwen-integration`
3. 更新 ASR/TTS engine 代码 / Update ASR/TTS engine code
4. 测试功能 / Test functionality
5. 提交 Pull Request / Submit Pull Request

### What We Need / 我们需要

- [ ] Qwen3-ASR 官方 API 文档链接 / Official Qwen3-ASR API docs link
- [ ] Qwen3-TTS 官方 API 文档链接 / Official Qwen3-TTS API docs link
- [ ] 实际的模型加载代码 / Actual model loading code
- [ ] 音色克隆实现 / Voice cloning implementation
- [ ] 性能基准测试 / Performance benchmarks

## References / 参考资料

- [Qwen GitHub Organization](https://github.com/QwenLM)
- [ModelScope Platform](https://modelscope.cn/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

**Note / 注意**: 本项目提供了完整的框架和接口，但需要根据实际的 Qwen 模型 API 进行适配。

This project provides a complete framework and interfaces, but needs adaptation based on actual Qwen model APIs.
