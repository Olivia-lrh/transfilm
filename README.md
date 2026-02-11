# Transfilm

Transfilm is an end-to-end pipeline that integrates Qwen3 models for **ASR → segmentation/translation → forced alignment → voice-cloned TTS → timeline stitching**. It is designed for audio and video inputs and produces outputs in the same container format as the input.

## Workflow
1. **Qwen3-ASR-1.7B** transcribes the full audio.
2. **Qwen3-0.6B** segments the transcript into short spoken units and translates each segment with a **length-aware** constraint to keep the synthesized audio duration close to the original.
3. **Qwen3-ForcedAligner-0.6B** aligns each segment to timestamps.
4. **Qwen3-TTS** voice-clones each segment and generates translated audio.
5. The translated audio segments are **stretched/compressed** to fit the timestamps, then **stitched** into a timeline that matches the original duration.

## Features
- Accepts common audio (`wav`, `mp3`, `flac`, `m4a`) and video (`mp4`, `mkv`, `mov`, `webm`) inputs
- Outputs the same container type as the input
- FFmpeg-based audio extraction, time-stretching, and muxing
- Modular model adapters for easy integration with Qwen3 repositories

## Requirements
- Python 3.13+
- PyTorch 2.10 + CUDA 13.0
- Flash-Attn
- FFmpeg

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

transfilm --input ./samples/input.mp4 --output ./outputs/input.mp4 --source-lang zh --target-lang en
```

## Docker
```bash
docker build -t transfilm:latest .
docker run --rm -it \
  --gpus all \
  -v $(pwd):/workspace \
  transfilm:latest \
  transfilm --input /workspace/samples/input.mp4 --output /workspace/outputs/input.mp4 \
  --source-lang zh --target-lang en
```

## Configuration and Model Integration
The project provides adapter classes where you should wire in the Qwen3 model calls:
- `src/transfilm/models/asr.py` → Qwen3-ASR-1.7B
- `src/transfilm/models/translate.py` → Qwen3-0.6B segmentation + translation
- `src/transfilm/models/align.py` → Qwen3-ForcedAligner-0.6B
- `src/transfilm/models/tts.py` → Qwen3-TTS

Each adapter exposes a minimal, stable interface and includes TODOs for the exact calls to the Qwen3 repositories.

## Output Structure
- Temporary files are stored under `workdir` (default: `.transfilm`)
- Final output matches the input container format
- Optional JSON artifacts (segments and timestamps) are written alongside outputs

## License
MIT