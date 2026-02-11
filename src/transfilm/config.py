"""from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    workdir: Path = Path(".transfilm")
    sample_rate: int = 48000
    channels: int = 1
    min_segment_chars: int = 10
    max_segment_chars: int = 120
    length_tolerance: float = 0.15
    match_tts_length: bool = True
    total_duration_tolerance: float = 0.25
    dummy_mode: bool = False
    device: str = "cuda"
"""