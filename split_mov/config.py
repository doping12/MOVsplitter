from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import yaml


@dataclass(slots=True)
class DetectionConfig:
    coarse_sample_fps: float = 4.0
    coarse_skip_after_loading_sec: float = 100.0
    coarse_boundary_sample_fps: float = 12.0
    refine_sample_fps: float = 60.0
    refine_window_sec: float = 2.0
    refine_parallel_workers: int = 0
    brightness_change_threshold: float = 0.18
    histogram_diff_threshold: float = 0.20
    ssim_threshold: float = 0.93
    low_information_threshold: float = 22.0
    color_std_threshold: float = 30.0
    frame_diff_threshold: float = 0.025
    freeze_min_duration_sec: float = 1.2
    loading_min_duration_sec: float = 1.8
    merge_gap_sec: float = 0.8
    min_segment_sec: float = 3.0
    loading_score_threshold: int = 3
    smoothing_window_sec: float = 0.8
    dark_brightness_threshold: float = 35.0
    bright_brightness_threshold: float = 220.0
    use_template_matching: bool = True
    loading_template_path: str | None = None
    title_template_path: str | None = None
    template_param_dir: str = "param"
    template_ignore_edge_sec: float = 0.5
    template_similarity_threshold: float = 0.62
    template_strict_similarity_threshold: float = 0.82
    title_template_similarity_threshold: float = 0.58
    title_template_strict_similarity_threshold: float = 0.78
    center_white_ratio_threshold: float = 0.010
    center_edge_ratio_threshold: float = 0.010
    center_outer_contrast_threshold: float = 4.0
    title_dark_brightness_threshold: float = 120.0
    head_edge_refine_enabled: bool = True
    head_loading_window_sec: float = 3.0
    head_min_stable_sec: float = 0.75
    head_density_threshold: float = 0.70
    head_template_similarity_threshold: float = 0.48
    head_jump_min_score: int = 3
    head_jump_delta: int = 2
    head_fine_tune_frame_offset: int = 0
    tail_edge_refine_enabled: bool = True
    tail_loading_window_sec: float = 3.0
    tail_min_stable_sec: float = 0.75
    tail_density_threshold: float = 0.70
    tail_template_similarity_threshold: float = 0.48
    tail_jump_min_score: int = 3
    tail_jump_delta: int = 2
    tail_fine_tune_frame_offset: int = 0
    ai_enabled: bool = False
    ai_confidence_threshold: float = 0.65
    ai_model_path: str | None = None
    ai_input_size: int = 224
    copy_cut_tolerance_sec: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not data:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be object: {path}")
    return data


def load_config(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> DetectionConfig:
    cfg = DetectionConfig()
    merged = cfg.to_dict()

    if config_path:
        path = Path(config_path)
        file_data = _read_config_file(path)
        merged.update(file_data)

    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})

    valid_keys = set(cfg.to_dict().keys())
    unknown = [k for k in merged.keys() if k not in valid_keys]
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

    return DetectionConfig(**merged)
