from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TimeRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def clamp_time_range(r: TimeRange, duration: float) -> TimeRange:
    start = max(0.0, min(r.start, duration))
    end = max(0.0, min(r.end, duration))
    if end < start:
        start, end = end, start
    return TimeRange(start, end)


def output_path_for_index(output_dir: Path, stem: str, index: int, ext: str) -> Path:
    safe_ext = ext.lower().lstrip(".")
    return output_dir / f"{stem}_{index}.{safe_ext}"


def detect_output_ext(input_path: Path, output_ext: str | None = None) -> str:
    if output_ext:
        ext = output_ext.lower().lstrip(".")
        if ext not in {"mov", "mp4"}:
            raise ValueError("--output-ext は mov または mp4 のみ指定できます")
        return ext

    src_ext = input_path.suffix.lower().lstrip(".")
    if src_ext in {"mov", "mp4"}:
        return src_ext
    return "mp4"


def is_supported_input(path: Path) -> bool:
    return path.suffix.lower() in {".mov", ".mp4"}
