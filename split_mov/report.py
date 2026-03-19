from __future__ import annotations

from pathlib import Path
import csv
import json

from .config import DetectionConfig
from .detect import DetectionResult
from .utils import TimeRange


def build_report(
    input_file: Path,
    detection: DetectionResult,
    output_ranges: list[TimeRange],
    output_files: list[Path],
    cfg: DetectionConfig,
) -> dict:
    return {
        "input_file": str(input_file),
        "duration": detection.duration,
        "detected_loading_ranges": [{"start": r.start, "end": r.end} for r in detection.loading_ranges],
        "output_ranges": [{"start": r.start, "end": r.end} for r in output_ranges],
        "output_files": [str(p) for p in output_files],
        "config_used": cfg.to_dict(),
        "detection_metrics_summary": detection.metrics_summary,
        "candidates": [
            {
                "start": c.start,
                "end": c.end,
                "mean_score": c.mean_score,
                "freeze_ratio": c.freeze_ratio,
                "low_info_ratio": c.low_info_ratio,
            }
            for c in detection.candidates
        ],
        "boundary_checks": detection.boundary_checks,
    }


def write_report_json(path: str | Path, report: dict) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_preview_csv(path: str | Path, loading_ranges: list[TimeRange], output_ranges: list[TimeRange]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "start", "end", "duration"])
        for r in loading_ranges:
            w.writerow(["loading", f"{r.start:.3f}", f"{r.end:.3f}", f"{r.duration:.3f}"])
        for r in output_ranges:
            w.writerow(["keep", f"{r.start:.3f}", f"{r.end:.3f}", f"{r.duration:.3f}"])
