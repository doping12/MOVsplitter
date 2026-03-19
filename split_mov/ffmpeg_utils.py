from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess

from .config import DetectionConfig
from .utils import TimeRange, output_path_for_index


class FFmpegError(RuntimeError):
    pass


def ensure_ffmpeg_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegError(f"{tool} が見つかりません。brew install ffmpeg を実行してください。")


def ffprobe_duration(video_path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.strip() or "ffmpeg failed")


def cut_segment_with_fallback(
    input_file: Path,
    output_file: Path,
    time_range: TimeRange,
    output_ext: str,
    cfg: DetectionConfig,
) -> None:
    ss = f"{time_range.start:.3f}"
    to = f"{time_range.end:.3f}"

    copy_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        ss,
        "-to",
        to,
        "-i",
        str(input_file),
        "-map",
        "0",
        "-c",
        "copy",
        str(output_file),
    ]

    try:
        _run_ffmpeg(copy_cmd)
        actual = ffprobe_duration(output_file)
        expected = max(0.0, time_range.duration)
        if abs(actual - expected) <= cfg.copy_cut_tolerance_sec:
            return
    except Exception:
        pass

    # copy が失敗/不正確な場合のみ再エンコード
    reencode_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        ss,
        "-to",
        to,
        "-i",
        str(input_file),
        "-map",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(output_file),
    ]

    if output_ext == "mov":
        reencode_cmd.insert(-1, "-movflags")
        reencode_cmd.insert(-1, "+faststart")

    _run_ffmpeg(reencode_cmd)


def split_video_ranges(
    input_file: Path,
    output_dir: Path,
    stem: str,
    output_ext: str,
    keep_ranges: list[TimeRange],
    cfg: DetectionConfig,
    parallel: int = 1,
) -> list[Path]:
    ensure_ffmpeg_tools()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(idx, r, output_path_for_index(output_dir, stem, idx, output_ext)) for idx, r in enumerate(keep_ranges, start=1)]
    if not jobs:
        return []

    if parallel <= 1:
        out_files: list[Path] = []
        for _, r, out in jobs:
            cut_segment_with_fallback(input_file, out, r, output_ext, cfg)
            out_files.append(out)
        return out_files

    def _worker(idx: int, r: TimeRange, out: Path) -> tuple[int, Path]:
        cut_segment_with_fallback(input_file, out, r, output_ext, cfg)
        return idx, out

    done: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as ex:
        futures = [ex.submit(_worker, idx, r, out) for idx, r, out in jobs]
        for fu in as_completed(futures):
            idx, out = fu.result()
            done[idx] = out

    return [done[idx] for idx in sorted(done.keys())]
