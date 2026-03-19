from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .ai_helper import OptionalAIHelper
from .config import DetectionConfig
from .features import (
    FrameMetrics,
    edge_ratio,
    extract_sampled_features,
    extract_sampled_features_window,
    frame_brightness,
    frame_color_std,
    frame_diff_ratio,
    frame_histogram,
    frame_ssim,
    histogram_diff,
    laplacian_variance,
    title_roi_features,
)
from .segment import merge_ranges
from .utils import TimeRange

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidateRange:
    start: float
    end: float
    mean_score: float
    freeze_ratio: float
    low_info_ratio: float
    start_idx: int
    end_idx: int


@dataclass(slots=True)
class DetectionResult:
    duration: float
    loading_ranges: list[TimeRange]
    candidates: list[CandidateRange]
    metrics_summary: dict[str, float]
    timeline: list[dict[str, float]]
    boundary_checks: list[dict[str, object]]


@dataclass(slots=True)
class TemplateProfile:
    means: dict[str, float]
    stds: dict[str, float]


def _template_profile_to_dict(profile: TemplateProfile) -> dict[str, dict[str, float]]:
    return {"means": profile.means, "stds": profile.stds}


def _template_profile_from_dict(data: dict[str, object]) -> TemplateProfile | None:
    means = data.get("means")
    stds = data.get("stds")
    if not isinstance(means, dict) or not isinstance(stds, dict):
        return None
    try:
        return TemplateProfile(
            means={str(k): float(v) for k, v in means.items()},
            stds={str(k): float(v) for k, v in stds.items()},
        )
    except Exception:
        return None


def _template_cache_path(cfg: DetectionConfig, kind: str) -> Path:
    d = Path(cfg.template_param_dir).expanduser()
    if not d.is_absolute():
        d = (Path.cwd() / d).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_template_profile.json"


def _save_template_profile(cfg: DetectionConfig, kind: str, profile: TemplateProfile) -> None:
    p = _template_cache_path(cfg, kind)
    p.write_text(json.dumps(_template_profile_to_dict(profile), ensure_ascii=False), encoding="utf-8")


def _load_template_profile(cfg: DetectionConfig, kind: str) -> TemplateProfile | None:
    p = _template_cache_path(cfg, kind)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _template_profile_from_dict(data)


def _build_template_profile(
    metrics: list[FrameMetrics],
    duration: float,
    ignore_edge_sec: float,
) -> TemplateProfile | None:
    if not metrics:
        return None

    lo = max(0.0, ignore_edge_sec)
    hi = max(lo, duration - max(0.0, ignore_edge_sec))
    core = [m for m in metrics if (m.time_sec > lo and m.time_sec < hi)]
    if not core:
        return None

    keys = (
        "brightness",
        "color_std",
        "laplacian_var",
        "hist_diff",
        "ssim",
        "frame_diff",
        "center_white_ratio",
        "center_edge_ratio",
        "center_outer_contrast",
    )
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for k in keys:
        arr = np.array([float(getattr(m, k)) for m in core], dtype=np.float64)
        means[k] = float(np.mean(arr))
        stds[k] = float(np.std(arr))
    return TemplateProfile(means=means, stds=stds)


def _gaussian_similarity(x: float, mean: float, std: float) -> float:
    denom = max(std, 1e-6)
    z = (x - mean) / denom
    return float(np.exp(-0.5 * z * z))


def _template_similarity(m: FrameMetrics, profile: TemplateProfile | None) -> float:
    if profile is None:
        return 0.0
    weights = {
        "brightness": 0.11,
        "color_std": 0.17,
        "laplacian_var": 0.16,
        "hist_diff": 0.10,
        "ssim": 0.10,
        "frame_diff": 0.10,
        "center_white_ratio": 0.10,
        "center_edge_ratio": 0.08,
        "center_outer_contrast": 0.08,
    }
    s = 0.0
    wsum = 0.0
    for k, w in weights.items():
        x = float(getattr(m, k))
        sim = _gaussian_similarity(x, profile.means[k], profile.stds[k])
        s += w * sim
        wsum += w
    return float(s / max(wsum, 1e-6))


def _frame_rule_score(m: FrameMetrics, cfg: DetectionConfig) -> tuple[int, dict[str, bool]]:
    freeze = (m.frame_diff <= cfg.frame_diff_threshold) and (m.ssim >= cfg.ssim_threshold)
    low_info = m.laplacian_var <= cfg.low_information_threshold
    low_color = m.color_std <= cfg.color_std_threshold
    dark_or_bright = (m.brightness <= cfg.dark_brightness_threshold) or (m.brightness >= cfg.bright_brightness_threshold)
    low_hist_change = m.hist_diff <= cfg.histogram_diff_threshold
    title_like = (
        (m.brightness <= cfg.title_dark_brightness_threshold)
        and (m.center_white_ratio >= cfg.center_white_ratio_threshold)
        and (m.center_edge_ratio >= cfg.center_edge_ratio_threshold)
        and (m.center_outer_contrast >= cfg.center_outer_contrast_threshold)
    )

    flags = {
        "freeze": freeze,
        "low_info": low_info,
        "low_color": low_color,
        "dark_or_bright": dark_or_bright,
        "low_hist_change": low_hist_change,
        "title_like": title_like,
    }
    score = sum(1 for v in flags.values() if v)
    return score, flags


def _smooth_binary(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size == 0:
        return arr.astype(np.uint8)
    k = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(arr.astype(np.float32), k, mode="same")
    return (smoothed >= 0.5).astype(np.uint8)


def _apply_edge_loading_rescue(
    loading_arr: np.ndarray,
    sample_dt: float,
    cfg: DetectionConfig,
    edge_flags: list[int],
    edge: str,
) -> np.ndarray:
    if loading_arr.size == 0:
        return loading_arr
    if edge == "head":
        enabled = bool(cfg.head_edge_refine_enabled)
        edge_window_sec = float(cfg.head_loading_window_sec)
        min_sec = float(cfg.head_min_stable_sec)
        density_thr = float(cfg.head_density_threshold)
    else:
        enabled = bool(cfg.tail_edge_refine_enabled)
        edge_window_sec = float(cfg.tail_loading_window_sec)
        min_sec = float(cfg.tail_min_stable_sec)
        density_thr = float(cfg.tail_density_threshold)
    if not enabled:
        return loading_arr

    edge_window = max(1, int(round(edge_window_sec / max(sample_dt, 1e-6))))
    min_frames = max(1, int(round(min_sec / max(sample_dt, 1e-6))))
    n = loading_arr.size
    if edge == "head":
        end_scan = min(n - 1, edge_window - 1)
        rescue_end = None
        for e in range(0, end_scan + 1):
            seg = edge_flags[0 : e + 1]
            if len(seg) < min_frames:
                continue
            pos = int(sum(seg))
            density = pos / float(len(seg))
            if pos >= min_frames and density >= density_thr:
                rescue_end = e
        if rescue_end is not None:
            loading_arr[0 : rescue_end + 1] = 1
    else:
        end_idx = n - 1
        start_scan = max(0, end_idx - edge_window + 1)
        rescue_start = None
        for s in range(start_scan, end_idx + 1):
            seg = edge_flags[s : end_idx + 1]
            if len(seg) < min_frames:
                continue
            pos = int(sum(seg))
            density = pos / float(len(seg))
            if pos >= min_frames and density >= density_thr:
                rescue_start = s
                break
        if rescue_start is not None:
            loading_arr[rescue_start : end_idx + 1] = 1
    return loading_arr


def _extract_coarse_features_with_loading_skip(
    video_path: str,
    sample_fps: float,
    cfg: DetectionConfig,
    template_profile: TemplateProfile | None,
    title_template_profile: TemplateProfile | None,
) -> tuple[list[FrameMetrics], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / native_fps) if frame_count > 0 else 0.0

    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))
    sample_dt = 1.0 / max(sample_fps, 0.1)
    min_loading_frames = max(1, int(round(cfg.loading_min_duration_sec / sample_dt)))
    skip_sec = max(0.0, float(cfg.coarse_skip_after_loading_sec))

    metrics: list[FrameMetrics] = []
    prev_frame: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    frame_idx = 0
    run_len = 0
    skip_count = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            frame_idx += 1
            continue

        brightness = frame_brightness(frame)
        color_std = frame_color_std(frame)
        lap_var = laplacian_variance(frame)
        edges = edge_ratio(frame)
        center_white_ratio, center_edge_ratio, center_outer_contrast = title_roi_features(frame)
        hist = frame_histogram(frame)
        time_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        if prev_frame is None or prev_hist is None:
            hist_d = 0.0
            ssim = 1.0
            diff = 0.0
        else:
            hist_d = histogram_diff(prev_hist, hist)
            ssim = frame_ssim(prev_frame, frame)
            diff = frame_diff_ratio(prev_frame, frame)

        m = FrameMetrics(
            time_sec=time_sec,
            brightness=brightness,
            color_std=color_std,
            laplacian_var=lap_var,
            edge_ratio=edges,
            hist_diff=hist_d,
            ssim=ssim,
            frame_diff=diff,
            center_white_ratio=center_white_ratio,
            center_edge_ratio=center_edge_ratio,
            center_outer_contrast=center_outer_contrast,
        )
        metrics.append(m)

        score, flags = _frame_rule_score(m, cfg)
        tsim = _template_similarity(m, template_profile) if template_profile else 0.0
        title_tsim = _template_similarity(m, title_template_profile) if title_template_profile else 0.0
        use_template = cfg.use_template_matching and (template_profile is not None or title_template_profile is not None)
        if use_template:
            strict_template = (
                (tsim >= cfg.template_strict_similarity_threshold)
                or (title_tsim >= cfg.title_template_strict_similarity_threshold)
            )
            template_ok = (
                (tsim >= cfg.template_similarity_threshold)
                or (title_tsim >= cfg.title_template_similarity_threshold)
            )
            loading_hit = strict_template or ((score >= cfg.loading_score_threshold or flags["title_like"]) and template_ok)
        else:
            loading_hit = (score >= cfg.loading_score_threshold) or flags["title_like"]

        if loading_hit:
            run_len += 1
        else:
            run_len = 0

        if skip_sec > 0.0 and run_len >= min_loading_frames:
            target_sec = time_sec + skip_sec
            if duration > 0:
                target_sec = min(target_sec, duration)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target_sec) * 1000.0)
            frame_idx = int(round(max(0.0, target_sec) * native_fps))
            prev_frame = None
            prev_hist = None
            run_len = 0
            skip_count += 1
            continue

        prev_frame = frame
        prev_hist = hist
        frame_idx += 1

    cap.release()
    if metrics and duration <= 0:
        duration = metrics[-1].time_sec
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "coarse extraction sampled=%d skip_count=%d skip_sec=%.3f",
            len(metrics),
            skip_count,
            skip_sec,
        )
    return metrics, duration


def detect_loading_ranges_from_metrics(
    metrics: list[FrameMetrics],
    duration: float,
    cfg: DetectionConfig,
    sample_fps: float,
    ai_helper: OptionalAIHelper | None = None,
    template_profile: TemplateProfile | None = None,
    title_template_profile: TemplateProfile | None = None,
    apply_edge_rescue: bool = True,
    skip_after_loading_sec: float = 0.0,
    on_loading_range: Callable[[TimeRange], None] | None = None,
) -> DetectionResult:
    if not metrics:
        return DetectionResult(
            duration=duration,
            loading_ranges=[],
            candidates=[],
            metrics_summary={},
            timeline=[],
            boundary_checks=[],
        )

    sample_dt = 1.0 / max(sample_fps, 0.1)
    continuity_gap_sec = sample_dt * 2.5
    smooth_window = max(1, int(round(cfg.smoothing_window_sec / sample_dt)))
    min_loading_frames = max(1, int(round(cfg.loading_min_duration_sec / sample_dt)))
    skip_frames = max(0, int(round(max(0.0, float(skip_after_loading_sec)) / sample_dt)))

    n = len(metrics)
    scores: list[int] = [0] * n
    freeze_flags: list[int] = [0] * n
    low_info_flags: list[int] = [0] * n
    loading_flags: list[int] = [0] * n
    template_sims: list[float] = [0.0] * n
    title_template_sims: list[float] = [0.0] * n
    title_like_flags: list[int] = [0] * n
    weak_start_flags: list[int] = [0] * n
    edge_flags: list[int] = [0] * n
    score_skipped_flags: list[int] = [0] * n
    computed_flags: list[int] = [0] * n

    run_len = 0
    idx = 0
    while idx < n:
        m = metrics[idx]
        score, flags = _frame_rule_score(m, cfg)
        tsim = _template_similarity(m, template_profile) if template_profile else 0.0
        title_tsim = _template_similarity(m, title_template_profile) if title_template_profile else 0.0
        use_template = cfg.use_template_matching and (template_profile is not None or title_template_profile is not None)
        if use_template:
            strict_template = (
                (tsim >= cfg.template_strict_similarity_threshold)
                or (title_tsim >= cfg.title_template_strict_similarity_threshold)
            )
            template_ok = (
                (tsim >= cfg.template_similarity_threshold)
                or (title_tsim >= cfg.title_template_similarity_threshold)
            )
            loading_hit = strict_template or (
                (score >= cfg.loading_score_threshold or flags["title_like"])
                and template_ok
            )
        else:
            loading_hit = (score >= cfg.loading_score_threshold) or flags["title_like"]

        scores[idx] = score
        template_sims[idx] = tsim
        title_template_sims[idx] = title_tsim
        title_like_flags[idx] = 1 if flags["title_like"] else 0
        freeze_flags[idx] = 1 if flags["freeze"] else 0
        low_info_flags[idx] = 1 if flags["low_info"] else 0
        loading_flags[idx] = 1 if loading_hit else 0
        computed_flags[idx] = 1
        weak_template = (
            (tsim >= cfg.template_similarity_threshold * 0.85)
            or (title_tsim >= cfg.title_template_similarity_threshold * 0.85)
        )
        weak_rule = (
            flags["title_like"]
            or ((flags["freeze"] or flags["low_info"]) and m.hist_diff <= cfg.histogram_diff_threshold * 1.2)
        )
        weak_start_flags[idx] = 1 if (weak_template or weak_rule) else 0
        edge_template_thr = min(cfg.head_template_similarity_threshold, cfg.tail_template_similarity_threshold)
        edge_cond = (
            (
                (tsim >= edge_template_thr)
                or (title_tsim >= edge_template_thr)
                or (score >= max(1, cfg.loading_score_threshold - 1))
            )
            and (m.frame_diff <= cfg.frame_diff_threshold * 1.8 or m.ssim >= max(0.0, cfg.ssim_threshold - 0.08))
        )
        edge_flags[idx] = 1 if edge_cond else 0

        if loading_hit:
            run_len += 1
        else:
            run_len = 0
        if skip_frames > 0 and run_len >= min_loading_frames:
            skip_from = idx + 1
            skip_to = min(n, skip_from + skip_frames)
            for j in range(skip_from, skip_to):
                score_skipped_flags[j] = 1
            run_len = 0
            idx = skip_to
            continue
        idx += 1

    loading_arr = _smooth_binary(np.array(loading_flags, dtype=np.uint8), smooth_window)
    if apply_edge_rescue:
        loading_arr = _apply_edge_loading_rescue(loading_arr, sample_dt, cfg, edge_flags=edge_flags, edge="head")
        loading_arr = _apply_edge_loading_rescue(loading_arr, sample_dt, cfg, edge_flags=edge_flags, edge="tail")
    # 抽出スキップで生じる大きな時間ギャップは連続区間として扱わない
    for k in range(1, len(metrics)):
        if (metrics[k].time_sec - metrics[k - 1].time_sec) > continuity_gap_sec:
            loading_arr[k - 1] = 0
            loading_arr[k] = 0

    ranges: list[TimeRange] = []
    candidates: list[CandidateRange] = []
    i = 0
    while i < len(metrics):
        if loading_arr[i] == 0:
            i += 1
            continue
        j = i + 1
        while (
            j < len(metrics)
            and loading_arr[j] == 1
            and (metrics[j].time_sec - metrics[j - 1].time_sec) <= continuity_gap_sec
        ):
            j += 1

        if (j - i) >= min_loading_frames:
            start = metrics[i].time_sec
            end = metrics[j - 1].time_sec + sample_dt
            refined_i = i
            start = metrics[refined_i].time_sec
            seg_scores = scores[i:j]
            seg_freeze = freeze_flags[i:j]
            seg_low_info = low_info_flags[i:j]
            mean_score = float(np.mean(seg_scores))
            freeze_ratio = float(np.mean(seg_freeze))
            low_info_ratio = float(np.mean(seg_low_info))

            ranges.append(TimeRange(start, end))
            if on_loading_range is not None:
                on_loading_range(TimeRange(start, end))
            candidates.append(
                CandidateRange(
                    start=start,
                    end=end,
                    mean_score=mean_score,
                    freeze_ratio=freeze_ratio,
                    low_info_ratio=low_info_ratio,
                    start_idx=refined_i,
                    end_idx=j - 1,
                )
            )
            if skip_frames > 0:
                i = min(len(metrics), j + skip_frames)
                continue
        i = j

    ranges = merge_ranges(ranges, cfg.merge_gap_sec)

    summary = {
        "sample_count": float(len(metrics)),
        "score_computed_count": float(sum(computed_flags)),
        "score_skipped_count": float(sum(score_skipped_flags)),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_max": float(np.max(scores)) if scores else 0.0,
        "freeze_ratio": float(np.mean(freeze_flags)) if freeze_flags else 0.0,
        "low_info_ratio": float(np.mean(low_info_flags)) if low_info_flags else 0.0,
        "title_like_ratio": float(np.mean(title_like_flags)) if title_like_flags else 0.0,
        "loading_frame_ratio": float(np.mean(loading_arr)) if loading_arr.size else 0.0,
        "template_similarity_mean": float(np.mean(template_sims)) if template_sims else 0.0,
        "title_template_similarity_mean": float(np.mean(title_template_sims)) if title_template_sims else 0.0,
        "template_enabled": float(
            1.0 if ((template_profile is not None or title_template_profile is not None) and cfg.use_template_matching) else 0.0
        ),
    }

    timeline: list[dict[str, float]] = []
    for idx, m in enumerate(metrics):
        timeline.append(
            {
                "time_sec": float(m.time_sec),
                "brightness": float(m.brightness),
                "color_std": float(m.color_std),
                "laplacian_var": float(m.laplacian_var),
                "hist_diff": float(m.hist_diff),
                "ssim": float(m.ssim),
                "frame_diff": float(m.frame_diff),
                "center_white_ratio": float(m.center_white_ratio),
                "center_edge_ratio": float(m.center_edge_ratio),
                "center_outer_contrast": float(m.center_outer_contrast),
                "template_similarity": float(template_sims[idx]) if template_sims else 0.0,
                "title_template_similarity": float(title_template_sims[idx]) if title_template_sims else 0.0,
                "title_like_flag": float(title_like_flags[idx]) if title_like_flags else 0.0,
                "rule_score": float(scores[idx]),
                "score_skipped": float(score_skipped_flags[idx]) if score_skipped_flags else 0.0,
                "raw_loading_flag": float(loading_flags[idx]) if loading_flags else 0.0,
                "loading_flag": float(loading_arr[idx]),
            }
        )

    return DetectionResult(
        duration=duration,
        loading_ranges=ranges,
        candidates=candidates,
        metrics_summary=summary,
        timeline=timeline,
        boundary_checks=[],
    )


def _find_score_jump_boundary(
    timeline: list[dict[str, float]],
    cfg: DetectionConfig,
    mode: str,
    fallback: float,
) -> float:
    if len(timeline) < 2:
        return fallback
    times = [float(p["time_sec"]) for p in timeline]
    scores = [float(p["rule_score"]) for p in timeline]
    if mode == "start":
        thr = float(cfg.head_jump_delta)
        min_score = float(cfg.head_jump_min_score)
    else:
        thr = float(cfg.tail_jump_delta)
        min_score = float(cfg.tail_jump_min_score)
    if mode == "start":
        candidates: list[tuple[float, float]] = []
        for k in range(1, len(scores)):
            d = scores[k] - scores[k - 1]
            if d >= thr and scores[k] >= min_score:
                candidates.append((times[k - 1], d))
        if not candidates:
            return fallback
        # Prefer expanding loading first: move start earlier.
        outward = [(t, d) for t, d in candidates if t <= fallback]
        if outward:
            # Pick the nearest valid point before fallback.
            return max(outward, key=lambda x: x[0])[0]
        # If no outward candidate, fallback to inward (later) side.
        inward = [(t, d) for t, d in candidates if t > fallback]
        if inward:
            return min(inward, key=lambda x: x[0])[0]
        return fallback

    candidates = []
    for k in range(1, len(scores)):
        d = scores[k] - scores[k - 1]
        if d <= -thr and scores[k - 1] >= min_score:
            candidates.append((times[k], d))
    if not candidates:
        return fallback
    # Prefer expanding loading first: move end later.
    outward = [(t, d) for t, d in candidates if t >= fallback]
    if outward:
        # Pick the nearest valid point after fallback.
        return min(outward, key=lambda x: x[0])[0]
    # If no outward candidate, fallback to inward (earlier) side.
    inward = [(t, d) for t, d in candidates if t < fallback]
    if inward:
        return max(inward, key=lambda x: x[0])[0]
    return fallback


def _refine_ranges_high_precision(
    video_path: str,
    coarse_ranges: list[TimeRange],
    cfg: DetectionConfig,
    template_profile: TemplateProfile | None,
    title_template_profile: TemplateProfile | None,
    on_refine_start: Callable[[int], None] | None = None,
) -> tuple[list[TimeRange], list[dict[str, object]]]:
    refined: list[TimeRange] = []
    checks: list[dict[str, object]] = []
    if not coarse_ranges:
        return refined, checks

    win = max(0.1, float(cfg.refine_window_sec))
    coarse_boundary_fps = float(cfg.coarse_boundary_sample_fps)
    refine_fps = float(cfg.refine_sample_fps)
    workers = int(cfg.refine_parallel_workers)
    if workers <= 0:
        workers = max(1, min(32, (os.cpu_count() or 4)))
    if on_refine_start is not None:
        on_refine_start(workers)

    def _work(i: int, r: TimeRange) -> tuple[int, TimeRange, dict[str, object]]:
        s0, e0 = float(r.start), float(r.end)
        # stage2: coarse boundary search
        s2_metrics, _ = extract_sampled_features_window(video_path, coarse_boundary_fps, s0 - win, s0 + win)
        e2_metrics, _ = extract_sampled_features_window(video_path, coarse_boundary_fps, e0 - win, e0 + win)
        s2_det = detect_loading_ranges_from_metrics(
            s2_metrics,
            duration=(2.0 * win),
            cfg=cfg,
            sample_fps=coarse_boundary_fps,
            ai_helper=None,
            template_profile=template_profile,
            title_template_profile=title_template_profile,
            apply_edge_rescue=False,
        )
        e2_det = detect_loading_ranges_from_metrics(
            e2_metrics,
            duration=(2.0 * win),
            cfg=cfg,
            sample_fps=coarse_boundary_fps,
            ai_helper=None,
            template_profile=template_profile,
            title_template_profile=title_template_profile,
            apply_edge_rescue=False,
        )
        s1 = _find_score_jump_boundary(s2_det.timeline, cfg, "start", s0)
        e1 = _find_score_jump_boundary(e2_det.timeline, cfg, "end", e0)

        # stage3: high precision boundary search
        s_metrics, _ = extract_sampled_features_window(video_path, refine_fps, s1 - win, s1 + win)
        e_metrics, _ = extract_sampled_features_window(video_path, refine_fps, e1 - win, e1 + win)
        s_det = detect_loading_ranges_from_metrics(
            s_metrics,
            duration=(2.0 * win),
            cfg=cfg,
            sample_fps=refine_fps,
            ai_helper=None,
            template_profile=template_profile,
            title_template_profile=title_template_profile,
            apply_edge_rescue=False,
        )
        e_det = detect_loading_ranges_from_metrics(
            e_metrics,
            duration=(2.0 * win),
            cfg=cfg,
            sample_fps=refine_fps,
            ai_helper=None,
            template_profile=template_profile,
            title_template_profile=title_template_profile,
            apply_edge_rescue=False,
        )
        s_ref = _find_score_jump_boundary(s_det.timeline, cfg, "start", s1)
        e_ref = _find_score_jump_boundary(e_det.timeline, cfg, "end", e1)
        s_ref += float(cfg.head_fine_tune_frame_offset) / max(1e-6, float(cfg.refine_sample_fps))
        e_ref += float(cfg.tail_fine_tune_frame_offset) / max(1e-6, float(cfg.refine_sample_fps))
        guard_reason = ""
        coarse_dur = max(0.0, e0 - s0)
        refined_dur = max(0.0, e_ref - s_ref)
        if e_ref <= s_ref:
            s_ref, e_ref = s0, e0
            refined_dur = coarse_dur
            guard_reason = "invalid_refined_range"
        # 高精度化でloading区間が極端に縮んで消えることを防ぐ
        # (coarseで捉えた区間は最低限維持して悪化回避)
        below_min = (coarse_dur >= float(cfg.loading_min_duration_sec)) and (
            refined_dur < float(cfg.loading_min_duration_sec)
        )
        severe_shrink = (coarse_dur > 0.0) and (refined_dur < coarse_dur * 0.35)
        if below_min or severe_shrink:
            s_ref, e_ref = s0, e0
            refined_dur = coarse_dur
            if below_min:
                guard_reason = "refined_duration_below_min"
            else:
                guard_reason = "refined_duration_shrunk_too_much"

        check = {
            "index": i,
            "coarse_start": s0,
            "coarse_end": e0,
            "coarse_duration": coarse_dur,
            "refined_start": s_ref,
            "refined_end": e_ref,
            "refined_duration": refined_dur,
            "guard_reason": guard_reason,
            "start_timeline": [{k: float(v) for k, v in p.items()} for p in s_det.timeline],
            "end_timeline": [{k: float(v) for k, v in p.items()} for p in e_det.timeline],
        }
        return i, TimeRange(s_ref, e_ref), check

    out: dict[int, tuple[TimeRange, dict[str, object]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, i, r) for i, r in enumerate(coarse_ranges, start=1)]
        for fut in as_completed(futs):
            i, rr, ck = fut.result()
            out[i] = (rr, ck)
    for i in sorted(out.keys()):
        rr, ck = out[i]
        refined.append(rr)
        checks.append(ck)
    return refined, checks


def detect_loading_ranges(
    video_path: str,
    cfg: DetectionConfig,
    ai_helper: OptionalAIHelper | None = None,
    on_coarse_loading_range: Callable[[TimeRange], None] | None = None,
    on_refine_start: Callable[[int], None] | None = None,
) -> DetectionResult:
    t_coarse_start = time.perf_counter()
    coarse_fps = float(cfg.coarse_sample_fps)

    template_profile: TemplateProfile | None = None
    title_template_profile: TemplateProfile | None = None
    if cfg.use_template_matching and cfg.loading_template_path:
        template_path = Path(cfg.loading_template_path).expanduser()
        if not template_path.is_absolute():
            template_path = (Path.cwd() / template_path).resolve()
        if template_path.exists():
            t_metrics, t_duration = extract_sampled_features(str(template_path), coarse_fps)
            template_profile = _build_template_profile(
                t_metrics,
                t_duration,
                ignore_edge_sec=cfg.template_ignore_edge_sec,
            )
            if template_profile is None:
                logger.warning("loading template profile could not be built: %s", template_path)
            else:
                _save_template_profile(cfg, "loading", template_profile)
        else:
            logger.warning("loading template not found: %s", template_path)
    elif cfg.use_template_matching and not cfg.loading_template_path:
        template_profile = _load_template_profile(cfg, "loading")
    if cfg.use_template_matching and cfg.title_template_path:
        title_path = Path(cfg.title_template_path).expanduser()
        if not title_path.is_absolute():
            title_path = (Path.cwd() / title_path).resolve()
        if title_path.exists():
            t_metrics, t_duration = extract_sampled_features(str(title_path), coarse_fps)
            title_template_profile = _build_template_profile(
                t_metrics,
                t_duration,
                ignore_edge_sec=cfg.template_ignore_edge_sec,
            )
            if title_template_profile is None:
                logger.warning("title template profile could not be built: %s", title_path)
            else:
                _save_template_profile(cfg, "title", title_template_profile)
        else:
            logger.warning("title template not found: %s", title_path)
    elif cfg.use_template_matching and not cfg.title_template_path:
        title_template_profile = _load_template_profile(cfg, "title")

    # テンプレ構築後にcoarse抽出を再実行し、テンプレ条件込みでのloading後スキップを反映
    metrics, duration = _extract_coarse_features_with_loading_skip(
        video_path=video_path,
        sample_fps=coarse_fps,
        cfg=cfg,
        template_profile=template_profile,
        title_template_profile=title_template_profile,
    )
    result = detect_loading_ranges_from_metrics(
        metrics,
        duration,
        cfg,
        sample_fps=coarse_fps,
        ai_helper=None,
        template_profile=template_profile,
        title_template_profile=title_template_profile,
        skip_after_loading_sec=0.0,
        on_loading_range=on_coarse_loading_range,
    )
    t_coarse_end = time.perf_counter()
    refined_ranges, boundary_checks = _refine_ranges_high_precision(
        video_path=video_path,
        coarse_ranges=result.loading_ranges,
        cfg=cfg,
        template_profile=template_profile,
        title_template_profile=title_template_profile,
        on_refine_start=on_refine_start,
    )
    t_refine_end = time.perf_counter()
    if refined_ranges:
        result.loading_ranges = merge_ranges(refined_ranges, cfg.merge_gap_sec)
    result.boundary_checks = boundary_checks

    if ai_helper and ai_helper.is_available() and cfg.ai_enabled and result.loading_ranges:
        refined: list[TimeRange] = []
        for r in result.loading_ranges:
            ai = ai_helper.score_range(video_path=video_path, start_sec=r.start, end_sec=r.end)
            if ai is None or ai.loading_probability >= cfg.ai_confidence_threshold:
                refined.append(r)
            else:
                logger.debug(
                    "AI rejected range %.3f-%.3f (prob=%.3f)",
                    r.start,
                    r.end,
                    ai.loading_probability,
                )
        result.loading_ranges = refined

    if logger.isEnabledFor(logging.DEBUG):
        for c in result.candidates:
            logger.debug(
                "candidate %.2f-%.2f score=%.2f freeze=%.2f low_info=%.2f",
                c.start,
                c.end,
                c.mean_score,
                c.freeze_ratio,
                c.low_info_ratio,
            )

    result.metrics_summary["coarse_elapsed_sec"] = float(max(0.0, t_coarse_end - t_coarse_start))
    result.metrics_summary["refine_elapsed_sec"] = float(max(0.0, t_refine_end - t_coarse_end))

    return result
