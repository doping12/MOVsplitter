from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


@dataclass(slots=True)
class FrameMetrics:
    time_sec: float
    brightness: float
    color_std: float
    laplacian_var: float
    edge_ratio: float
    hist_diff: float
    ssim: float
    frame_diff: float
    center_white_ratio: float
    center_edge_ratio: float
    center_outer_contrast: float


def frame_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def frame_color_std(frame_bgr: np.ndarray) -> float:
    return float(np.mean(np.std(frame_bgr.reshape(-1, 3), axis=0)))


def frame_histogram(frame_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def histogram_diff(h1: np.ndarray, h2: np.ndarray) -> float:
    # 0: ほぼ同一, 1: 大きく異なる
    return float(np.clip(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA), 0.0, 1.0))


def laplacian_variance(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_ratio(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def title_roi_features(frame_bgr: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y1, y2 = int(h * 0.35), int(h * 0.65)
    x1, x2 = int(w * 0.30), int(w * 0.70)
    center = gray[y1:y2, x1:x2]
    if center.size == 0:
        return 0.0, 0.0, 0.0

    white_ratio = float(np.mean(center >= 220))
    center_edges = cv2.Canny(center, 80, 180)
    center_edge_ratio = float(np.mean(center_edges > 0))

    outer_mask = np.ones_like(gray, dtype=np.uint8)
    outer_mask[y1:y2, x1:x2] = 0
    outer_vals = gray[outer_mask > 0]
    outer_mean = float(np.mean(outer_vals)) if outer_vals.size else float(np.mean(gray))
    center_mean = float(np.mean(center))
    contrast = center_mean - outer_mean
    return white_ratio, center_edge_ratio, contrast


def frame_diff_ratio(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff) / 255.0)


def frame_ssim(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    score = structural_similarity(prev_gray, curr_gray, data_range=255)
    return float(np.clip(score, -1.0, 1.0))


def extract_sampled_features(video_path: str | Path, sample_fps: float) -> tuple[list[FrameMetrics], float]:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / native_fps) if frame_count > 0 else 0.0

    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))

    metrics: list[FrameMetrics] = []
    prev_frame: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    frame_idx = 0
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

        metrics.append(
            FrameMetrics(
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
        )

        prev_frame = frame
        prev_hist = hist
        frame_idx += 1

    cap.release()

    if metrics and duration <= 0:
        duration = metrics[-1].time_sec

    return metrics, duration


def extract_sampled_features_window(
    video_path: str | Path,
    sample_fps: float,
    start_sec: float,
    end_sec: float,
) -> tuple[list[FrameMetrics], float]:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / native_fps) if frame_count > 0 else 0.0
    lo = max(0.0, float(start_sec))
    hi = min(duration, float(end_sec)) if duration > 0 else float(end_sec)
    if hi <= lo:
        cap.release()
        return [], duration

    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))
    start_frame = max(0, int(round(lo * native_fps)))
    end_frame = max(start_frame, int(round(hi * native_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    metrics: list[FrameMetrics] = []
    prev_frame: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    frame_idx = start_frame
    local_idx = 0

    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if local_idx % step != 0:
            frame_idx += 1
            local_idx += 1
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

        metrics.append(
            FrameMetrics(
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
        )
        prev_frame = frame
        prev_hist = hist
        frame_idx += 1
        local_idx += 1

    cap.release()
    return metrics, duration
