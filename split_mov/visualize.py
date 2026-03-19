from __future__ import annotations

from pathlib import Path
import base64
import io

import cv2

from .detect import DetectionResult
from .utils import TimeRange


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "可視化には matplotlib が必要です。`uv pip install matplotlib` を実行してください。"
        ) from e
    return plt


def _draw_timeline_figure(detection: DetectionResult, output_ranges: list[TimeRange]):
    plt = _import_matplotlib()
    times = [p["time_sec"] for p in detection.timeline]
    if not times:
        fig, ax = plt.subplots(figsize=(14, 4), dpi=120)
        ax.text(0.5, 0.5, "No timeline data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    score = [p["rule_score"] for p in detection.timeline]
    ssim = [p["ssim"] for p in detection.timeline]
    hist_diff = [p["hist_diff"] for p in detection.timeline]
    frame_diff = [p["frame_diff"] for p in detection.timeline]
    brightness = [p["brightness"] for p in detection.timeline]
    template_similarity = [p.get("template_similarity", 0.0) for p in detection.timeline]
    title_template_similarity = [p.get("title_template_similarity", 0.0) for p in detection.timeline]
    center_white_ratio = [p.get("center_white_ratio", 0.0) for p in detection.timeline]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), dpi=120, sharex=True)

    for ax in axes:
        for r in detection.loading_ranges:
            ax.axvspan(r.start, r.end, color="#ff6b6b", alpha=0.20)
        for r in output_ranges:
            ax.axvspan(r.start, r.end, color="#51cf66", alpha=0.12)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    axes[0].plot(times, score, color="#d9480f", label="rule_score")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right")

    axes[1].plot(times, ssim, color="#1c7ed6", label="ssim")
    axes[1].plot(times, hist_diff, color="#2b8a3e", label="hist_diff")
    axes[1].plot(times, frame_diff, color="#f08c00", label="frame_diff")
    axes[1].plot(times, template_similarity, color="#7048e8", label="template_similarity")
    axes[1].plot(times, title_template_similarity, color="#9c36b5", label="title_template_similarity")
    axes[1].set_ylabel("Similarity/Diff")
    axes[1].legend(loc="upper right")

    axes[2].plot(times, brightness, color="#495057", label="brightness")
    axes[2].plot(times, center_white_ratio, color="#e03131", label="center_white_ratio")
    axes[2].set_xlabel("Time (sec)")
    axes[2].set_ylabel("Brightness")
    axes[2].legend(loc="upper right")

    fig.suptitle("Loading Candidate Timeline (red=loading, green=kept)")
    fig.tight_layout()
    return fig


def export_check_png(path: str | Path, detection: DetectionResult, output_ranges: list[TimeRange]) -> None:
    plt = _import_matplotlib()
    fig = _draw_timeline_figure(detection, output_ranges)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png")
    plt.close(fig)


def export_check_html(path: str | Path, detection: DetectionResult, output_ranges: list[TimeRange]) -> None:
    plt = _import_matplotlib()
    fig = _draw_timeline_figure(detection, output_ranges)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    lines = [
        "<html><head><meta charset='utf-8'><title>split-mov check timeline</title></head><body>",
        "<h1>split-mov check timeline</h1>",
        "<p>赤: loading判定 / 緑: 出力(keep)区間</p>",
        f"<img style='max-width:100%;border:1px solid #ccc' src='data:image/png;base64,{img_b64}' />",
        "<h2>Loading ranges</h2>",
        "<ul>",
    ]
    for r in detection.loading_ranges:
        lines.append(f"<li>{r.start:.3f} - {r.end:.3f} ({r.duration:.3f}s)</li>")
    lines.extend(["</ul>", "<h2>Output ranges</h2>", "<ul>"])
    for r in output_ranges:
        lines.append(f"<li>{r.start:.3f} - {r.end:.3f} ({r.duration:.3f}s)</li>")
    lines.extend(["</ul>", "</body></html>"])

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def export_score_timeline_txt(path: str | Path, detection: DetectionResult) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "time_sec",
        "rule_score",
        "rule_score_delta",
        "score_skipped",
        "raw_loading_flag",
        "loading_flag",
        "template_similarity",
        "title_template_similarity",
        "ssim",
        "hist_diff",
        "frame_diff",
        "brightness",
    ]
    lines = ["\t".join(headers)]
    prev_score = None
    for p in detection.timeline:
        score = float(p.get("rule_score", 0.0))
        delta = 0.0 if prev_score is None else (score - prev_score)
        prev_score = score
        row = [
            f"{float(p.get('time_sec', 0.0)):.3f}",
            f"{score:.3f}",
            f"{delta:.3f}",
            f"{float(p.get('score_skipped', 0.0)):.0f}",
            f"{float(p.get('raw_loading_flag', p.get('loading_flag', 0.0))):.0f}",
            f"{float(p.get('loading_flag', 0.0)):.0f}",
            f"{float(p.get('template_similarity', 0.0)):.3f}",
            f"{float(p.get('title_template_similarity', 0.0)):.3f}",
            f"{float(p.get('ssim', 0.0)):.6f}",
            f"{float(p.get('hist_diff', 0.0)):.6f}",
            f"{float(p.get('frame_diff', 0.0)):.6f}",
            f"{float(p.get('brightness', 0.0)):.3f}",
        ]
        lines.append("\t".join(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_boundary_score_timeline_txt(path: str | Path, detection: DetectionResult) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "boundary_index",
        "window",
        "time_sec",
        "rule_score",
        "rule_score_delta",
        "score_skipped",
        "raw_loading_flag",
        "loading_flag",
        "template_similarity",
        "title_template_similarity",
        "ssim",
        "hist_diff",
        "frame_diff",
        "brightness",
    ]
    lines = ["\t".join(headers)]
    for bc in detection.boundary_checks:
        idx = int(bc.get("index", 0))
        for window_key, window_name in (("start_timeline", "start"), ("end_timeline", "end")):
            tl = bc.get(window_key, [])
            if not isinstance(tl, list):
                continue
            prev_score = None
            for p in tl:
                if not isinstance(p, dict):
                    continue
                score = float(p.get("rule_score", 0.0))
                delta = 0.0 if prev_score is None else (score - prev_score)
                prev_score = score
                row = [
                    str(idx),
                    window_name,
                    f"{float(p.get('time_sec', 0.0)):.3f}",
                    f"{score:.3f}",
                    f"{delta:.3f}",
                    f"{float(p.get('score_skipped', 0.0)):.0f}",
                    f"{float(p.get('raw_loading_flag', p.get('loading_flag', 0.0))):.0f}",
                    f"{float(p.get('loading_flag', 0.0)):.0f}",
                    f"{float(p.get('template_similarity', 0.0)):.3f}",
                    f"{float(p.get('title_template_similarity', 0.0)):.3f}",
                    f"{float(p.get('ssim', 0.0)):.6f}",
                    f"{float(p.get('hist_diff', 0.0)):.6f}",
                    f"{float(p.get('frame_diff', 0.0)):.6f}",
                    f"{float(p.get('brightness', 0.0)):.3f}",
                ]
                lines.append("\t".join(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_check_boundary_score_plots(
    out_dir: str | Path,
    detection: DetectionResult,
    window_sec: float = 5.0,
) -> list[Path]:
    plt = _import_matplotlib()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    times = [float(p["time_sec"]) for p in detection.timeline]
    scores = [float(p["rule_score"]) for p in detection.timeline]
    if not times:
        return []

    w = max(0.1, float(window_sec))
    written: list[Path] = []

    if detection.boundary_checks:
        for bc in detection.boundary_checks:
            i = int(bc.get("index", 0))
            st = bc.get("start_timeline", [])
            et = bc.get("end_timeline", [])
            if not st or not et:
                continue
            sx = [float(p.get("time_sec", 0.0)) for p in st]
            sy = [float(p.get("rule_score", 0.0)) for p in st]
            ex = [float(p.get("time_sec", 0.0)) for p in et]
            ey = [float(p.get("rule_score", 0.0)) for p in et]
            ssim_s = [float(p.get("ssim", 0.0)) for p in st]
            ssim_e = [float(p.get("ssim", 0.0)) for p in et]
            hist_s = [float(p.get("hist_diff", 0.0)) for p in st]
            hist_e = [float(p.get("hist_diff", 0.0)) for p in et]
            frame_s = [float(p.get("frame_diff", 0.0)) for p in st]
            frame_e = [float(p.get("frame_diff", 0.0)) for p in et]
            tmpl_s = [float(p.get("template_similarity", 0.0)) for p in st]
            tmpl_e = [float(p.get("template_similarity", 0.0)) for p in et]
            title_s = [float(p.get("title_template_similarity", 0.0)) for p in st]
            title_e = [float(p.get("title_template_similarity", 0.0)) for p in et]
            bri_s = [float(p.get("brightness", 0.0)) for p in st]
            bri_e = [float(p.get("brightness", 0.0)) for p in et]
            cwr_s = [float(p.get("center_white_ratio", 0.0)) for p in st]
            cwr_e = [float(p.get("center_white_ratio", 0.0)) for p in et]
            rs = float(bc.get("refined_start", sx[-1]))
            re = float(bc.get("refined_end", ex[0]))
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=120, sharex=True)

            ax0 = axes[0]
            ax0.plot(sx, sy, color="#d9480f", linewidth=1.6, label="start-window score")
            ax0.plot(ex, ey, color="#1971c2", linewidth=1.6, label="end-window score")
            ax0.plot([sx[-1], ex[0]], [sy[-1], ey[0]], color="#868e96", linestyle=":", linewidth=1.2, label="gap")
            ax0.axvline(rs, color="#e67700", linestyle="--", linewidth=1.0, label="refined start")
            ax0.axvline(re, color="#1c7ed6", linestyle="--", linewidth=1.0, label="refined end")
            ax0.set_ylabel("Score")
            ax0.grid(alpha=0.25, linestyle="--", linewidth=0.6)
            ax0.legend(loc="upper right")

            ax1 = axes[1]
            ax1.plot(sx, ssim_s, color="#1c7ed6", linewidth=1.2, label="start ssim")
            ax1.plot(ex, ssim_e, color="#1c7ed6", linewidth=1.2, linestyle="--", label="end ssim")
            ax1.plot(sx, hist_s, color="#2b8a3e", linewidth=1.2, label="start hist_diff")
            ax1.plot(ex, hist_e, color="#2b8a3e", linewidth=1.2, linestyle="--", label="end hist_diff")
            ax1.plot(sx, frame_s, color="#f08c00", linewidth=1.2, label="start frame_diff")
            ax1.plot(ex, frame_e, color="#f08c00", linewidth=1.2, linestyle="--", label="end frame_diff")
            ax1.plot(sx, tmpl_s, color="#7048e8", linewidth=1.2, label="start template")
            ax1.plot(ex, tmpl_e, color="#7048e8", linewidth=1.2, linestyle="--", label="end template")
            ax1.plot(sx, title_s, color="#9c36b5", linewidth=1.2, label="start title_template")
            ax1.plot(ex, title_e, color="#9c36b5", linewidth=1.2, linestyle="--", label="end title_template")
            ax1.set_ylabel("Similarity/Diff")
            ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)
            ax1.legend(loc="upper right", ncol=2, fontsize=8)

            ax2 = axes[2]
            ax2.plot(sx, bri_s, color="#495057", linewidth=1.2, label="start brightness")
            ax2.plot(ex, bri_e, color="#495057", linewidth=1.2, linestyle="--", label="end brightness")
            ax2.plot(sx, cwr_s, color="#e03131", linewidth=1.2, label="start center_white_ratio")
            ax2.plot(ex, cwr_e, color="#e03131", linewidth=1.2, linestyle="--", label="end center_white_ratio")
            ax2.set_xlabel("Time (sec)")
            ax2.set_ylabel("Brightness")
            ax2.grid(alpha=0.25, linestyle="--", linewidth=0.6)
            ax2.legend(loc="upper right")

            fig.suptitle(f"Boundary #{i} High-Precision Timeline")
            fig.tight_layout()
            p = out / f"boundary_{i}_combined.png"
            fig.savefig(p, format="png")
            plt.close(fig)
            written.append(p)
        return written

    for i, r in enumerate(detection.loading_ranges, start=1):
        x0 = max(times[0], float(r.start) - w)
        x1 = min(times[-1], float(r.end) + w)
        idx = [k for k, t in enumerate(times) if x0 <= t <= x1]
        if not idx:
            continue
        xs = [times[k] for k in idx]
        ys = [scores[k] for k in idx]
        fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
        ax.plot(xs, ys, color="#d9480f", linewidth=1.6, label="rule_score")
        ax.axvline(float(r.start), color="#e67700", linestyle="--", linewidth=1.0, label="start")
        ax.axvline(float(r.end), color="#1c7ed6", linestyle="--", linewidth=1.0, label="end")
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Score")
        ax.set_title(f"Boundary #{i} Score +/- {w:.1f}s")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        p = out / f"boundary_{i}_combined.png"
        fig.savefig(p, format="png")
        plt.close(fig)
        written.append(p)
    return written


def _capture_frame(video_path: str | Path, time_sec: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def _write_low_quality_jpeg(
    frame,
    out_path: Path,
    jpeg_quality: int,
    max_width: int,
) -> None:
    h, w = frame.shape[:2]
    if w > max_width > 0:
        scale = max_width / float(w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])


def export_check_frames_jpeg(
    video_path: str | Path,
    detection: DetectionResult,
    output_ranges: list[TimeRange],
    frames_dir: str | Path,
    jpeg_quality: int = 35,
    max_width: int = 480,
) -> list[Path]:
    out_dir = Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write_point(prefix: str, idx: int, point: str, sec: float) -> None:
        frame = _capture_frame(video_path, sec)
        if frame is None:
            return
        out = out_dir / f"{prefix}_{idx}_{point}_{sec:.3f}.jpg"
        _write_low_quality_jpeg(frame, out, jpeg_quality=jpeg_quality, max_width=max_width)
        written.append(out)

    for i, r in enumerate(detection.loading_ranges, start=1):
        write_point("loading", i, "start", r.start)
        write_point("loading", i, "end", max(r.start, r.end - 1e-3))

    for i, r in enumerate(output_ranges, start=1):
        write_point("keep", i, "start", r.start)
        write_point("keep", i, "end", max(r.start, r.end - 1e-3))

    return written


# Backward compatible aliases
export_debug_png = export_check_png
export_debug_html = export_check_html
export_boundary_frames_jpeg = export_check_frames_jpeg
