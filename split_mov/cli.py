from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import shutil
import time

import yaml

from .ai_helper import OptionalAIHelper
from .config import load_config
from .cut_table import build_rows_from_checked_dir, read_cut_table_csv, rows_to_ranges, write_cut_table_csv
from .detect import detect_loading_ranges
from .ffmpeg_utils import split_video_ranges
from .report import build_report, write_preview_csv, write_report_json
from .segment import build_output_ranges
from .title_ocr import extract_titles_for_files, extract_titles_for_ranges, write_title_pairs
from .utils import detect_output_ext, is_supported_input, output_path_for_index
from .visualize import (
    export_boundary_score_timeline_txt,
    export_check_boundary_score_plots,
    export_check_frames_jpeg,
    export_check_html,
    export_check_png,
    export_score_timeline_txt,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MOV/MP4動画からローディング画面を除外して分割する")
    p.add_argument("input_file", nargs="?", help="入力動画 (.mp4 / .mov)")
    p.add_argument("--output-dir", default=None, help="出力先ディレクトリ")
    p.add_argument("--build-cut-table-from-dir", default=None, help="JPEG確認済みディレクトリから切り出し表を作成（フロー1）")
    p.add_argument("--cut-table", default=None, help="切り出し表CSVを使って分割（フロー2）")
    p.add_argument("--cut-table-out", default=None, help="フロー1で作るCSVの出力先")
    p.add_argument("--titles-from-dir", default=None, help="動画ファイル群からタイトルOCR一覧を作成")
    p.add_argument("--extract-titles", action="store_true", help="分割前動画のkeep区間に対応するタイトルOCR一覧を作成")
    p.add_argument("--titles-output", default="titles.txt", help="タイトルOCR結果の2カラム出力先")
    p.add_argument("--title-frame-offset-sec", type=float, default=0.8, help="タイトルOCRに使うフレーム時刻オフセット(秒)")
    p.add_argument("--title-ocr-lang", default="jpn+eng", help="OCR言語")
    p.add_argument("--title-ocr-psm", type=int, default=7, help="OCR PSM")
    p.add_argument("--title-ocr-backend", choices=["tesseract", "easyocr"], default="tesseract", help="タイトルOCRバックエンド")
    p.add_argument("--parallel", type=int, default=1, help="切り出し並列数（CPU/IO並列）")
    p.add_argument("--config", default=None, help="YAML/JSON設定ファイル")
    p.add_argument("--min-segment-sec", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--export-report", default=None)
    p.add_argument("--export-preview", default=None)
    p.add_argument("--check-boundary-window-sec", type=float, default=5.0, help="境目前後plotの片側秒数")
    p.add_argument("--check-jpeg-quality", type=int, default=30, help="check JPEG品質 (1-95, 低いほど軽量)")
    p.add_argument("--check-frame-width", type=int, default=360, help="check JPEGの最大幅(px)")
    p.add_argument("--visualize-only", action="store_true", help="分割出力は行わず、解析と可視化のみ行う")
    p.add_argument("--keep-temp", action="store_true", help="将来拡張用（現時点で一時ファイルは作成しない）")
    p.add_argument("--ai", action="store_true", help="AI補助判定を有効化")
    p.add_argument("--output-ext", choices=["mp4", "mov"], default=None, help="出力拡張子（既定は入力拡張子）")
    return p


def build_default_run_dir(cwd: Path, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return (cwd / f"split_mov_{stamp}").resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    run_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_dir = build_default_run_dir(Path.cwd())
        output_dir = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / "execution_timing.txt"

    def _fmt_hms(sec: float) -> str:
        v = int(max(0, round(float(sec))))
        h = v // 3600
        m = (v % 3600) // 60
        s = v % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _resolve_optional_output_path(value: str | None) -> Path | None:
        if value is None:
            return None
        p = Path(value).expanduser()
        if p.is_absolute():
            return p
        base = output_dir
        return (base / p).resolve()

    if args.titles_from_dir:
        src_dir = Path(args.titles_from_dir).expanduser().resolve()
        if not src_dir.exists():
            raise FileNotFoundError(f"titles-from-dir が存在しません: {src_dir}")
        files = sorted(
            [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov"}],
            key=lambda x: x.name.lower(),
        )
        if not files:
            raise RuntimeError(f"動画ファイル(.mp4/.mov)が見つかりません: {src_dir}")
        titles_out = _resolve_optional_output_path(args.titles_output) or (output_dir / "titles.txt")
        frames_dir = output_dir / "title_frames"
        pairs = extract_titles_for_files(
            files=files,
            frames_dir=frames_dir,
            frame_offset_sec=float(args.title_frame_offset_sec),
            lang=str(args.title_ocr_lang),
            psm=int(args.title_ocr_psm),
            backend=str(args.title_ocr_backend),
        )
        write_title_pairs(titles_out, pairs)
        print(f"title_frames_dir: {frames_dir}")
        print(f"titles_output: {titles_out}")
        for k, v in pairs:
            print(f"{k}\t{v}")
        return 0

    # フロー1: JPEG確認済みディレクトリ -> 切り出し表CSV
    if args.build_cut_table_from_dir:
        checked_dir = Path(args.build_cut_table_from_dir).expanduser().resolve()
        if not checked_dir.exists():
            raise FileNotFoundError(f"確認ディレクトリが存在しません: {checked_dir}")
        rows = build_rows_from_checked_dir(checked_dir)
        if not rows:
            raise RuntimeError(f"keep_*.jpg から有効な区間を抽出できませんでした: {checked_dir}")

        table_out = _resolve_optional_output_path(args.cut_table_out)
        if table_out is None:
            table_out = output_dir / "cut_table.csv"

        write_cut_table_csv(table_out, rows)
        print(f"checked_dir: {checked_dir}")
        print(f"cut_table: {table_out}")
        print("rows:")
        for r in rows:
            print(f"  - seg={r.segment} start={r.start_sec:.3f} end={r.end_sec:.3f} dur={r.duration_sec:.3f} enabled={r.enabled}")
        return 0

    if not args.input_file:
        raise ValueError("入力動画が必要です（または --build-cut-table-from-dir を指定してください）")

    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが存在しません: {input_path}")
    if not is_supported_input(input_path):
        raise ValueError("入力は .mp4 または .mov のみ対応です")

    # フロー2: 切り出し表CSV + 入力動画 -> 実切り出し
    if args.cut_table:
        table_path = Path(args.cut_table).expanduser().resolve()
        if not table_path.exists():
            raise FileNotFoundError(f"cut table が存在しません: {table_path}")

        rows = read_cut_table_csv(table_path)
        keep_ranges = rows_to_ranges(rows)
        if not keep_ranges:
            raise RuntimeError("cut table に有効な enabled=1 区間がありません")

        output_ext = detect_output_ext(input_path, args.output_ext)
        stem = input_path.stem
        planned_files = [output_path_for_index(output_dir, stem, i + 1, output_ext) for i in range(len(keep_ranges))]

        print(f"input: {input_path}")
        print(f"output_dir: {output_dir}")
        print(f"cut_table: {table_path}")
        print("table_ranges:")
        for i, r in enumerate(keep_ranges, start=1):
            print(f"  - #{i}: {r.start:.3f} -> {r.end:.3f} ({r.duration:.3f}s)")

        print("planned_output_files:")
        for pth in planned_files:
            print(f"  - {pth}")

        if args.extract_titles:
            titles_out = _resolve_optional_output_path(args.titles_output) or (output_dir / "titles.txt")
            frames_dir = output_dir / "title_frames"
            labels = [p.name for p in planned_files]
            pairs = extract_titles_for_ranges(
                video_path=input_path,
                ranges=keep_ranges,
                labels=labels,
                frames_dir=frames_dir,
                frame_offset_sec=float(args.title_frame_offset_sec),
                lang=str(args.title_ocr_lang),
                psm=int(args.title_ocr_psm),
                backend=str(args.title_ocr_backend),
            )
            write_title_pairs(titles_out, pairs)
            print(f"title_frames_dir: {frames_dir}")
            print(f"titles_output: {titles_out}")

        if args.dry_run or args.visualize_only:
            return 0

        print(f"video_split start (parallel={max(1, int(args.parallel))})")
        t_split_start = time.perf_counter()
        split_video_ranges(
            input_file=input_path,
            output_dir=output_dir,
            stem=stem,
            output_ext=output_ext,
            keep_ranges=keep_ranges,
            cfg=load_config(args.config),
            parallel=max(1, int(args.parallel)),
        )
        split_elapsed = max(0.0, time.perf_counter() - t_split_start)
        print(f"video_split elapsed: {split_elapsed:.3f}s")
        timing_path.write_text(
            "\n".join(
                [
                    "coarse_detection_elapsed_sec\t0.000",
                    "high_precision_detection_elapsed_sec\t0.000",
                    f"video_split_elapsed_sec\t{split_elapsed:.3f}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    overrides = {
        "min_segment_sec": args.min_segment_sec,
        "ai_enabled": args.ai,
    }
    cfg = load_config(args.config, overrides=overrides)
    if args.config:
        src_cfg = Path(args.config).expanduser().resolve()
        if src_cfg.exists():
            shutil.copy2(src_cfg, output_dir / f"used_config{src_cfg.suffix.lower() or '.yaml'}")
    else:
        (output_dir / "used_config.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    ai = OptionalAIHelper(cfg)
    print("coarse_detection start")

    def _on_coarse_loading_range(r):
        print(f"loading検出されました (範囲 {_fmt_hms(r.start)} ~ {_fmt_hms(r.end)})")

    def _on_refine_start(workers: int):
        print(f"high_precision_detection start (parallel={workers})")

    detection = detect_loading_ranges(
        str(input_path),
        cfg,
        ai_helper=ai,
        on_coarse_loading_range=_on_coarse_loading_range,
        on_refine_start=_on_refine_start,
    )

    output_ranges = build_output_ranges(
        loading_ranges=detection.loading_ranges,
        duration=detection.duration,
        merge_gap_sec=cfg.merge_gap_sec,
        loading_min_duration_sec=cfg.loading_min_duration_sec,
        min_segment_sec=cfg.min_segment_sec,
    )

    output_ext = detect_output_ext(input_path, args.output_ext)
    stem = input_path.stem

    planned_files = [
        output_path_for_index(output_dir, stem, i + 1, output_ext)
        for i in range(len(output_ranges))
    ]

    report_path = _resolve_optional_output_path(args.export_report)
    preview_path = _resolve_optional_output_path(args.export_preview)
    check_png_path = output_dir / "check_timeline.png"
    check_html_path = output_dir / "check_timeline.html"
    score_txt_path = output_dir / "score_timeline.txt"
    refine_score_txt_path = output_dir / "score_timeline_refine.txt"
    check_frames_dir = output_dir / "check_frames"
    check_boundary_plots_dir = output_dir / "check_boundary_plots"

    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"duration: {detection.duration:.3f}s")
    print("detected_loading_ranges:")
    for r in detection.loading_ranges:
        print(f"  - {r.start:.3f} -> {r.end:.3f} ({r.duration:.3f}s)")

    print("output_ranges:")
    for i, r in enumerate(output_ranges, start=1):
        print(f"  - #{i}: {r.start:.3f} -> {r.end:.3f} ({r.duration:.3f}s)")

    print("planned_output_files:")
    for pth in planned_files:
        print(f"  - {pth}")

    if args.extract_titles:
        titles_out = _resolve_optional_output_path(args.titles_output) or (output_dir / "titles.txt")
        frames_dir = output_dir / "title_frames"
        labels = [p.name for p in planned_files]
        pairs = extract_titles_for_ranges(
            video_path=input_path,
            ranges=output_ranges,
            labels=labels,
            frames_dir=frames_dir,
            frame_offset_sec=float(args.title_frame_offset_sec),
            lang=str(args.title_ocr_lang),
            psm=int(args.title_ocr_psm),
            backend=str(args.title_ocr_backend),
        )
        write_title_pairs(titles_out, pairs)
        print(f"title_frames_dir: {frames_dir}")
        print(f"titles_output: {titles_out}")

    skip_split = args.dry_run or args.visualize_only
    export_check_png(check_png_path, detection, output_ranges)
    export_check_html(check_html_path, detection, output_ranges)
    export_score_timeline_txt(score_txt_path, detection)
    export_boundary_score_timeline_txt(refine_score_txt_path, detection)
    plots = export_check_boundary_score_plots(
        out_dir=check_boundary_plots_dir,
        detection=detection,
        window_sec=float(args.check_boundary_window_sec),
    )
    print(f"check_boundary_plots: {len(plots)} files")
    for p in plots:
        print(f"  - {p}")
    files = export_check_frames_jpeg(
        video_path=input_path,
        detection=detection,
        output_ranges=output_ranges,
        frames_dir=check_frames_dir,
        jpeg_quality=max(1, min(95, int(args.check_jpeg_quality))),
        max_width=max(64, int(args.check_frame_width)),
    )
    print(f"check_frame_jpegs: {len(files)} files")
    for p in files:
        print(f"  - {p}")
    # JPEG出力時は同時に同ディレクトリ系統へ cut table を作る
    rows = build_rows_from_checked_dir(check_frames_dir)
    auto_table = _resolve_optional_output_path(args.cut_table_out)
    if auto_table is None:
        auto_table = Path(check_frames_dir).resolve().parent / "cut_table.csv"
    write_cut_table_csv(auto_table, rows)
    print(f"cut_table: {auto_table}")

    output_files = planned_files
    if not skip_split:
        print(f"video_split start (parallel={max(1, int(args.parallel))})")
        t_split_start = time.perf_counter()
        output_files = split_video_ranges(
            input_file=input_path,
            output_dir=output_dir,
            stem=stem,
            output_ext=output_ext,
            keep_ranges=output_ranges,
            cfg=cfg,
            parallel=max(1, int(args.parallel)),
        )
        split_elapsed = max(0.0, time.perf_counter() - t_split_start)
    else:
        split_elapsed = 0.0

    report = build_report(
        input_file=input_path,
        detection=detection,
        output_ranges=output_ranges,
        output_files=output_files,
        cfg=cfg,
    )

    if report_path:
        write_report_json(report_path, report)
    if preview_path:
        write_preview_csv(preview_path, detection.loading_ranges, output_ranges)

    coarse_elapsed = float(detection.metrics_summary.get("coarse_elapsed_sec", 0.0))
    refine_elapsed = float(detection.metrics_summary.get("refine_elapsed_sec", 0.0))
    print(f"coarse_detection elapsed: {coarse_elapsed:.3f}s")
    print(f"high_precision_detection elapsed: {refine_elapsed:.3f}s")
    if not skip_split:
        print(f"video_split elapsed: {split_elapsed:.3f}s")
    timing_path.write_text(
        "\n".join(
            [
                f"coarse_detection_elapsed_sec\t{coarse_elapsed:.3f}",
                f"high_precision_detection_elapsed_sec\t{refine_elapsed:.3f}",
                f"video_split_elapsed_sec\t{split_elapsed:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return 0
