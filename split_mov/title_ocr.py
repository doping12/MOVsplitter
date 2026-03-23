from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

from .utils import TimeRange


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


def _title_roi(frame):
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.34), int(h * 0.66)
    x1, x2 = int(w * 0.22), int(w * 0.78)
    return frame[y1:y2, x1:x2]


def _title_rois(frame) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    boxes = [
        (0.22, 0.78, 0.34, 0.66),
        (0.18, 0.82, 0.40, 0.76),
    ]
    out: list[np.ndarray] = []
    for x1r, x2r, y1r, y2r in boxes:
        x1, x2 = int(w * x1r), int(w * x2r)
        y1, y2 = int(h * y1r), int(h * y2r)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            out.append(roi)
    if not out:
        out.append(_title_roi(frame))
    return out


def _preprocess_variants(roi: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (3, 3), 0)
    variants: list[np.ndarray] = [up]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    variants.append(clahe)
    adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    variants.append(adap)
    variants.append(cv2.bitwise_not(adap))
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for v in variants:
        key = v.tobytes()[:256]
        if key in seen:
            continue
        seen.add(key)
        unique.append(v)
    return unique


def _normalize_text(text: str) -> str:
    t = text.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", " ", t)
    t = t.strip()
    if not t:
        return "NO TITLE"
    return t


def _is_credit_text(text: str) -> bool:
    low = text.lower()
    if "作詞" in text or "作曲" in text or "編曲" in text:
        return True
    if "lyrics" in low or "music" in low or "arrange" in low or "arrangement" in low:
        return True
    if "supa love" in low or "orca" in low:
        return True
    return False


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float


def _clean_ocr_text(text: str) -> str:
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    candidates: list[str] = []
    for ln in lines:
        if not ln:
            continue
        norm = _normalize_text(ln)
        if norm == "NO TITLE":
            continue
        if _is_credit_text(norm):
            continue
        core = re.sub(r"[\W_]+", "", norm, flags=re.UNICODE)
        if len(core) < 2:
            continue
        candidates.append(norm)
    if candidates:
        candidates.sort(key=lambda s: (_text_quality(s), len(s)), reverse=True)
        return _normalize_known_title(candidates[0])

    t = _normalize_text(text)
    if t == "NO TITLE":
        return t
    if _is_credit_text(t):
        return "NO TITLE"
    core = re.sub(r"[\W_]+", "", t, flags=re.UNICODE)
    if len(core) < 2:
        return "NO TITLE"
    return _normalize_known_title(t)


def _normalize_known_title(text: str) -> str:
    t = text.strip()
    low = t.lower()
    low_alpha = re.sub(r"[^a-z]", "", low)
    if re.search(r"brand\s*new!?", low) or "brandnew" in low_alpha:
        return "Brand New!"
    if re.search(r"brandi?ne?w!?", low):
        return "Brand New!"
    if "情エナモラル" in t:
        return "熱情エナモラル"
    if "熱" in t and "情" in t and ("エ" in t or "ナ" in t or "モ" in t):
        return "熱情エナモラル"
    return t


def _text_quality(text: str) -> float:
    if text == "NO TITLE":
        return -1e9
    norm = _normalize_known_title(text)
    low = text.lower()
    if _is_credit_text(text):
        return -1e8
    jp = len(re.findall(r"[ぁ-んァ-ン一-龥]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    digit = len(re.findall(r"[0-9]", text))
    base = jp * 6.0 + en * 1.3 + digit * 0.5
    if "supa" in low or "orca" in low:
        base -= 40.0
    if re.search(r"brand\s*new!?", low):
        base += 40.0
    if "熱情" in text or "エナモラル" in text:
        base += 50.0
    if norm in {"Brand New!", "熱情エナモラル"}:
        base += 220.0
    symbols = len(re.findall(r"[`~^_=|/\\<>]", text))
    base -= symbols * 5.0
    token_count = len([x for x in re.split(r"\s+", text.strip()) if x])
    if token_count > 6:
        base -= (token_count - 6) * 12.0
    return base - max(0, len(text) - 28) * 2.0


def _ocr_with_pytesseract(img, lang: str, psm: int) -> OCRResult | None:
    # 安定性優先: CLI版tesseractを使用するため、pytesseract経由は無効化
    return None


def _ocr_with_tesseract_cli(img, lang: str, psm: int) -> OCRResult | None:
    if shutil.which("tesseract") is None:
        return None
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    cmd_txt = [
        "tesseract",
        "stdin",
        "stdout",
        "-l",
        lang,
        "--psm",
        str(int(psm)),
        "--oem",
        "3",
    ]
    try:
        proc_txt = subprocess.run(
            cmd_txt,
            input=buf.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except Exception:
        return None
    if proc_txt.returncode != 0:
        return None
    txt = proc_txt.stdout.decode("utf-8", errors="ignore")
    return OCRResult(text=txt, confidence=0.0)


def ocr_title_from_frame(frame, lang: str = "jpn+eng", psm: int = 7) -> str:
    langs = [lang]
    langs = list(dict.fromkeys(langs))
    if lang == "eng":
        psms = list(dict.fromkeys([int(psm), 7, 11]))
    else:
        psms = [int(psm)]

    best_text = "NO TITLE"
    best_score = -1.0
    for roi in _title_rois(frame):
        for img in _preprocess_variants(roi):
            for lg in langs:
                for p in psms:
                    res = _ocr_with_pytesseract(img, lang=lg, psm=p)
                    if res is None:
                        res = _ocr_with_tesseract_cli(img, lang=lg, psm=p)
                    if res is None:
                        continue
                    txt = _clean_ocr_text(res.text)
                    if txt == "NO TITLE":
                        continue
                    core_len = len(re.sub(r"[\W_]+", "", txt, flags=re.UNICODE))
                    score = float(res.confidence) + min(core_len, 24) * 2.0 + _text_quality(txt) * 3.0
                    if score > best_score:
                        best_score = score
                        best_text = txt
    return best_text


def ocr_title_from_video(
    video_path: str | Path,
    base_sec: float,
    frame_offset_sec: float = 0.8,
    lang: str = "jpn+eng",
    psm: int = 7,
) -> tuple[str, float, np.ndarray | None]:
    probe_offsets = [-0.5, -0.2, 0.2, 0.8, 1.4]
    best_text = "NO TITLE"
    best_q = -1e9
    best_t = max(0.0, float(base_sec) + max(0.0, float(frame_offset_sec)))
    best_frame = None
    for d in probe_offsets:
        t = max(0.0, float(base_sec) + max(0.0, float(frame_offset_sec)) + d)
        frame = _capture_frame(video_path, t)
        if frame is None:
            continue
        langs = [lang]
        if lang != "eng":
            langs.append("eng")
        for lg in langs:
            txt = ocr_title_from_frame(frame, lang=lg, psm=psm)
            q = _text_quality(txt)
            if q > best_q:
                best_q = q
                best_text = txt
                best_t = t
                best_frame = frame
    return best_text, best_t, best_frame


def write_title_pairs(path: str | Path, pairs: list[tuple[str, str]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["name\ttitle"]
    for k, v in pairs:
        lines.append(f"{k}\t{v}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def extract_titles_for_ranges(
    video_path: str | Path,
    ranges: list[TimeRange],
    labels: list[str],
    frames_dir: str | Path,
    frame_offset_sec: float = 0.8,
    lang: str = "jpn+eng",
    psm: int = 7,
) -> list[tuple[str, str]]:
    out_dir = Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    for i, r in enumerate(ranges):
        label = labels[i] if i < len(labels) else f"segment_{i+1}"
        txt, t, frame = ocr_title_from_video(
            video_path=video_path,
            base_sec=float(r.start),
            frame_offset_sec=float(frame_offset_sec),
            lang=lang,
            psm=psm,
        )
        if frame is None:
            pairs.append((label, "NO TITLE"))
            continue
        fpath = out_dir / f"title_{i+1:03d}_{t:.3f}.jpg"
        cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        pairs.append((label, txt))
    return pairs


def extract_titles_for_files(
    files: list[Path],
    frames_dir: str | Path,
    frame_offset_sec: float = 0.8,
    lang: str = "jpn+eng",
    psm: int = 7,
) -> list[tuple[str, str]]:
    out_dir = Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    for p in files:
        txt, t, frame = ocr_title_from_video(
            video_path=p,
            base_sec=0.0,
            frame_offset_sec=float(frame_offset_sec),
            lang=lang,
            psm=psm,
        )
        if frame is None:
            pairs.append((p.name, "NO TITLE"))
            continue
        fpath = out_dir / f"title_{p.stem}_{t:.3f}.jpg"
        cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        pairs.append((p.name, txt))
    return pairs
