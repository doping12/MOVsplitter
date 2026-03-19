from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re

from .utils import TimeRange

KEEP_RE = re.compile(r"keep_(\d+)_(start|end)_([0-9]+(?:\.[0-9]+)?)\.jpg$", re.IGNORECASE)


@dataclass(slots=True)
class CutRow:
    segment: int
    start_sec: float
    end_sec: float
    enabled: int = 1
    note: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def build_rows_from_checked_dir(checked_dir: str | Path) -> list[CutRow]:
    root = Path(checked_dir)
    files = list(root.rglob("keep_*_*.jpg"))

    buf: dict[int, dict[str, float]] = {}
    for p in files:
        m = KEEP_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        side = m.group(2).lower()
        sec = float(m.group(3))
        buf.setdefault(idx, {})[side] = sec

    rows: list[CutRow] = []
    for idx in sorted(buf.keys()):
        d = buf[idx]
        if "start" not in d or "end" not in d:
            continue
        s = min(d["start"], d["end"])
        e = max(d["start"], d["end"])
        rows.append(CutRow(segment=idx, start_sec=s, end_sec=e, enabled=1, note="from_checked_dir"))
    return rows


def write_cut_table_csv(path: str | Path, rows: list[CutRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "start_sec", "end_sec", "duration_sec", "enabled", "note"])
        for r in rows:
            w.writerow([
                r.segment,
                f"{r.start_sec:.3f}",
                f"{r.end_sec:.3f}",
                f"{r.duration_sec:.3f}",
                int(r.enabled),
                r.note,
            ])


def read_cut_table_csv(path: str | Path) -> list[CutRow]:
    rows: list[CutRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                CutRow(
                    segment=int(row["segment"]),
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    enabled=int(float(row.get("enabled", "1") or 1)),
                    note=row.get("note", ""),
                )
            )
    rows.sort(key=lambda x: x.segment)
    return rows


def rows_to_ranges(rows: list[CutRow]) -> list[TimeRange]:
    out: list[TimeRange] = []
    for r in rows:
        if int(r.enabled) != 1:
            continue
        if r.end_sec <= r.start_sec:
            continue
        out.append(TimeRange(r.start_sec, r.end_sec))
    return out
