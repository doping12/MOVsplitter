from __future__ import annotations

from typing import Iterable

from .utils import TimeRange, clamp_time_range


def merge_ranges(ranges: Iterable[TimeRange], gap_sec: float) -> list[TimeRange]:
    ordered = sorted(ranges, key=lambda x: x.start)
    if not ordered:
        return []

    merged: list[TimeRange] = [ordered[0]]
    for r in ordered[1:]:
        last = merged[-1]
        if r.start <= last.end + gap_sec:
            merged[-1] = TimeRange(last.start, max(last.end, r.end))
        else:
            merged.append(r)
    return merged


def filter_short_ranges(ranges: Iterable[TimeRange], min_sec: float) -> list[TimeRange]:
    return [r for r in ranges if r.duration >= min_sec]


def invert_ranges(removals: Iterable[TimeRange], duration: float) -> list[TimeRange]:
    clamped = sorted((clamp_time_range(r, duration) for r in removals), key=lambda x: x.start)
    out: list[TimeRange] = []
    cursor = 0.0
    for r in clamped:
        if r.start > cursor:
            out.append(TimeRange(cursor, r.start))
        cursor = max(cursor, r.end)
    if cursor < duration:
        out.append(TimeRange(cursor, duration))
    return out


def build_output_ranges(
    loading_ranges: list[TimeRange],
    duration: float,
    merge_gap_sec: float,
    loading_min_duration_sec: float,
    min_segment_sec: float,
) -> list[TimeRange]:
    merged_loading = merge_ranges(loading_ranges, merge_gap_sec)
    merged_loading = filter_short_ranges(merged_loading, loading_min_duration_sec)

    keep_ranges = invert_ranges(merged_loading, duration)
    keep_ranges = filter_short_ranges(keep_ranges, min_segment_sec)
    return keep_ranges
