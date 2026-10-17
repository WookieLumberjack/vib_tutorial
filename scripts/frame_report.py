"""Summarize a frame log (VIB_FRAME_STATS=frames.csv) by what was on screen.

    uv run python scripts/frame_report.py frames.csv [--min-frames 20]

The log is cut wherever the view changed (page, sub-tab, preset, window size)
or the user did something, and each stretch gets a line: its frames per second
and the median and 95th percentile of the computing (work) and drawing (paint)
per frame. Stretches with fewer than --min-frames frames are folded into the
next, their inputs still listed. Then a line per view, all its stretches
together, slowest paint first. A log from before the views and inputs were
logged is cut by page only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field

import numpy as np

LATE = 25.0  # ms between frames that counts as late (1.5 refreshes at 60 Hz)


@dataclass
class Stretch:
    start: float
    view: str
    inputs: list[str] = field(default_factory=list)
    frames: list[dict[str, str]] = field(default_factory=list)


def read(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if "kind" not in r:  # the first format: frames only, the page as the view
            r["kind"], r["detail"] = "frame", ""
    return rows


def stretches(rows: list[dict[str, str]], min_frames: int) -> list[Stretch]:
    out: list[Stretch] = []
    view = ""
    current: Stretch | None = None
    for r in rows:
        kind, t = r["kind"], float(r["wall_s"])
        if kind == "frame" and not view and (current is None or current.view != r["page"]):
            current = Stretch(t, r["page"])  # an old log: a new stretch per page
            out.append(current)
        if kind == "view":
            view = r["detail"]
        if kind in ("view", "input") or current is None:
            if current is not None and len(current.frames) < min_frames and out:
                current.view = view or current.view  # too short to tell: fold it into this one
                if kind == "input":
                    current.inputs.append(f"{t:7.1f}s  {r['detail']}")
                continue
            current = Stretch(t, view or r["page"])
            out.append(current)
            if kind == "input":
                current.inputs.append(f"{t:7.1f}s  {r['detail']}")
            continue
        if kind == "frame":
            current.frames.append(r)
    return [s for s in out if s.frames or s.inputs]


def stats(frames: list[dict[str, str]]) -> str:
    if len(frames) < 2:
        return f"{len(frames):6d} frames"
    iv = np.array([float(f["interval_ms"]) for f in frames])
    iv = iv[np.isfinite(iv)]
    work = np.array([float(f["work_ms"]) for f in frames])
    paint = np.array([float(f["paint_ms"]) for f in frames])
    fps = 1000.0 / iv.mean() if iv.size else float("nan")
    late = float(np.mean(iv > LATE)) if iv.size else 0.0
    return (f"{len(frames):6d} frames {fps:5.1f} fps  work {np.median(work):5.1f}/{np.percentile(work, 95):5.1f}"
            f"  paint {np.median(paint):5.1f}/{np.percentile(paint, 95):5.1f} ms  late {late:4.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--min-frames", type=int, default=20)
    args = parser.parse_args()
    rows = read(args.log)
    parts = stretches(rows, args.min_frames)
    print("Stretches (work and paint: median/95th percentile per frame)")
    for s in parts:
        print(f"{s.start:7.1f}s  {stats(s.frames)}  {s.view}")
        for line in s.inputs:
            print(f"           ↳ {line}")
    print("\nBy view, slowest paint first")
    views: dict[str, list[dict[str, str]]] = {}
    for s in parts:
        views.setdefault(s.view, []).extend(s.frames)
    ranked = sorted((v for v in views.items() if len(v[1]) >= 2),
                    key=lambda v: -np.median([float(f["paint_ms"]) for f in v[1]]))
    for view, frames in ranked:
        print(f"  {stats(frames)}  {view}")


if __name__ == "__main__":
    main()
