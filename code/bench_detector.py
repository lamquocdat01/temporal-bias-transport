"""Throughput of the g_t detector (YOLO26s-seg, conf 0.25, imgsz 640) on MEVA test clips.

Same weights/conf/imgsz as the benchmark detections (docs/DETECTOR_PROVENANCE.md).
Measures CPU (and CUDA if available) frames/s on N frames per clip after warm-up,
plus the cost of the frame-difference cue (gray, 320x180). Writes results/compute_bench.csv.
"""
from __future__ import annotations
import paths
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from paths import MEVA_DIR

WEIGHTS = paths.WEIGHTS
CLIPS = sorted((MEVA_DIR / "test3").glob("*.avi"))
REPO = Path(__file__).resolve().parents[1]
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
WARMUP = 5


def read_frames(path: Path, n: int):
    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < n + WARMUP:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def main():
    torch.manual_seed(42)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for clip in CLIPS:
        frames = read_frames(clip, N_FRAMES)
        # frame-difference cue cost
        t0 = time.perf_counter()
        prev = None
        for f in frames:
            g = cv2.cvtColor(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                d = cv2.absdiff(g, prev)
                _ = float(d.mean()), float((d > 15).mean())
            prev = g
        cue_fps = len(frames) / (time.perf_counter() - t0)
        for dev in devices:
            model = YOLO(str(WEIGHTS))
            for f in frames[:WARMUP]:
                model.predict(f, conf=0.25, imgsz=640, device=dev, verbose=False)
            t0 = time.perf_counter()
            nb = 0
            for f in frames[WARMUP:]:
                r = model.predict(f, conf=0.25, imgsz=640, device=dev, verbose=False)
                nb += len(r[0].boxes)
            dt = time.perf_counter() - t0
            n = len(frames) - WARMUP
            rows.append({"clip": clip.name, "device": dev, "n_frames": n, "sec": dt,
                         "det_fps": n / dt, "boxes_per_frame": nb / n,
                         "cue_fps_decode_excluded": cue_fps,
                         "torch_threads": torch.get_num_threads()})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "compute_bench.csv", index=False, float_format="%.4g")
    print(df.groupby("device").det_fps.agg(["mean", "min", "max"]))


if __name__ == "__main__":
    main()
