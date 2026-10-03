"""B0.2: detector throughput per route on the 3 MEVA test clips (same conf/imgsz, no stride).

For each route and clip: N frames after warm-up, timing (i) inference only (model.predict on a
decoded frame) and (ii) end-to-end (cv2 decode + predict), which is what a full pass costs.
Writes results/bench_routes.csv.   Usage: python code/bench_routes.py [N] [route ...]
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights_equiv import load_model, CONF, IMGSZ  # noqa: E402
from paths import MEVA_DIR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CLIPS = sorted((MEVA_DIR / "test3").glob("*.avi"))
WARMUP = 5


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    routes = sys.argv[2:] or ["torch_cpu", "ov_cpu_fp32", "ov_gpu_fp16", "ov_gpu_fp32"]
    out = REPO / "results" / "bench_routes.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    rows = [r for r in rows if r["route"] not in routes]
    for route in routes:
        model, dev = load_model(route)
        for clip in CLIPS:
            cap = cv2.VideoCapture(str(clip))
            for _ in range(WARMUP):
                ok, f = cap.read()
                model.predict(f, conf=CONF, imgsz=IMGSZ, device=dev, verbose=False)
            t_inf, t0 = 0.0, time.perf_counter()
            for _ in range(n):
                ok, f = cap.read()
                t1 = time.perf_counter()
                model.predict(f, conf=CONF, imgsz=IMGSZ, device=dev, verbose=False)
                t_inf += time.perf_counter() - t1
            t_e2e = time.perf_counter() - t0
            cap.release()
            rows.append(dict(route=route, clip=clip.name, n_frames=n,
                             fps_inference=n / t_inf, fps_end_to_end=n / t_e2e))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, float_format="%.4g")
    print(df.groupby("route")[["fps_inference", "fps_end_to_end"]].mean())


if __name__ == "__main__":
    main()
