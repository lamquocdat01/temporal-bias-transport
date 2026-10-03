"""B0: detector-route equivalence against the committed benchmark detections (detections_cpu).

Re-runs YOLO26s-seg (conf 0.25, imgsz 640, no class filter) on two short benchmark videos with the
same frame source / temporalROI logic as scm/dump_p1_detections.py, then compares per frame:
box count, class, bbox (|d| <= 1 px), conf (|d| <= 1e-3), g_t(a) has-box, g_t(b) person >= 0.7,
and the number of (b) events.

Routes:  torch_cpu      .pt backup weights, PyTorch CPU FP32 (B0.1, must match 100 % on g(a), g(b))
         ov_cpu_fp32    OpenVINO IR FP32 on CPU                       (B0.2)
         ov_gpu_fp16    OpenVINO IR FP16 on Intel Iris Xe             (B0.2)

Usage: python code/weights_equiv.py <route> [<route> ...]
Writes results/weights_equiv_<route>.json and results/weights_equiv_frames_<route>.csv
"""
from __future__ import annotations
import paths
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(paths.SCM_DIR / "scm"))
import dump_p1_detections as dp  # noqa: E402  (read-only import of the original frame iterators)

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = paths.WEIGHTS
WEIGHTS_SHA = "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"
OV_DIR = paths.OV_DIR          # exported IR (outside cloud-sync folder)
DETROOT = paths.SCM_DIR / "data" / "detections_cpu"
CFG = dp.load_paths(str(paths.SCM_DIR / "configs" / "paths.yaml"))
CONF, IMGSZ, PERSON_CONF = 0.25, 640, 0.7
VIDEOS = [("CDnet2014", "CDnet2014/thermal/park"), ("LASIESTA", "LASIESTA/I_OC/I_OC_01")]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load_model(route: str):
    from ultralytics import YOLO
    if route == "torch_cpu":
        assert sha256(WEIGHTS) == WEIGHTS_SHA
        return YOLO(str(WEIGHTS), task="segment"), "cpu"
    if route == "ov_cpu_fp32":
        return YOLO(str(OV_DIR / "yolo26s-seg_fp32_openvino_model"), task="segment"), "intel:cpu"
    if route == "ov_gpu_fp16":
        return YOLO(str(OV_DIR / "yolo26s-seg_fp16_openvino_model"), task="segment"), "intel:gpu"
    if route in ("ov_gpu_fp32", "ov_gpu_fp32x"):
        # extra routes (not in the Dot-2 list): FP32 IR on Iris Xe. The GPU plugin defaults to
        # f16 inference precision whatever the IR is; "x" forces INFERENCE_PRECISION_HINT=f32.
        if route == "ov_gpu_fp32x":
            import openvino as ov
            _orig = ov.Core.compile_model
            if not getattr(ov.Core.compile_model, "_f32", False):
                def _cm(self, model, device_name=None, config=None, *a, **k):
                    config = dict(config or {})
                    if device_name and "GPU" in str(device_name).upper():
                        config["INFERENCE_PRECISION_HINT"] = "f32"
                    return _orig(self, model, device_name, config, *a, **k)
                _cm._f32 = True
                ov.Core.compile_model = _cm
        return YOLO(str(OV_DIR / "yolo26s-seg_fp32_openvino_model"), task="segment"), "intel:gpu"
    raise ValueError(route)


def predict(model, dev, frame):
    r = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=dev, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    d = r.boxes.data.cpu().numpy()
    # same serialisation as dump_p1_detections: int xyxy, conf rounded to 4 dp, cls int (cols 4,5)
    return [[*(int(v) for v in row[:4]), round(float(row[4]), 4), int(row[5])] for row in d]


def match(ref, new):
    """Greedy match per class by bbox L_inf distance. Returns (n_matched, max_dpx, max_dconf, all_matched)."""
    used, dpx, dcf, n = set(), 0.0, 0.0, 0
    for b in ref:
        best, bi = None, None
        for j, c in enumerate(new):
            if j in used or c[5] != b[5]:
                continue
            dist = max(abs(c[k] - b[k]) for k in range(4))
            if best is None or dist < best:
                best, bi = dist, j
        if bi is None:
            continue
        used.add(bi)
        n += 1
        dpx = max(dpx, best)
        dcf = max(dcf, abs(new[bi][4] - b[4]))
    return n, dpx, dcf, n == len(ref) == len(new)


def n_events(g):
    g = np.asarray(g, np.int8)
    return int(((np.diff(np.r_[0, g]) == 1)).sum())


def run(route: str):
    model, dev = load_model(route)
    records = {r[0]: r for r in dp.collect_videos(CFG, ["CDnet2014", "LASIESTA"], None)}
    rows, per_video, t_inf, n_inf = [], {}, 0.0, 0
    for ds, vid in VIDEOS:
        vid_id, vpath, fdir, roi_s, roi_e = records[vid]
        ref = {f["frame"]: f["boxes"] for f in
               json.load(open(DETROOT / ds / dp.video_json_name(vid_id, ds)))["frames"]}
        # warm-up (not timed)
        for _, fr in zip(range(3), dp.make_frame_iter(vpath, fdir, roi_s, roi_e)):
            predict(model, dev, fr[1])
        for fidx, frame in dp.make_frame_iter(vpath, fdir, roi_s, roi_e):
            t0 = time.perf_counter()
            new = predict(model, dev, frame)
            t_inf += time.perf_counter() - t0
            n_inf += 1
            rb = ref.get(fidx, [])
            nm, dpx, dcf, allm = match(rb, new)
            ga_r, ga_n = int(len(rb) > 0), int(len(new) > 0)
            gb_r = int(any(b[5] == 0 and b[4] >= PERSON_CONF for b in rb))
            gb_n = int(any(b[5] == 0 and b[4] >= PERSON_CONF for b in new))
            rows.append(dict(video=vid, frame=fidx, n_ref=len(rb), n_new=len(new), n_matched=nm,
                             all_matched=allm, max_dpx=dpx, max_dconf=dcf,
                             ga_ref=ga_r, ga_new=ga_n, gb_ref=gb_r, gb_new=gb_n))
        assert set(ref) == {r["frame"] for r in rows if r["video"] == vid}, "frame index mismatch"
    df = pd.DataFrame(rows)
    for vid, x in df.groupby("video"):
        per_video[vid] = dict(
            n_frames=len(x), frac_same_count=float((x.n_ref == x.n_new).mean()),
            frac_all_matched=float(x.all_matched.mean()),
            max_dpx=float(x.max_dpx.max()), max_dconf=float(x.max_dconf.max()),
            frac_bbox_le1px_conf_le1e3=float(((x.all_matched) & (x.max_dpx <= 1) & (x.max_dconf <= 1e-3)).mean()),
            g_a_agree=float((x.ga_ref == x.ga_new).mean()), g_b_agree=float((x.gb_ref == x.gb_new).mean()),
            events_b_ref=n_events(x.gb_ref), events_b_new=n_events(x.gb_new),
            events_a_ref=n_events(x.ga_ref), events_a_new=n_events(x.ga_new))
    tot = dict(route=route, n_frames=len(df), g_a_agree=float((df.ga_ref == df.ga_new).mean()),
               g_b_agree=float((df.gb_ref == df.gb_new).mean()),
               frac_all_boxes_within_tol=float(((df.all_matched) & (df.max_dpx <= 1) & (df.max_dconf <= 1e-3)).mean()),
               events_b_ref=sum(v["events_b_ref"] for v in per_video.values()),
               events_b_new=sum(v["events_b_new"] for v in per_video.values()),
               bench_fps_inference_only=n_inf / t_inf, per_video=per_video)
    tot["events_b_rel_diff"] = abs(tot["events_b_new"] - tot["events_b_ref"]) / max(1, tot["events_b_ref"])
    (REPO / "results" / f"weights_equiv_{route}.json").write_text(json.dumps(tot, indent=2))
    df.to_csv(REPO / "results" / f"weights_equiv_frames_{route}.csv", index=False)
    print(json.dumps(tot, indent=2), flush=True)


if __name__ == "__main__":
    for rt in sys.argv[1:] or ["torch_cpu"]:
        run(rt)
