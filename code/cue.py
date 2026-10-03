"""Frame-difference cue d_t (docs/STRATA_SPEC.md §5), identical on benchmark and MEVA.

Library: thumb(), raw_cue(), normalise().
Check: python code/cue.py --check-paths   (load scm configs/paths.yaml, assert the 4 dataset roots exist)
Main (benchmark): python code/cue.py [shard nshard] -> cache/bench_cue/<video>.parquet, then
  cache/bench_cue.parquet once all 131 videos are cached (resumable, shardable)
  columns: dataset, video, frame, mad, fpx   (raw; first frame of each segment = NaN)
  Frame indices are asserted equal to frame_scores.parquet / frame_scores_viratp.parquet.
"""
from __future__ import annotations
import paths
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import variant as V

SHORT_SIDE = 180
DIFF_THR = 15
NORM_WIN = 1800
MAD_FLOOR, FPX_FLOOR = 0.1, 1e-4

REPO = Path(__file__).resolve().parents[1]
CACHE = V.CACHE
PROC = paths.BENCH_PROC
SCM_DIR = paths.SCM_DIR
SCM_PATHS = SCM_DIR / "configs" / "paths.yaml"
ROOT_KEYS = ("cdnet_root", "lasiesta_root", "bmc_root", "virat_root")


def thumb(frame: np.ndarray) -> np.ndarray:
    """Gray, aspect-preserving resize so that the short side is 180 px."""
    h, w = frame.shape[:2]
    s = SHORT_SIDE / min(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(g, size, interpolation=interp)


def diff_stats(prev: np.ndarray, cur: np.ndarray):
    d = cv2.absdiff(cur, prev)
    return float(d.mean()), float(np.count_nonzero(d > DIFF_THR)) / d.size


def normalise(mad: np.ndarray, fpx: np.ndarray):
    """Causal normalisation within ONE segment: divide by the median of up to NORM_WIN previous
    values (excluding t). Index 0 has no cue (NaN). Index 1 has no history -> uses itself."""
    s_m, s_f = pd.Series(mad, dtype=float), pd.Series(fpx, dtype=float)
    med_m = s_m.rolling(NORM_WIN, min_periods=1).median().shift(1)
    med_f = s_f.rolling(NORM_WIN, min_periods=1).median().shift(1)
    med_m.iloc[:2] = s_m.iloc[:2]
    med_f.iloc[:2] = s_f.iloc[:2]
    dn = s_m / np.maximum(med_m, MAD_FLOOR)
    fn = s_f / np.maximum(med_f, FPX_FLOOR)
    return dn.to_numpy(), fn.to_numpy()


# ------------------------------------------------------------------ path check
def check_paths() -> None:
    """Load scm paths.yaml only and assert the 4 dataset roots exist (datasets moved 28-09-2026)."""
    import yaml
    cfg = yaml.safe_load(SCM_PATHS.read_text(encoding="utf-8"))
    for k in ROOT_KEYS:
        print(f"{k:14s} {cfg[k]}  exists={Path(cfg[k]).is_dir()}")
    assert all(Path(cfg[k]).is_dir() for k in ROOT_KEYS), "missing dataset root(s)"
    print("paths OK")


# ------------------------------------------------------------------ benchmark main
def bench_main():
    sys.path.insert(0, str(SCM_DIR / "scm"))
    import dump_p1_detections as dp
    cfg = dp.load_paths(str(SCM_PATHS))
    fs = pd.read_parquet(PROC / "frame_scores.parquet")
    fs = fs[fs.dataset.isin(["CDnet2014", "LASIESTA", "BMC"])]
    vp = pd.read_parquet(PROC / "frame_scores_viratp.parquet")
    recs = dp.collect_videos(cfg, ["CDnet2014", "LASIESTA", "BMC", "VIRAT"], None)
    # frame_scores uses the last path component as video name (VIRAT-P: file stem)
    by_name = {}
    for r in recs:
        ds = r[0].split("/")[0]
        by_name[("VIRAT-P" if ds == "VIRAT" else ds, r[0].split("/")[-1])] = r
    want = {(d, v): np.sort(g.frame.values) for (d, v), g in fs.groupby(["dataset", "video"])}
    want.update({("VIRAT-P", v): np.sort(g.frame.values) for v, g in vp.groupby("video")})
    assert len(want) == 131
    shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    vdir = CACHE / "bench_cue"
    vdir.mkdir(parents=True, exist_ok=True)
    items = sorted(want.items())
    for i, ((ds, v), frames) in enumerate(items):
        if i % nshard != shard or (vdir / f"{ds}__{v}.parquet").exists():
            continue
        vid_id, vpath, fdir, roi_s, roi_e = by_name[(ds, v)]
        if ds == "VIRAT-P":
            roi_e = -1                                  # full video (virat_full detections)
        fset = set(frames.tolist())
        rows, prev = [], None
        if V.CORRECTED and ds == "LASIESTA":             # AMENDMENT 01: numeric (temporal) file order
            it = ((t, cv2.imread(fp)) for t, fp in enumerate(V.natural_files(ds, v)))
        else:
            it = dp.make_frame_iter(vpath, fdir, roi_s, roi_e)
        for fidx, frame in it:
            if fidx not in fset:
                continue
            th = thumb(frame)
            m, f = diff_stats(prev, th) if prev is not None else (np.nan, np.nan)
            rows.append((fidx, m, f))
            prev = th
        got = np.array([r[0] for r in rows])
        assert np.array_equal(got, frames), (ds, v, len(got), len(frames))
        df = pd.DataFrame(rows, columns=["frame", "mad", "fpx"])
        df.insert(0, "video", v)
        df.insert(0, "dataset", ds)
        df.to_parquet(vdir / f"{ds}__{v}.parquet", index=False)
        print(f"[{i+1}/131] {ds}/{v}: {len(df)} frames", flush=True)
    done = sorted(vdir.glob("*.parquet"))
    if len(done) < len(items):
        print(f"shard {shard} finished; {len(done)}/{len(items)} videos cached", flush=True)
        return
    res = pd.concat([pd.read_parquet(f) for f in done], ignore_index=True)
    res["frame"] = res.frame.astype("int32")
    CACHE.mkdir(exist_ok=True)
    res.to_parquet(CACHE / "bench_cue.parquet", index=False)
    print("frames:", res.groupby("dataset").size().to_dict(), "total", len(res))


if __name__ == "__main__":
    if "--check-paths" in sys.argv:
        check_paths()
    else:
        bench_main()
