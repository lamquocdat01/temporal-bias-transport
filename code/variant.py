"""Result variant (prereg/AMENDMENT_01.md): NINE_VARIANT = orig (default) | corrected.

orig       the pre-registered run as committed (results/orig/, cache/, prereg/PREREG_MEVA.json); read-only
corrected  LASIESTA frames in numeric (temporal) order: detections re-indexed, frame_scores g_t permuted, events
           rebuilt, cue recomputed from the images in that order (results/corrected/, cache/corrected/,
           prereg/PREREG_MEVA_corrected.json). Every other input is identical.

The only data change is lasiesta_order(); CDnet2014 and BMC are asserted lex == numeric (checklist B10).
"""
from __future__ import annotations
import paths
import functools
import glob as _glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT = os.environ.get("NINE_VARIANT", "orig")
assert VARIANT in ("orig", "corrected"), VARIANT
CORRECTED = VARIANT == "corrected"

REPO = Path(__file__).resolve().parents[1]
# overrides used by the profile-only reproduction (release make_numbers.py)
RES = Path(os.environ.get("NINE_RES") or REPO / "results" / VARIANT)
RES_ORIG = Path(os.environ.get("NINE_RES_ORIG") or REPO / "results" / "orig")
PRE_DIR_REL = os.environ.get("NINE_PREREG_DIR", "prereg")
PRE_DIR = REPO / PRE_DIR_REL
MEVA_WINDOWS = os.environ.get("NINE_MEVA_WINDOWS")     # profile table replacing the MEVA frame cache
CACHE = REPO / "cache" / "corrected" if CORRECTED else REPO / "cache"
PRE_NAME = "PREREG_MEVA_corrected" if CORRECTED else "PREREG_MEVA"
PRE_JSON = PRE_DIR / f"{PRE_NAME}.json"
PRE_SHA = PRE_DIR / f"{PRE_NAME}.json.sha256"
PRE_MD = PRE_DIR / f"{PRE_NAME}.md"
PRE_REL = f"{PRE_DIR_REL}/{PRE_NAME}.json"
NUMBERS = RES / ("numbers_corrected.json" if CORRECTED else "numbers.json")

PROC = paths.BENCH_PROC
SCM_DIR = paths.SCM_DIR
DET = SCM_DIR / "data"
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
EXPECT_EVENTS_A = {"CDnet2014": 2516, "LASIESTA": 393 if CORRECTED else 936, "BMC": 261}

RES.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def rel(p: Path) -> str:
    return str(Path(p).relative_to(REPO)).replace("\\", "/")


# ------------------------------------------------------------------ frame order
def lex_files(fdir: str) -> list[str]:
    """File order of scm dump_p1_detections.iter_frames_dir (sorted(set(glob)), i.e. lexicographic)."""
    files = []
    for ext in EXTS:
        files += _glob.glob(os.path.join(fdir, ext)) + _glob.glob(os.path.join(fdir, ext.upper()))
    return sorted(set(files))


def file_num(fp: str) -> int:
    return int(re.findall(r"(\d+)", os.path.basename(fp))[-1])


@functools.lru_cache(maxsize=None)
def _records():
    import sys
    sys.path.insert(0, str(SCM_DIR / "scm"))
    import dump_p1_detections as dp
    cfg = dp.load_paths(str(SCM_DIR / "configs" / "paths.yaml"))
    return {(r[0].split("/")[0], r[0].split("/")[-1]): r for r in dp.collect_videos(cfg, ["CDnet2014", "LASIESTA", "BMC"], None)}


@functools.lru_cache(maxsize=None)
def lasiesta_order(video: str) -> np.ndarray:
    """order[t] = lex position of the frame with temporal index t (file number t + 1)."""
    files = lex_files(_records()[("LASIESTA", video)][2])
    t = np.array([file_num(f) - 1 for f in files])
    assert np.array_equal(np.sort(t), np.arange(len(files))), f"LASIESTA {video}: numbering not 1..n"
    return np.argsort(t)


def natural_files(ds: str, video: str) -> list[str]:
    files = lex_files(_records()[(ds, video)][2])
    if ds == "LASIESTA" and CORRECTED:
        return [files[i] for i in lasiesta_order(video)]
    return files


def assert_lex_is_numeric(datasets=("CDnet2014", "BMC")) -> int:
    n = 0
    for (ds, v), r in _records().items():
        if ds in datasets:
            nums = [file_num(f) for f in lex_files(r[2])]
            assert nums == sorted(nums), f"{ds}/{v}: lex order != numeric order"
            n += 1
    return n


# ------------------------------------------------------------------ benchmark inputs (read-only sources)
def load_det_raw(path: Path) -> dict:
    return {fr["frame"]: fr.get("boxes", []) for fr in json.load(open(path))["frames"]}


def bench_det(ds: str, key: str) -> dict:
    """detections_cpu of one benchmark video, frame -> boxes; LASIESTA re-indexed to temporal order (corrected)."""
    det = load_det_raw(DET / "detections_cpu" / ds / f"{key}.json")
    if ds == "LASIESTA" and CORRECTED:
        order = lasiesta_order(key.split("__")[-1])
        assert len(det) == len(order), (key, len(det), len(order))
        det = {t: det.get(int(i), []) for t, i in enumerate(order)}
    return det


def frame_scores() -> pd.DataFrame:
    fs = pd.read_parquet(PROC / "frame_scores.parquet")
    if not CORRECTED:
        return fs
    out = []
    for (ds, v), x in fs.groupby(["dataset", "video"], sort=False):
        if ds == "LASIESTA":
            x = x.sort_values("frame").copy()
            assert np.array_equal(x.frame.values, np.arange(len(x)))
            x["g_t"] = x.g_t.values[lasiesta_order(v)]
        out.append(x)
    return pd.concat(out, ignore_index=True)


def events_durations() -> pd.DataFrame:
    """Deep3 events table (definition (a)); corrected: LASIESTA rows rebuilt from the re-ordered g_t."""
    ev = pd.read_csv(PROC / "events_durations.csv")
    if not CORRECTED:
        return ev
    fs = frame_scores()
    fs = fs[fs.dataset == "LASIESTA"]
    keys = {v.split("__")[-1]: v for v in ev[ev.dataset == "LASIESTA"].video.unique()}
    rows = []
    for v, x in fs.groupby("video"):
        g = x.sort_values("frame").g_t.values.astype(np.int8)
        d = np.diff(np.r_[0, g, 0])
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        rows += [dict(dataset="LASIESTA", video=keys[v], event_id=i, duration=int(b - a)) for i, (a, b) in enumerate(zip(s, e))]
    return pd.concat([ev[ev.dataset != "LASIESTA"], pd.DataFrame(rows)], ignore_index=True)
