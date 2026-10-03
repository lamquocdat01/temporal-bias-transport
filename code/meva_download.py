"""Download the PRIMARY MEVA clips of selection v1 (docs/MEVA_selection.md) and write a manifest.

Content-agnostic: only file integrity is checked (R7-E1: size vs inventory, SHA-256, cv2 opens,
header frame count). The decoded-frame-count half of R7-E1 is checked in the cue pass
(code/meva_cue.py), which decodes every frame anyway.
Pauses if free space would drop below 20 GB; resumes automatically at >= 27 GB free. Resumable: a clip whose file
exists with the inventory size is not downloaded again.

Output: results/meva_manifest.csv  (one row per PRIMARY clip).
"""
from __future__ import annotations
import hashlib
import shutil
import sys
import time
from pathlib import Path

import boto3
import cv2
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from paths import MEVA_DIR

REPO = Path(__file__).resolve().parents[1]
DATA = MEVA_DIR
CLIPS = DATA / "clips"
TEST3 = DATA / "test3"
BUCKET = "mevadata-public-01"
MIN_FREE_GB = 20.0             # video download only (decision 27-09)
RESUME_FREE_GB = 27.0
MANIFEST = REPO / "results" / "meva_manifest.csv"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def header(p: Path):
    cap = cv2.VideoCapture(str(p))
    ok = cap.isOpened()
    info = dict(opens=ok, fps=cap.get(cv2.CAP_PROP_FPS) if ok else None,
                n_frames_header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else None,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else None,
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else None)
    cap.release()
    return info


def main():
    sel = pd.read_csv(REPO / "results" / "meva_selection.csv")
    sel = sel[sel.tier == "PRIMARY"].reset_index(drop=True)
    assert len(sel) == 250, len(sel)
    # download order only (not a selection): detector subset S8 (docs/MEVA_detector_plan.md) first
    s8_path = REPO / "results" / "meva_S8_blocks.csv"
    if s8_path.exists():
        s8 = set(pd.read_csv(s8_path).block_id)
        sel = pd.concat([sel[sel.block_id.isin(s8)], sel[~sel.block_id.isin(s8)]]).reset_index(drop=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10}))
    done = {}
    if MANIFEST.exists():
        m = pd.read_csv(MANIFEST)
        done = {r.key: r._asdict() for r in m.itertuples(index=False)}
    rows = []
    t0 = time.time()
    rank = {k: i for i, k in enumerate(sel.key)}           # S8 (118 clips) first, then the rest

    def save(rs):
        pd.DataFrame(rs).sort_values("key", key=lambda c: c.map(rank)).to_csv(MANIFEST, index=False)
    for i, r in sel.iterrows():
        dst = CLIPS / Path(r.key).name
        if r.key in done and dst.exists() and dst.stat().st_size == r.size_bytes and done[r.key].get("sha256"):
            rows.append(done[r.key])
            continue
        free_gb = shutil.disk_usage(DATA).free / 1e9
        if free_gb - r.size_bytes / 1e9 < MIN_FREE_GB:
            # decision 27-09: pause, resume automatically once D: has >= RESUME_FREE_GB free
            print(f"PAUSE: free {free_gb:.1f} GB < {MIN_FREE_GB} GB after next clip; "
                  f"waiting for >= {RESUME_FREE_GB} GB", flush=True)
            while shutil.disk_usage(DATA).free / 1e9 < RESUME_FREE_GB:
                time.sleep(600)
            free_gb = shutil.disk_usage(DATA).free / 1e9
            print(f"RESUME: free {free_gb:.1f} GB", flush=True)
        if not (dst.exists() and dst.stat().st_size == r.size_bytes):
            t3 = TEST3 / dst.name
            if t3.exists() and t3.stat().st_size == r.size_bytes:
                shutil.copy2(t3, dst)
            else:
                tmp = dst.with_suffix(".part")
                for attempt in range(3):
                    try:
                        s3.download_file(BUCKET, r.key, str(tmp))
                        break
                    except Exception as e:  # network hiccup
                        print(f"retry {attempt} {r.key}: {e}", flush=True)
                        time.sleep(10)
                tmp.replace(dst)
        size = dst.stat().st_size
        row = dict(cam_tag=r.cam_tag, camera=r.camera, block_id=r.block_id, date=r.date,
                   start=r.start, end=r.end, dur_s=r.dur_s, key=r.key, path=str(dst),
                   size_bytes=size, size_ok=size == r.size_bytes, sha256=sha256(dst), **header(dst))
        rows.append(row)
        save(rows)
        el = time.time() - t0
        print(f"[{i+1}/250] {dst.name} {size/1e6:.0f} MB ok={row['size_ok']} opens={row['opens']} "
              f"hdr={row['n_frames_header']} elapsed={el/60:.1f} min free={free_gb:.1f} GB", flush=True)
    save(rows)
    m = pd.DataFrame(rows)
    print(f"DONE {len(m)} clips, {m.size_bytes.sum()/1e9:.2f} GB, size_ok={m.size_ok.all()}, "
          f"opens={m.opens.all()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
