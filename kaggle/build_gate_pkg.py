"""Build the small Kaggle dataset for kaggle/s8_detect.ipynb (gate B0.1 + S8 run inputs), < 200 MB.

Contents (written to <DATA_ROOT>/_derived\\09\\kaggle_s8_pkg, outside the Drive-synced area):
  gate/<video>.npz        the exact 600 decoded frames fed to YOLO in B0.1 (scm frame iterators, same as
                          code/weights_equiv.py): frames uint8 [N,H,W,3] BGR + idx int32 (lossless, so the
                          Kaggle side needs no JPEG/BMP decoder that could differ)
  gate/<video>.ref.json   reference detections (detections_cpu, the committed benchmark detections)
  s8_manifest.csv         118 S8 clips in detector queue order: key (S3), size_bytes, sha256, block_id
  yolo26s-seg.pt          backup weights (SHA 3da1d83e...)
  MEVA_detector_plan.md (+ addendum) and their .sha256
  SHA256SUMS              every file above
Usage: python kaggle/build_gate_pkg.py
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
from paths import DATA_ROOT  # noqa: E402
from weights_equiv import DETROOT, CFG, VIDEOS, WEIGHTS, WEIGHTS_SHA, dp  # noqa: E402

OUT = DATA_ROOT / "_derived" / "09" / "kaggle_s8_pkg"
PLAN_SHA = "1101322c0d35c8b653c902045b0f94b49a1816d26295c139be42e7850a3bd4be"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def s8_queue() -> pd.DataFrame:
    """Same order as code/meva_detect.queue(extend=False): S8 blocks in file order, clips by date/start."""
    sel = pd.read_csv(REPO / "results" / "meva_selection.csv")
    sel = sel[sel.tier == "PRIMARY"]
    man = pd.read_csv(REPO / "results" / "meva_manifest.csv").set_index("key")
    rows = []
    for b in pd.read_csv(REPO / "results" / "meva_S8_blocks.csv").block_id:
        for k in sel[sel.block_id == b].sort_values(["date", "start"]).key:
            rows.append(dict(key=k, block_id=b, size_bytes=int(man.loc[k, "size_bytes"]),
                             sha256=man.loc[k, "sha256"]))
    return pd.DataFrame(rows)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "gate").mkdir(parents=True)
    recs = {r[0]: r for r in dp.collect_videos(CFG, ["CDnet2014", "LASIESTA"], None)}
    for ds, vid in VIDEOS:
        vid_id, vpath, fdir, roi_s, roi_e = recs[vid]
        idx, frames = zip(*dp.make_frame_iter(vpath, fdir, roi_s, roi_e))
        name = vid.replace("/", "__")
        np.savez_compressed(OUT / "gate" / f"{name}.npz", frames=np.stack(frames), idx=np.array(idx, np.int32))
        shutil.copy2(DETROOT / ds / dp.video_json_name(vid_id, ds), OUT / "gate" / f"{name}.ref.json")
        print(vid, len(idx), frames[0].shape)
    q = s8_queue()
    assert len(q) == 118 and q.key.is_unique, len(q)
    q.to_csv(OUT / "s8_manifest.csv", index=False)
    assert sha256(WEIGHTS) == WEIGHTS_SHA
    shutil.copy2(WEIGHTS, OUT / "yolo26s-seg.pt")
    for f in ["MEVA_detector_plan.md", "MEVA_detector_plan.md.sha256",
              "MEVA_detector_plan_addendum.md", "MEVA_detector_plan_addendum.md.sha256"]:
        shutil.copy2(REPO / "docs" / f, OUT / f)
    assert sha256(OUT / "MEVA_detector_plan.md") == PLAN_SHA
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    (OUT / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}\n" for p in files))
    tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {len(files) + 1} files, {tot / 1e6:.1f} MB")
    assert tot < 200e6


if __name__ == "__main__":
    main()
