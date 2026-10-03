"""Cue d_t on every frame of the 250 PRIMARY MEVA clips (docs/STRATA_SPEC.md §5) + R7 decode check.

Per clip -> <DATA_ROOT>/MEVA_20h\\cue\\<stem>.npz with
  mad, fpx (float32, frame 0 = NaN), first/last thumbnail (for cross-clip junctions in a block),
  n_decoded, fps, width, height.
Resumable; waits for clips still downloading. Writes results/meva_cue_manifest.csv with the
R7-E1 decoded-frame check (decoded >= 95 % of dur_s x fps).
This is profile-side computation only (no policy is run on MEVA).

Usage: python code/meva_cue.py [--available-only] [--wait-s8]
  --available-only  process the clips already downloaded, then write the manifest and exit
  --wait-s8         old rule of 27-09 (start only after the S8 detector pass); off since 28-09 (route K:
                    S8 detector runs on Kaggle, cue runs on the laptop now)
Before every clip: idle gate (other processes' CPU <= 60 %, free RAM >= 6 GB), else wait 5 min.
At start: if the previous run's log (logs/meva_cue_*.log) has no clean end and its .err is empty/missing,
print "resume after unexpected stop" (decision 29-09).
"""
from __future__ import annotations
import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cue import thumb, diff_stats
from meva_detect import machine_idle
from paths import MEVA_DIR

REPO = Path(__file__).resolve().parents[1]
DATA = MEVA_DIR
OUT = DATA / "cue"
MIN_FREE_GB = 5.0              # decision 27-09 (video download keeps 20 GB)


def wait_for_s8_detector():
    """Decision 27-09: the cue pass runs after the S8 detector pass."""
    s8 = set(pd.read_csv(REPO / "results" / "meva_S8_blocks.csv").block_id)
    sel = pd.read_csv(REPO / "results" / "meva_selection.csv")
    stems = [Path(k).stem for k in sel[(sel.tier == "PRIMARY") & sel.block_id.isin(s8)].key]
    while True:
        done = sum((DATA / "detections" / f"{st}.json").exists() for st in stems)
        if done == len(stems):
            print(f"S8 detector complete ({done} clips); starting cue", flush=True)
            return
        print(f"waiting for S8 detector: {done}/{len(stems)}", flush=True)
        time.sleep(1800)


def note_previous_run():
    """Log how the previous run ended. The current run's log is still empty at this point, so it is skipped."""
    logs = sorted((p for p in (REPO / "logs").glob("meva_cue_*.log") if p.stat().st_size > 0),
                  key=lambda p: p.stat().st_mtime)
    if not logs:
        return
    prev = logs[-1]
    lines = [l for l in prev.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    if last.startswith(("R7 decode check failures", "STOP:")):
        return                                   # previous run ended cleanly
    err = prev.with_suffix(".err")
    done = sum(1 for q in OUT.glob("*.npz") if not q.name.endswith(".tmp.npz"))
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(prev.stat().st_mtime))
    if not err.exists() or err.stat().st_size == 0:
        print(f"resume after unexpected stop: previous log {prev.name} last written {when}, last line '{last}', "
              f"{err.name} empty/missing; {done} clips already done", flush=True)
    else:
        print(f"resume after error stop: previous log {prev.name} last written {when}, see {err.name}; "
              f"{done} clips already done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--available-only", action="store_true")
    ap.add_argument("--wait-s8", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    note_previous_run()
    if a.wait_s8:
        wait_for_s8_detector()
    sel = pd.read_csv(REPO / "results" / "meva_selection.csv")
    sel = sel[sel.tier == "PRIMARY"]
    s8 = set(pd.read_csv(REPO / "results" / "meva_S8_blocks.csv").block_id)
    sel = pd.concat([sel[sel.block_id.isin(s8)], sel[~sel.block_id.isin(s8)]])
    pending = list(sel.itertuples(index=False))
    while pending:
        try:
            man = pd.read_csv(REPO / "results" / "meva_manifest.csv")
        except Exception:
            time.sleep(5)
            continue
        ready = set(man.loc[man.size_ok & man.opens, "key"])
        if a.available_only:
            pending = [r for r in pending if r.key in ready]
        progressed = False
        for r in list(pending):
            stem = Path(r.key).stem
            dst = OUT / f"{stem}.npz"
            if dst.exists():
                pending.remove(r)
                continue
            if r.key not in ready:
                continue
            if shutil.disk_usage(DATA).free / 1e9 < MIN_FREE_GB:
                print(f"STOP: free space < {MIN_FREE_GB} GB", flush=True)
                return
            ok, why = machine_idle(own="meva_cue.py")
            if not ok:
                print(f"busy ({why}); waiting", flush=True)
                time.sleep(300)
                break                        # re-read manifest, retry
            t0 = time.time()
            cap = cv2.VideoCapture(str(DATA / "clips" / Path(r.key).name))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            mad, fpx, prev, first = [], [], None, None
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                th = thumb(f)
                if prev is None:
                    first = th
                    mad.append(np.nan)
                    fpx.append(np.nan)
                else:
                    m, x = diff_stats(prev, th)
                    mad.append(m)
                    fpx.append(x)
                prev = th
            cap.release()
            tmp = dst.with_name(dst.stem + ".tmp.npz")
            np.savez_compressed(tmp, mad=np.array(mad, np.float32), fpx=np.array(fpx, np.float32),
                                first=first, last=prev, n_decoded=len(mad), fps=fps, width=w, height=h)
            tmp.replace(dst)
            pending.remove(r)
            progressed = True
            print(f"{stem}: {len(mad)} fr, {len(mad) / (time.time() - t0):.1f} fps, left {len(pending)}",
                  flush=True)
        if pending and not progressed:
            time.sleep(60)
    # manifest + R7-E1 decoded check
    rows = []
    for r in sel.itertuples(index=False):
        if not (OUT / f"{Path(r.key).stem}.npz").exists():
            continue
        z = np.load(OUT / f"{Path(r.key).stem}.npz")
        exp = r.dur_s * float(z["fps"])
        rows.append(dict(key=r.key, cam_tag=r.cam_tag, block_id=r.block_id, n_decoded=int(z["n_decoded"]),
                         fps=float(z["fps"]), width=int(z["width"]), height=int(z["height"]),
                         expected_frames=exp, decoded_ok=int(z["n_decoded"]) >= 0.95 * exp))
    m = pd.DataFrame(rows)
    m.to_csv(REPO / "results" / "meva_cue_manifest.csv", index=False)
    print("R7 decode check failures:", int((~m.decoded_ok).sum()), "of", len(m))


if __name__ == "__main__":
    main()
