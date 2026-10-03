"""Run YOLO26s-seg on MEVA clips per docs/MEVA_detector_plan.md (checkpoint/resume per clip).

Every frame (no stride), conf 0.25, imgsz 640, no class filter; output format identical to
detections_cpu: {"frames":[{"frame":i,"boxes":[[x1,y1,x2,y2,score,cls],...]}], "meta":{...}}.
Several workers (one per equivalent route) can run at once: a clip is claimed with an O_EXCL lock
file, results are written atomically. Clips not yet downloaded are skipped and retried.

Usage: python code/meva_detect.py --route ov_gpu_fp32x [--extend]
  queue = S8 clips (results/meva_S8_blocks.csv); with --extend, followed by the remaining PRIMARY
  blocks, new recording sessions first (plan addendum A1; sensitivity only).
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights_equiv import load_model, predict, CONF, IMGSZ, WEIGHTS_SHA  # noqa: E402
from paths import MEVA_DIR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = MEVA_DIR
DET = DATA / "detections"
MIN_FREE_GB = 5.0              # decision 27-09: 20 GB applies to video download only
# idle gate (decision 27-09: CPU of other processes <= 60 %, free RAM >= 6 GB), checked before every
# clip; by day only the Iris Xe f32 worker runs, at night (22:00-07:00 local) both workers run.
IDLE_OTHERS_CPU_MAX = 60.0     # % of total CPU used by processes other than detector workers
IDLE_RAM_MIN_GB = 6.0
NIGHT = (22, 7)                # [22:00, 07:00)
DAY_ROUTES = ("ov_gpu_fp32x",)


def is_night() -> bool:
    h = time.localtime().tm_hour
    return h >= NIGHT[0] or h < NIGHT[1]


def machine_idle(own: str = "meva_detect.py") -> tuple[bool, str]:
    import psutil
    mine = {p.pid for p in psutil.process_iter(["cmdline"])
            if p.info["cmdline"] and any(own in c for c in p.info["cmdline"])}
    procs = [p for p in psutil.process_iter() if p.pid not in mine and p.pid != 0]
    for p in procs:
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    time.sleep(10)
    others = 0.0
    for p in procs:
        try:
            others += p.cpu_percent(None)
        except Exception:
            pass
    others /= psutil.cpu_count()
    ram = psutil.virtual_memory().available / 1e9
    ok = others <= IDLE_OTHERS_CPU_MAX and ram >= IDLE_RAM_MIN_GB
    return ok, f"others_cpu={others:.0f}% ram_free={ram:.1f}GB"


def queue(extend: bool) -> list[str]:
    sel = pd.read_csv(REPO / "results" / "meva_selection.csv")
    sel = sel[sel.tier == "PRIMARY"]
    blocks = pd.read_csv(REPO / "results" / "meva_selection_blocks.csv")
    blocks = blocks[blocks.tier == "PRIMARY"]
    s8 = pd.read_csv(REPO / "results" / "meva_S8_blocks.csv")
    order = list(s8.block_id)
    if extend:
        # plan addendum A1 (docs/MEVA_detector_plan_addendum.md): recording sessions NOT in S8 first,
        # sessions with more cameras first, then chronological; then remaining blocks of S8 sessions.
        from meva_profile import sessions
        rest = blocks[~blocks.block_id.isin(order)].copy()
        rest["session"] = rest.block_id.map(sessions(blocks))
        s8_sess = {sessions(blocks)[b] for b in order}
        ncam = blocks.assign(session=blocks.block_id.map(sessions(blocks))).groupby("session").cam_tag.nunique()
        rest["old"] = rest.session.isin(s8_sess)
        rest["ncam"] = rest.session.map(ncam)
        rest = rest.sort_values(["old", "ncam", "session", "cam_tag"], ascending=[True, False, True, True])
        order += list(rest.block_id)
    keys = []
    for b in order:
        keys += list(sel[sel.block_id == b].sort_values(["date", "start"]).key)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=["torch_cpu", "ov_cpu_fp32", "ov_gpu_fp32x"])
    ap.add_argument("--extend", action="store_true")
    a = ap.parse_args()
    DET.mkdir(parents=True, exist_ok=True)
    model, dev = load_model(a.route)
    keys = queue(a.extend)
    pending = list(keys)
    for lk in DET.glob("*.lock"):            # stale locks of this route (crash/restart) are released
        if lk.read_text() == a.route:
            lk.unlink()
    while pending:
        progressed = False
        try:
            man = pd.read_csv(REPO / "results" / "meva_manifest.csv")
        except Exception:                    # manifest being rewritten by the downloader
            time.sleep(5)
            continue
        ready = set(man.loc[man.size_ok & man.opens, "key"])
        for key in list(pending):
            stem = Path(key).stem
            out = DET / f"{stem}.json"
            if out.exists():
                pending.remove(key)
                continue
            if key not in ready:
                continue
            if a.route not in DAY_ROUTES and not is_night():
                print("day: only the Iris Xe worker runs; waiting for 22:00", flush=True)
                time.sleep(600)
                break
            ok, why = machine_idle()
            if not ok:
                print(f"busy ({why}); waiting", flush=True)
                time.sleep(300)
                break                        # re-read manifest, retry
            lock = DET / f"{stem}.lock"
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, a.route.encode())
                os.close(fd)
            except FileExistsError:
                pending.remove(key)          # another worker owns it
                continue
            if shutil.disk_usage(DATA).free / 1e9 < MIN_FREE_GB:
                print(f"STOP: free space < {MIN_FREE_GB} GB", flush=True)
                lock.unlink()
                return
            t0 = time.time()
            cap = cv2.VideoCapture(str(DATA / "clips" / Path(key).name))
            frames, i = [], 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append({"frame": i, "boxes": predict(model, dev, f)})
                i += 1
            cap.release()
            meta = dict(key=key, route=a.route, weights_sha256=WEIGHTS_SHA, conf=CONF, imgsz=IMGSZ,
                        n_frames=i, sec=round(time.time() - t0, 1))
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")))
            tmp.replace(out)
            lock.unlink()
            pending.remove(key)
            progressed = True
            print(f"{a.route} {stem} {i} fr {i / (time.time() - t0):.2f} fps "
                  f"done={len(keys) - len(pending)}/{len(keys)}", flush=True)
        if pending and not progressed:
            time.sleep(60)                   # waiting for downloads
    print("QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
