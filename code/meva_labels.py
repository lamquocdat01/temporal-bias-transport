"""Definition (c) on MEVA: g_c[t] = 1 iff frame t lies in the timespan of >= 1 annotated activity.

Source: Kitware annotations in the meva-data-repo clone (annotation/DIVA-phase-2/MEVA/):
  kitware/<date>/<hh>/<clip>.activities.yml if present, else kitware-meva-training/...;
  clips with neither -> (c) undefined (reported, excluded from (c) statistics).
Timespans 'tsr0: [a, b]' are inclusive frame indices. All activity types are used.
Output: <DATA_ROOT>/MEVA_20h\\labels\\<stem>.npz (g_c int8, n_acts, src) and
results/meva_labels_manifest.csv. n_frames is taken from the cue pass (decoded count) when available,
else from the video header.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from meva_profile import clip_frames
from paths import MEVA_DIR, MEVA_REPO

REPO = Path(__file__).resolve().parents[1]
DATA = MEVA_DIR
ANN = MEVA_REPO / "annotation" / "DIVA-phase-2" / "MEVA"
OUT = DATA / "labels"
Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def find(src: str, clip: str, date: str, hh: str):
    p = ANN / src / date / hh / f"{clip}.activities.yml"
    return p if p.exists() else None


def main():
    OUT.mkdir(exist_ok=True)
    sel = pd.read_csv(REPO / "results" / "meva_selection.csv")
    sel = sel[sel.tier == "PRIMARY"]
    man = pd.read_csv(REPO / "results" / "meva_manifest.csv").set_index("key")
    cue_dir = DATA / "cue"
    rows = []
    for r in sel.itertuples(index=False):
        stem = Path(r.key).stem                                  # ...G639.r13
        clip = stem.replace(".r13", "")
        date, hh = r.key.split("/")[1], r.key.split("/")[2]
        if r.key in man.index:                        # same frame-count source as meva_profile.clip_frames
            n = clip_frames(stem, r.key, man)
            nsrc = ("decoded" if (cue_dir / f"{stem}.npz").exists() else
                    "detector" if (DATA / "detections" / f"{stem}.json").exists() else "header")
        else:
            n = int(round(r.dur_s * 30))
            nsrc = "nominal"
        src, path = "", None
        for s in ("kitware", "kitware-meva-training"):
            path = find(s, clip, date, hh)
            if path:
                src = s
                break
        g = np.zeros(n, np.int8)
        n_acts, n_out = 0, 0
        if path:
            docs = yaml.load(open(path, encoding="utf-8"), Loader=Loader) or []
            for d in docs:
                act = d.get("act") if isinstance(d, dict) else None
                if not act:
                    continue
                for ts in act.get("timespan", []):
                    a, b = ts["tsr0"]
                    n_acts += 1
                    if b >= n:
                        n_out += 1
                    g[max(0, int(a)):min(n, int(b) + 1)] = 1
        np.savez_compressed(OUT / f"{stem}.npz", g_c=g, n_acts=n_acts, src=src)
        rows.append(dict(key=r.key, cam_tag=r.cam_tag, block_id=r.block_id, n_frames=n, n_frames_src=nsrc,
                         label_src=src, n_acts=n_acts, n_spans_past_end=n_out,
                         duty_c=float(g.mean()) if src else np.nan))
    m = pd.DataFrame(rows)
    m.to_csv(REPO / "results" / "meva_labels_manifest.csv", index=False)
    print(m.groupby("cam_tag").agg(clips=("key", "size"), labeled=("label_src", lambda s: (s != "").sum()),
                                   acts=("n_acts", "sum")).to_string())
    print("frames source:", m.n_frames_src.value_counts().to_dict())


if __name__ == "__main__":
    main()
