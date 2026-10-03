"""Assemble the MEVA S8 frame streams per SEGMENT (STRATA_SPEC v2 §1, §5) -- profile inputs only.

For every S8 block (results/meva_S8_blocks.csv), clips are concatenated in time order and the block is
split into segments at gaps > 2 s (same rule as code/meva_profile.py). Per segment:
  g_b   person conf >= 0.7 (definition (b)) from the Kaggle S8 detections
  g_bp  (b') = g_b with idle gaps <= round(0.2 s x fps) merged inside the segment
  g_a   has-box (definition (a)), descriptive only
  mad, fpx  raw frame-difference cue; at a junction between two continuous clips the cue of the first
            frame of the later clip is computed from the stored thumbnails (last of previous clip, first
            of next clip); the first frame of a segment has no cue (NaN).
No policy is run here. The window table of every segment is asserted equal to cache/meva_windows.parquet
(definition (b'), in_S8), i.e. to the profile frozen in Dot 2.

Cache: cache/meva_s8_frames.pkl (derived, ~20 MB, gitignored). Library: load_segments(det_dir=None).
"""
from __future__ import annotations
import variant as V
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cue import diff_stats
from meva_profile import tsec, g_det, sessions, GAP_S
from paths import MEVA_DIR
from strata import merge_gaps, k_merge, window_table

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
CACHE = REPO / "cache" / "meva_s8_frames.pkl"


def _g_det_dir(stem: str, n: int, det_dir: Path):
    import json
    fr = json.load(open(det_dir / f"{stem}.json"))["frames"]
    assert len(fr) == n, (stem, len(fr), n)
    ga = np.array([len(f["boxes"]) > 0 for f in fr], np.int8)
    gb = np.array([any(b[5] == 0 and b[4] >= 0.7 for b in f["boxes"]) for f in fr], np.int8)
    return ga, gb


def build(det_dir: Path | None = None, stems: set | None = None) -> list[dict]:
    sel = pd.read_csv(RES / "meva_selection.csv")
    sel = sel[sel.tier == "PRIMARY"]
    blocks = pd.read_csv(RES / "meva_selection_blocks.csv")
    blocks = blocks[blocks.tier == "PRIMARY"].copy()
    blocks["session"] = blocks.block_id.map(sessions(blocks))
    s8 = set(pd.read_csv(RES / "meva_S8_blocks.csv").block_id)
    man = pd.read_csv(RES / "meva_manifest.csv").set_index("key")
    segs = []
    for blk in blocks[blocks.block_id.isin(s8)].itertuples(index=False):
        clips = sel[sel.block_id == blk.block_id].sort_values(["date", "start"])
        if stems is not None and not any(Path(k).stem in stems for k in clips.key):
            continue
        fps = float(man.loc[clips.key.iloc[0], "fps"])
        cur, prev = None, None
        for c in clips.itertuples(index=False):
            stem = Path(c.key).stem
            z = np.load(MEVA_DIR / "cue" / f"{stem}.npz")
            n = int(z["n_decoded"])
            if det_dir is None:
                ga, gb = g_det(stem, n)
            else:
                ga, gb = _g_det_dir(stem, n, det_dir)
            assert gb is not None, f"no detections for S8 clip {stem}"
            mad, fpx = z["mad"].astype(float), z["fpx"].astype(float)
            assert np.isnan(mad[0]) and len(mad) == n
            new_seg = prev is None or tsec(c.start) - prev["end"] > GAP_S
            if new_seg:
                if cur is not None:
                    segs.append(cur)
                cur = dict(block_id=blk.block_id, cam_tag=blk.cam_tag, session=blk.session, fps=fps,
                           seg=0 if prev is None else cur["seg"] + 1, stems=[], parts=[])
            else:                                    # continuous junction: cue across clips from thumbnails
                mad = mad.copy(); fpx = fpx.copy()
                mad[0], fpx[0] = diff_stats(prev["last"], z["first"])
            cur["stems"].append(stem)
            cur["parts"].append((ga, gb, mad, fpx))
            prev = dict(end=tsec(c.end), last=z["last"])
        segs.append(cur)
    out = []
    for s in segs:
        parts = s.pop("parts")
        ga, gb, mad, fpx = (np.concatenate([p[i] for p in parts]) for i in range(4))
        s.update(n=len(gb), g_a=ga, g_b=gb, g_bp=merge_gaps(gb, k_merge(s["fps"])), mad=mad, fpx=fpx)
        out.append(s)
    return out


def check_against_profile(segs: list[dict]) -> int:
    """Window keys of every segment must equal the Dot 2 profile (cache/meva_windows.parquet)."""
    w = pd.read_parquet(V.CACHE / "meva_windows.parquet")
    w = w[(w.definition == "bprime") & w.in_S8]
    nwin = 0
    for s in segs:
        ref = w[(w.block_id == s["block_id"]) & (w.seg == s["seg"])].sort_values("w_start")
        wt = window_table(s["g_bp"])
        assert len(ref) == len(wt) and (ref.key.values == wt.key.values).all() and \
            (ref.n.values == wt.n.values).all(), (s["block_id"], s["seg"])
        nwin += len(wt)
    assert nwin == len(w), (nwin, len(w))
    return nwin


def load_segments() -> list[dict]:
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    segs = build()
    check_against_profile(segs)
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(segs, open(CACHE, "wb"))
    return segs


if __name__ == "__main__":
    segs = load_segments()
    n = check_against_profile(segs)
    print(f"{len(segs)} segments, {sum(s['n'] for s in segs)} frames, {n} windows == profile")
