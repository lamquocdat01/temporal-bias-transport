"""Q-B evidence for STRATA_SPEC v2: how many short (1-3 frame) definition-(b) events on VIRAT-P have
no ground-truth person? And what does the gap-merge rule (b') do to the event counts?

GT: official VIRAT Ground 2.0 annotations (<video>.viratdata.objects.txt, data.kitware.com,
columns: object_id, duration, frame, x, y, w, h, type; type 1 = person). Stored outside the repo at
<DATA_ROOT>/MEVA_20h\\virat_gt\\ (SHA256SUMS there). Detections: detections_cpu_virat_full.
Frame alignment between GT and VIRAT-P frame numbers is checked empirically (offset in -5..5
maximising IoU >= 0.3 matches) is reported as a sensitivity;
the main count uses offset 0 (GT frames are 0-based, like the decoded frame index).

An event (maximal run of g_b = 1) is
  temporally unsupported: no GT person box in any of its frames;
  spatially unsupported: no (person, conf >= 0.7) detection box in any of its frames matches a GT
                         person box with IoU >= 0.3.
Caveat: VIRAT annotates moving objects; a static person may be unannotated.
Output: results/viratp_flicker.json, results/viratp_flicker_by_video.csv
"""
from __future__ import annotations
import paths
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e1_profile import DET, load_det
from strata import events, merge_gaps
from paths import MEVA_DIR

REPO = Path(__file__).resolve().parents[1]
GT = MEVA_DIR / "virat_gt"
PROC = paths.BENCH_PROC
IOU_T, CONF = 0.3, 0.7
K_MERGE = round(0.2 * 30)          # (b'): gaps <= 0.2 s at 30 fps -> 6 frames


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


SKIPPED: dict = {}


def load_gt(v):
    p = GT / f"{v}.viratdata.objects.txt"
    if not p.exists() or p.stat().st_size < 100:
        return None
    rows = [ln.split() for ln in open(p)]
    SKIPPED[v] = sum(len(r) != 8 for r in rows)          # malformed rows in the official file
    a = np.array([r for r in rows if len(r) == 8], float)
    a = a[a[:, 7] == 1]
    gt = {}
    for _, _, f, x, y, w, h, _ in a:
        gt.setdefault(int(f), []).append((x, y, x + w, y + h))
    return gt


def main():
    vp = pd.read_parquet(PROC / "frame_scores_viratp.parquet")
    rows, per_ev = [], []
    for v, x in vp.groupby("video", sort=True):
        gt = load_gt(v)
        if gt is None:
            rows.append(dict(video=v, gt_available=False))
            continue
        x = x.sort_values("frame")
        fr = x.frame.values
        g = x.g_t.values.astype(np.int8)
        det = load_det(next((DET / "detections_cpu_virat_full" / "VIRAT").glob(f"*__{v}.json")))
        pb = {int(f): [b for b in det.get(int(f), []) if int(b[5]) == 0 and b[4] >= CONF] for f in fr}
        # empirical frame offset between GT numbering and VIRAT-P frame numbers
        best = None
        for off in range(-5, 6):
            m = sum(any(iou(b[:4], q) >= IOU_T for q in gt.get(f + off, [])) for f in fr for b in pb[f])
            best = (m, off) if best is None or m > best[0] else best
        off = best[1]
        s, L, cens = events(g)
        for a, l, c in zip(s, L, cens):
            fs = fr[a:a + l]
            rec = dict(video=v, start=int(fr[a]), length=int(l), censored=bool(c))
            # main: offset 0 (GT frames are 0-based like the decoded index); sensitivity: fitted offset
            for tag, o in (("", 0), ("_fitted", off)):
                rec["temporally_supported" + tag] = any(len(gt.get(int(f) + o, [])) > 0 for f in fs)
                rec["spatially_supported" + tag] = any(iou(b[:4], q) >= IOU_T for f in fs for b in pb[int(f)]
                                                       for q in gt.get(int(f) + o, []))
            per_ev.append(rec)
        g2 = merge_gaps(g, K_MERGE)
        rows.append(dict(video=v, gt_available=True, frame_offset=off, matched_boxes_at_offset=best[0],
                         n_frames=len(g), n_events_b=len(L), n_events_bprime=len(events(g2)[1]),
                         duty_b=float(g.mean()), duty_bprime=float(g2.mean()),
                         median_len_b=float(np.median(L)) if len(L) else np.nan,
                         median_len_bprime=float(np.median(events(g2)[1])) if g2.any() else np.nan,
                         gt_person_frame_frac=float(np.mean([len(gt.get(int(f) + off, [])) > 0 for f in fr]))))
    bv = pd.DataFrame(rows)
    ev = pd.DataFrame(per_ev)
    bv.to_csv(REPO / "results" / "viratp_flicker_by_video.csv", index=False, float_format="%.4g")

    def frac(d, col):
        return dict(n=int(len(d)), frac_unsupported=float(1 - d[col].mean()) if len(d) else None)
    short, longer = ev[ev.length <= 3], ev[ev.length > 3]
    out = dict(k_merge_frames=K_MERGE, iou_threshold=IOU_T, videos_with_gt=int(bv.gt_available.sum()),
               videos_without_gt=bv.loc[~bv.gt_available, "video"].tolist(),
               events_total=int(len(ev)), events_len_1_3=int(len(short)),
               share_of_events_len_1_3=float(len(short) / len(ev)),
               short_temporal=frac(short, "temporally_supported"), short_spatial=frac(short, "spatially_supported"),
               long_temporal=frac(longer, "temporally_supported"), long_spatial=frac(longer, "spatially_supported"),
               sensitivity_fitted_offset=dict(
                   short_temporal=frac(short, "temporally_supported_fitted"),
                   short_spatial=frac(short, "spatially_supported_fitted"),
                   long_temporal=frac(longer, "temporally_supported_fitted"),
                   long_spatial=frac(longer, "spatially_supported_fitted")),
               events_b=int(bv.n_events_b.sum()), events_bprime=int(bv.n_events_bprime.sum()),
               malformed_rows_skipped=SKIPPED,
               frame_offsets=bv.dropna(subset=["frame_offset"]).set_index("video").frame_offset.astype(int).to_dict())
    (REPO / "results" / "viratp_flicker.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(bv.to_string())


if __name__ == "__main__":
    main()
