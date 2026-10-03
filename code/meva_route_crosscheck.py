"""Cross-route check on MEVA: laptop ov_gpu_fp32x JSONs vs Kaggle cuda_fp32_kaggle JSONs, same stems.

Supplementary evidence for docs/DETECTOR_PROVENANCE.md (not a gate; gate B0.1 is prereg'd separately).
Box matching = weights_equiv.match (greedy, same class, L_inf bbox distance). Per clip: boxes, unmatched
per side (+ max conf), max |dbbox|, max |dconf|, frames with g(a)/g(b) equal, person pairs straddling 0.7,
events under (a), (b), (b') (gap merge k_merge(fps)) and events with identical bounds.
Usage: python code/meva_route_crosscheck.py <laptop_dir> <kaggle_dir>
Writes results/meva_route_crosscheck.csv and .json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strata import merge_gaps, k_merge

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
STRICT_PX = 2          # 'strict' counts: a greedy pair with L_inf > 2 px is two unmatched boxes


def match_frame(ref, new):
    """weights_equiv.match, extended to return the pairs and unmatched indices."""
    used, pairs = set(), []
    for i, b in enumerate(ref):
        best, bi = None, None
        for j, c in enumerate(new):
            if j in used or c[5] != b[5]:
                continue
            dist = max(abs(c[k] - b[k]) for k in range(4))
            if best is None or dist < best:
                best, bi = dist, j
        if bi is not None:
            used.add(bi)
            pairs.append((i, bi, best))
    um_ref = [i for i in range(len(ref)) if i not in {p[0] for p in pairs}]
    um_new = [j for j in range(len(new)) if j not in used]
    return pairs, um_ref, um_new


def ev_bounds(g):
    g = np.r_[0, np.asarray(g, np.int8), 0]
    d = np.diff(g)
    return set(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def compare(p_ref: Path, p_new: Path) -> dict:
    fr, fn = json.load(open(p_ref))["frames"], json.load(open(p_new))["frames"]
    assert len(fr) == len(fn), (p_ref.name, len(fr), len(fn))
    r = dict(stem=p_ref.stem, n_frames=len(fr), boxes_ref=0, boxes_new=0, n_pairs=0, unm_ref=0, unm_new=0,
             max_unm_conf_ref=0.0, max_unm_conf_new=0.0, unm_conf_ge06=0, max_dpx=0.0, max_dconf=0.0,
             frames_all_matched=0, person_straddle_07=0, strict_unm_ref=0, strict_unm_new=0,
             strict_max_unm_conf=0.0, strict_unm_conf_ge06=0, strict_max_dpx=0.0, strict_max_dconf=0.0,
             strict_person_straddle_07=0)
    dconf, dpx = [], []
    g = {k: ([], []) for k in "ab"}
    for a, b in zip(fr, fn):
        A, B = a["boxes"], b["boxes"]
        pairs, ur, un = match_frame(A, B)
        r["boxes_ref"] += len(A); r["boxes_new"] += len(B); r["n_pairs"] += len(pairs)
        r["unm_ref"] += len(ur); r["unm_new"] += len(un)
        r["frames_all_matched"] += int(not ur and not un)
        for i in ur:
            r["max_unm_conf_ref"] = max(r["max_unm_conf_ref"], A[i][4]); r["unm_conf_ge06"] += A[i][4] >= 0.6
            r["strict_unm_ref"] += 1; r["strict_max_unm_conf"] = max(r["strict_max_unm_conf"], A[i][4])
            r["strict_unm_conf_ge06"] += A[i][4] >= 0.6
        for j in un:
            r["max_unm_conf_new"] = max(r["max_unm_conf_new"], B[j][4]); r["unm_conf_ge06"] += B[j][4] >= 0.6
            r["strict_unm_new"] += 1; r["strict_max_unm_conf"] = max(r["strict_max_unm_conf"], B[j][4])
            r["strict_unm_conf_ge06"] += B[j][4] >= 0.6
        for i, j, d in pairs:
            dpx.append(d); dconf.append(abs(A[i][4] - B[j][4]))
            if d > STRICT_PX:                   # greedy cross-pair of two one-sided boxes
                r["strict_unm_ref"] += 1; r["strict_unm_new"] += 1
                r["strict_max_unm_conf"] = max(r["strict_max_unm_conf"], A[i][4], B[j][4])
                r["strict_unm_conf_ge06"] += (A[i][4] >= 0.6) + (B[j][4] >= 0.6)
            else:
                r["strict_max_dpx"] = max(r["strict_max_dpx"], d)
                r["strict_max_dconf"] = max(r["strict_max_dconf"], abs(A[i][4] - B[j][4]))
                if A[i][5] == 0 and (A[i][4] >= 0.7) != (B[j][4] >= 0.7):
                    r["strict_person_straddle_07"] += 1
            if A[i][5] == 0 and (A[i][4] >= 0.7) != (B[j][4] >= 0.7):
                r["person_straddle_07"] += 1
        for side, X in ((0, A), (1, B)):
            g["a"][side].append(len(X) > 0)
            g["b"][side].append(any(x[5] == 0 and x[4] >= 0.7 for x in X))
    if dpx:
        r.update(max_dpx=float(max(dpx)), p99_dpx=float(np.percentile(dpx, 99)), mean_dpx=float(np.mean(dpx)),
                 max_dconf=float(max(dconf)), p99_dconf=float(np.percentile(dconf, 99)),
                 mean_dconf=float(np.mean(dconf)))
    k = k_merge(FPS)
    for d in ("a", "b", "bprime"):
        src = "b" if d == "bprime" else d
        g0, g1 = (np.array(x, np.int8) for x in g[src])
        if d == "bprime":
            g0, g1 = merge_gaps(g0, k), merge_gaps(g1, k)
        e0, e1 = ev_bounds(g0), ev_bounds(g1)
        r[f"g_{d}_equal_frames"] = int((g0 == g1).sum())
        r[f"ev_{d}_ref"], r[f"ev_{d}_new"], r[f"ev_{d}_same_bounds"] = len(e0), len(e1), len(e0 & e1)
    return r


def main():
    lap, kag = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [compare(p, kag / p.name) for p in sorted(lap.glob("*.json"))]
    df = pd.DataFrame(rows)
    tot = df.drop(columns="stem").sum(numeric_only=True)
    for c in ("max_dpx", "max_dconf", "max_unm_conf_ref", "max_unm_conf_new", "p99_dpx", "p99_dconf",
              "strict_max_unm_conf", "strict_max_dpx", "strict_max_dconf"):
        tot[c] = df[c].max()
    tot = tot.drop(["mean_dpx", "mean_dconf"])
    df.to_csv(REPO / "results" / "meva_route_crosscheck.csv", index=False, float_format="%.6g")
    out = dict(ref_route="ov_gpu_fp32x (laptop)", new_route="cuda_fp32_kaggle", k_merge=k_merge(FPS),
               matching="weights_equiv.match (greedy, same class, L_inf)",
               strict=f"greedy pairs with L_inf > {STRICT_PX} px counted as unmatched on both sides", total=tot.to_dict())
    (REPO / "results" / "meva_route_crosscheck.json").write_text(json.dumps(out, indent=1, default=float))
    print(df.T.to_string())
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
