"""Does the synthetic generator of code/synthetic_check.py satisfy A1 BY CONSTRUCTION?

A1: P_B(window numerator/denominator | h) = P_T(. | h). The generator shares only process-level laws
(on-durations, cue | g, policies); the idle law (medium idles, camera x session scale) and the
segment length differ between benchmark and target. This script measures A1 directly on the large
population samples (2,000 benchmark clips, 300 target 1-h segments; same seed/sequence as the
coverage study): per stratum h (v2 cells) and metric, the per-stratum ratio m_B(h) vs m_T(h), and
the within-stratum composition (mean in-event fraction, mean max event length) on each side.
Output: results/synthetic_a1_check.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import synthetic_check as SC
import strata as ST

REPO = Path(__file__).resolve().parents[1]


def main():
    rng = np.random.default_rng(SC.SEED)
    tau, _ = SC.pick_tau(rng)
    (pb, pt), truth = SC.population(rng, tau)
    bw = pb.assign(video_uid=np.arange(len(pb)) // 5)     # support counting only
    mp, cells = ST.build_strata_cells(bw.assign(video_uid=np.arange(len(bw))), "v2")
    out, worst = {}, {}
    for pol in SC.POL:
        for m in SC.METRICS:
            nc, dc = SC.cols(pol, m)
            rows = []
            for cid, members in cells.items():
                b = pb[pb.key.isin(members)]
                t = pt[pt.key.map(lambda k: mp.get(k, "U")) == cid]
                if len(t) < 30 or b[dc].sum() == 0 or t[dc].sum() == 0:
                    continue
                mb, mt = b[nc].sum() / b[dc].sum(), t[nc].sum() / t[dc].sum()
                rows.append(dict(h=cid, n_win_B=len(b), n_win_T=len(t), m_B=float(mb), m_T=float(mt),
                                 diff=float(mt - mb), wT=float(t.n.sum() / pt.n.sum()),
                                 infrac_B=float(b.E.sum() / b.n.sum()), infrac_T=float(t.E.sum() / t.n.sum())))
            out[f"{pol}_{m}"] = rows
            if rows:
                r = max(rows, key=lambda x: abs(x["diff"]) * x["wT"])
                worst[f"{pol}_{m}"] = dict(max_abs_diff=max(abs(x["diff"]) for x in rows),
                                           wT_weighted_abs_diff=sum(abs(x["diff"]) * x["wT"] for x in rows),
                                           largest_contrib=r["h"])
    res = dict(tau=tau, n_bench_windows=len(pb), n_target_windows=len(pt), summary=worst, per_stratum=out)
    (REPO / "results" / "synthetic_a1_check.json").write_text(json.dumps(res, indent=1))
    for k, v in worst.items():
        print(f"{k:9s} max|m_T-m_B|={v['max_abs_diff']:.3f}  sum wT|diff|={v['wT_weighted_abs_diff']:.3f}  ({v['largest_contrib']})")
    for r in out["P2_frec"]:
        print(f"  {r['h']:40s} wT={r['wT']:.3f} m_B={r['m_B']:.3f} m_T={r['m_T']:.3f} infrac B/T={r['infrac_B']:.3f}/{r['infrac_T']:.3f}")


if __name__ == "__main__":
    main()
