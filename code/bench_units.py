"""Benchmark side of transport (STRATA_SPEC v2): strata/cells, P4 fit, frozen operating points, window metrics.

Inputs: cache/bench_cue.parquet (code/cue.py), frame_scores*.parquet, detections_cpu.
Main benchmark = 121 videos (CDnet2014 53, LASIESTA 48, BMC 20); VIRAT-P (10) = sensitivity arm "121+VP".
Event definitions: "bprime" = (b') main (gap-merge <= round(0.2 s x fps)); "b" = raw (b), E6.
Cell rules: "v2" pooled (R -> D -> C) main; "v1" partition merge, E6.

Outputs
  results/operating_points.json        tau2, tau3, tau4 at pooled event recall (b') = 0.90; P1 K cells {2,5,10,30}
  results/benchmark_units.parquet      one row per (definition, video, window, policy, K)
  results/strata_map_benchmark.csv     definition x rule x arm x raw key -> cell (+ support counts)
  results/m_B_strata.csv               definition x rule x policy x K x cell: per-frame numerators/denominators, m_B(h)
  results/w_B_strata.csv               definition x rule x arm x cell: benchmark frame share w_B(h)
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cue import normalise, PROC
import variant as V
from e1_profile import load_det, g_from_det, DET, PERSON_CONF
from policies import p1, p2, p3, p4, p4_features, window_metrics, MCOLS, METRICS
from strata import window_table, build_strata_cells, merge_gaps, k_merge

SEED = 42
REPO = Path(__file__).resolve().parents[1]
RES = V.RES
TARGET_RECALL = 0.90
K_ALL = range(1, 31)
K_PREREG = (2, 5, 10, 30)
DEFS = ("bprime", "b")
RULES = ("v2", "v1")
LOWFR = {"port_0_17fps": 0.17, "tunnelExit_0_35fps": 0.35, "turnpike_0_5fps": 0.5, "tramCrossroad_1fps": 1.0}


def fps_of(ds, v):
    if ds == "BMC":
        return 25.0
    if ds == "VIRAT-P":
        return 30.0
    return LOWFR.get(v, 25.0)


def load_segments():
    cue = pd.read_parquet(V.CACHE / "bench_cue.parquet")
    fs = V.frame_scores()
    fs = fs[fs.dataset.isin(["CDnet2014", "LASIESTA", "BMC"])]
    ev = pd.read_csv(PROC / "events_durations.csv")
    key_of = {(d, v.split("__")[-1]): v for d, v in ev[["dataset", "video"]].drop_duplicates().itertuples(index=False)}
    vp = pd.read_parquet(PROC / "frame_scores_viratp.parquet")
    segs = []
    for (ds, v), c in cue.groupby(["dataset", "video"], sort=True):
        c = c.sort_values("frame")
        fr = c.frame.values
        if ds == "VIRAT-P":
            x = vp[vp.video == v].sort_values("frame")
            assert np.array_equal(x.frame.values, fr)
            g = x.g_t.values.astype(np.int8)
        else:
            x = fs[(fs.dataset == ds) & (fs.video == v)].sort_values("frame")
            assert np.array_equal(x.frame.values, fr), (ds, v)
            det = V.bench_det(ds, key_of[(ds, v)])
            g = g_from_det(det, fr, PERSON_CONF)
        fps = fps_of(ds, v)
        dn, fn = normalise(c.mad.values, c.fpx.values)
        segs.append(dict(dataset=ds, video=v, uid=f"{ds}/{v}", fps=fps, k=k_merge(fps), dn=dn, fn=fn,
                         g={"b": g, "bprime": merge_gaps(g, k_merge(fps))}, main=ds != "VIRAT-P"))
    assert sum(s["main"] for s in segs) == 121 and len(segs) == 131
    return segs


def pooled_recall(segs, run, d="bprime"):
    c = n = 0
    for s in segs:
        m = window_metrics(run(s), s["g"][d])
        c += m[:, 5].sum()
        n += m[:, 4].sum()
    return c / n


def search_tau(segs, make, lo, hi, log=True):
    """Largest tau with pooled event recall (b') >= TARGET (grid then bisection)."""
    grid = np.geomspace(lo, hi, 60) if log else np.linspace(lo, hi, 60)
    rec = np.array([pooled_recall(segs, lambda s: make(s, t)) for t in grid])
    ok = np.where(rec >= TARGET_RECALL)[0]
    assert len(ok), "target recall unreachable"
    i = ok.max()
    if i == len(grid) - 1:
        return float(grid[i]), float(rec[i])
    a, b = grid[i], grid[i + 1]
    for _ in range(25):
        mid = np.sqrt(a * b) if log else (a + b) / 2
        if pooled_recall(segs, lambda s: make(s, mid)) >= TARGET_RECALL:
            a = mid
        else:
            b = mid
    return float(a), float(pooled_recall(segs, lambda s: make(s, a)))


def main():
    segs = load_segments()
    main_segs = [s for s in segs if s["main"]]

    # ---- P4: logistic on causal cue features predicting g(b'), fit on 50 % of the 121 videos, frozen
    rng = np.random.default_rng(SEED)
    fit = []
    for ds in ("BMC", "CDnet2014", "LASIESTA"):
        u = sorted(s["uid"] for s in main_segs if s["dataset"] == ds)
        fit += list(rng.permutation(u)[: len(u) // 2])
    fit = set(fit)
    for s in segs:
        s["X"] = p4_features(s["dn"], s["fn"])
    X = np.vstack([s["X"] for s in main_segs if s["uid"] in fit])
    y = np.concatenate([s["g"]["bprime"] for s in main_segs if s["uid"] in fit])
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    for s in segs:
        s["pi"] = clf.predict_proba(s["X"])[:, 1]

    # ---- operating points at pooled event recall (b') = 0.90 on the 121 videos, frozen
    tau2, r2 = search_tau(main_segs, lambda s, t: p2(s["dn"], t, s["g"]["bprime"]), 0.2, 200)
    tau3, r3 = search_tau(main_segs, lambda s, t: p3(s["dn"], t, s["g"]["bprime"]), 0.2, 200)
    tau4, r4 = search_tau(main_segs, lambda s, t: p4(s["pi"], t, s["g"]["bprime"]), 1e-4, 0.999)
    rec_k = {K: float(np.mean([pooled_recall(main_segs, lambda s, K=K, f=f: p1(len(s["dn"]), K, f))
                               for f in range(K)])) for K in K_ALL}
    op = dict(spec="STRATA_SPEC v2", target_event_recall=TARGET_RECALL,
              event_definition="(b') = person conf>=0.7, idle gaps <= round(0.2 s x fps) merged",
              k_merge={"25 fps": 5, "30 fps": 6, "CDnet lowFramerate": 0}, pooled_over="121 main videos",
              P1=dict(K_prereg=list(K_PREREG), K_appendix="2..30", recall_by_K=rec_k),
              P2=dict(tau=tau2, recall=r2, Kr=15), P3=dict(tau=tau3, recall=r3, Kr=15, W=4),
              P4=dict(tau=tau4, recall=r4, Kr=15, coef=clf.coef_.ravel().tolist(),
                      intercept=float(clf.intercept_[0]),
                      features=["log dn", "log fn", "log max5 dn", "log mean30 dn"], target="g(b')",
                      fit_videos=sorted(fit),
                      fit_videos_sha256=hashlib.sha256("\n".join(sorted(fit)).encode()).hexdigest()),
              seed=SEED)
    (RES / "operating_points.json").write_text(json.dumps(op, indent=2))
    print(f"tau2 {tau2:.4f} ({r2:.4f})  tau3 {tau3:.4f} ({r3:.4f})  tau4 {tau4:.4f} ({r4:.4f})")
    print("P1 recall(b') by K:", {K: round(rec_k[K], 3) for K in (1, 2, 5, 10, 30)})

    # ---- windows and metrics for both definitions, same frozen operating points
    rows = []
    for d in DEFS:
        for s in segs:
            g = s["g"][d]
            wt = window_table(g)
            base = wt.assign(definition=d, dataset=s["dataset"], video=s["video"], video_uid=s["uid"],
                             fps=s["fps"], main=s["main"], win_idx=np.arange(len(wt)))
            pol = {"P2": p2(s["dn"], tau2, g), "P3": p3(s["dn"], tau3, g), "P4": p4(s["pi"], tau4, g)}
            for name, p in pol.items():
                m = window_metrics(p, g)
                rows.append(base.assign(policy=name, K=0, **{c: m[:, i] for i, c in enumerate(MCOLS)}))
            for K in K_ALL:
                m = np.mean([window_metrics(p1(len(g), K, f), g) for f in range(K)], axis=0)
                rows.append(base.assign(policy="P1", K=K, **{c: m[:, i] for i, c in enumerate(MCOLS)}))
    units = pd.concat(rows, ignore_index=True)

    # ---- cells (from the policy-free window table), per definition x rule x arm
    smap, wB, cells_all = [], [], {}
    for d in DEFS:
        win = units[(units.definition == d) & (units.policy == "P2")][["video_uid", "main", "key", "n"]]
        for rule in RULES:
            for arm, w in (("main", win[win.main]), ("121+VP", win)):
                mp, cells = build_strata_cells(w, rule)
                cells_all[(d, rule, arm)] = (mp, cells)
                units.loc[units.definition == d, f"h_{rule}_{arm}"] = units.loc[units.definition == d, "key"].map(
                    lambda k, mp=mp: mp.get(k, "U"))
                st = w.groupby("key").agg(n_win=("key", "size"), n_vid=("video_uid", "nunique"), frames=("n", "sum"))
                smap += [dict(definition=d, rule=rule, arm=arm, key=k, n_win=r.n_win, n_vid=r.n_vid,
                              frames=r.frames, h=mp[k]) for k, r in st.iterrows()]
                f = w.assign(h=w.key.map(mp)).groupby("h").n.sum()
                wB += [dict(definition=d, rule=rule, arm=arm, h=h, frames=int(v), w_B=float(v / f.sum()))
                       for h, v in f.items()]
    units.to_parquet(RES / "benchmark_units.parquet", index=False)
    pd.DataFrame(smap).to_csv(RES / "strata_map_benchmark.csv", index=False)
    pd.DataFrame(wB).to_csv(RES / "w_B_strata.csv", index=False, float_format="%.6g")

    # ---- m_B(cell): per-frame numerators/denominators over ALL member raw strata of the cell (main arm)
    mrows = []
    for d in DEFS:
        ud = units[(units.definition == d) & units.main]
        for rule in RULES:
            _, cells = cells_all[(d, rule, "main")]
            for cid, members in cells.items():
                x = ud[ud.key.isin(members)]
                g = x.groupby(["policy", "K"])
                agg = g[MCOLS].sum()
                agg["n_win"] = g.size() / 1
                agg["n_vid"] = g.video_uid.nunique()
                for met, (num, den) in METRICS.items():
                    agg[f"nu_{met}"] = agg[num] / agg["n"]
                    agg[f"de_{met}"] = agg[den] / agg["n"]
                    agg[f"m_{met}"] = agg[num] / agg[den].where(agg[den] > 0)
                mrows.append(agg.reset_index().assign(definition=d, rule=rule, h=cid, n_members=len(members)))
    mB = pd.concat(mrows, ignore_index=True)
    mB.to_csv(RES / "m_B_strata.csv", index=False, float_format="%.6g")

    # ---- summary
    sm = pd.DataFrame(smap)
    for d in DEFS:
        for rule in RULES:
            x = sm[(sm.definition == d) & (sm.rule == rule) & (sm.arm == "main")]
            print(f"{d:6s} {rule}: cells {x.h[x.h != 'U'].nunique():2d}  benchmark frames in U "
                  f"{x.frames[x.h == 'U'].sum() / x.frames.sum():.3f}")
    ou = units[(units.definition == "bprime") & units.main & ((units.policy != "P1") | units.K.isin(K_PREREG))]
    pooled = ou.groupby(["policy", "K"])[MCOLS].sum()
    for met, (num, den) in METRICS.items():
        pooled[met] = pooled[num] / pooled[den]
    print(pooled[list(METRICS)].round(4).to_string())


if __name__ == "__main__":
    main()
