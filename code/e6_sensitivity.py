"""E6 sensitivity (prereg/PREREG_MEVA.json §E6): repeat E3 along each pre-registered axis. Does not change H1-H4.

Generic pipeline per axis: benchmark (121 main videos) and MEVA S8 policy runs with FROZEN operating points / P4
coefficients -> window keys -> cells (support rule unchanged) -> transport + PI (t_2, conditional bootstrap + v_obs)
-> observed on the transportable part (pooling-dependent block removed, D3.2) -> cell PASS iff
|transport - M_obs| <= 0.5 |naive - M_obs|. Cell roles (denominator / expected-FAIL / deterministic / undefined) as in
the pre-registration. The 'main' axis re-derives E3 and must reproduce results/E3_transport_cells.csv.

Axes: main, b_raw, rule_v1, D_fine, cue_baseline, window_1800, norm_300, fps_15, fps_30, route_laptop.
Implementation choices not spelled out in the pre-registration (documented in DOT3_REPORT):
  D_fine      classes <=2 / 3-5 / 6-10 / 11-30 / 31-100 / >100; at the v2 D-step fine classes coarsen to le30 / gt30
  cue_baseline 4th key component B1..B3 = benchmark terciles of log10 max(m_t, 0.1) at the window's first frame
              (m_t = causal 1800-frame median of mad, frame 2 for a segment's first window); v2 steps R -> D -> C,
              then a 4th step merging B
Outputs: results/E6_cells.csv (axis x cell), results/E6_summary.csv, results/E6_route_laptop.json
"""
from __future__ import annotations
import variant as V
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import strata as ST
from bench_units import load_segments as bench_load, LOWFR
from meva_frames import load_segments as meva_load
from meva_outcomes import check_lock
from policies import MCOLS, METRICS
from run_policies import policy_windows
from transport import (Bench, Target, units, bootstrap_unit, v_obs, transport_point, POLICIES, PNAME, MET,
                       TCRIT)

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
KS = (2, 5, 10, 30)
PDEP = {"G331_2018-03-07_165000"}
D_FINE = ((2, "d1-2"), (5, "d3-5"), (10, "d6-10"), (30, "d11-30"), (100, "d31-100"), (10 ** 9, "dgt100"))


# ------------------------------------------------------------------ window keys
def keys_standard(g, win, **_):
    return ST.window_table(g, win)[["w_start", "n", "key"]]


def keys_dfine(g, win, **_):
    wt = ST.window_table(g, win)
    s, L, cens = ST.events(g)
    e = s + L
    out = []
    for r in wt.itertuples(index=False):
        if r.S == "idle":
            out.append(r.key)
            continue
        a, b = r.w_start, r.w_start + r.n
        ov = (s < b) & (e > a)
        Lmax = int(L[ov].max())
        lab = next(lb for ub, lb in D_FINE if Lmax <= ub)
        if Lmax <= 30 and cens[ov].any():
            lab = "cens>=" + lab
        out.append("|".join([r.S, lab, r.R, f"C{r.C}"]))
    return wt.assign(key=out)[["w_start", "n", "key"]]


def m_baseline(mad):
    s = pd.Series(mad, dtype=float)
    m = s.rolling(1800, min_periods=1).median().shift(1)
    m.iloc[:2] = s.iloc[:2]
    return np.log10(np.maximum(m.to_numpy(), 0.1))


def keys_cue(g, win, mad=None, cuts=None, **_):
    wt = ST.window_table(g, win)
    lm = m_baseline(mad)
    idx = np.maximum(wt.w_start.values, 2).clip(max=len(lm) - 1)
    v = lm[idx]
    cls = np.where(np.isnan(v), 1, 1 + (v > cuts[0]).astype(int) + (v > cuts[1]).astype(int))
    return wt.assign(key=[f"{k}|B{c}" for k, c in zip(wt.key, cls)])[["w_start", "n", "key"]]


# ------------------------------------------------------------------ cells (generalised v2 pooled rule)
def coarsen(key, dims, fine=False):
    parts = key.split("|")
    S, D, R, C = parts[:4]
    extra = parts[4:]
    if "R" in dims and R in ("lt300", "300-1800"):
        R = "lt1800"
    if "D" in dims:
        if fine:
            base = D.replace("cens>=", "")
            hi = base in ("d31-100", "dgt100")
            D = ("cens>=" if D.startswith("cens>=") else "") + ("gt30" if hi else "le30") if base != "none" else D
        else:
            D = {"le5": "le30", "6-30": "le30", "cens>=le5": "cens>=le30", "cens>=6-30": "cens>=le30"}.get(D, D)
    if "C" in dims and not R.startswith("cens") and not D.startswith("cens"):
        C = "C0"
    if "B" in dims:
        extra = []
    return "|".join([S, D, R, C] + extra)


def cells_v2(bw, order=("R", "D", "C"), fine=False):
    stats = bw.groupby("key").agg(n_win=("key", "size"))
    vids = bw.groupby("key").video_uid.apply(set).to_dict()
    raw = sorted(stats.index)
    ok = lambda ks: stats.loc[ks, "n_win"].sum() >= ST.MIN_WIN and len(set().union(*[vids[k] for k in ks])) >= ST.MIN_VID  # noqa: E731
    mp = {}
    for k in raw:
        if ok([k]):
            mp[k] = k
            continue
        mp[k] = "U"
        for i in range(1, len(order) + 1):
            dims = order[:i]
            ck = coarsen(k, dims, fine)
            members = [j for j in raw if coarsen(j, dims, fine) == ck]
            if ok(members):
                mp[k] = f"P{i}:{ck}"
                break
    return mp


# ------------------------------------------------------------------ policy runs -> window rows
def run_side(segs, gkey, win, norm_win, keyf, idcols, cuts=None):
    rows = []
    for s in segs:
        g = s[gkey] if isinstance(gkey, str) else gkey(s)
        kt = keyf(g, win, mad=s["mad"], cuts=cuts)
        res = policy_windows(s["mad"], s["fpx"], g, Ks=KS, win=win, norm_win=norm_win)
        base = kt.assign(**{c: s[c] for c in idcols})
        for (pol, K), m in res.items():
            assert np.allclose(m[:, 0], kt.n.values)
            rows.append(base.assign(policy=pol, K=K, **{c: m[:, i] for i, c in enumerate(MCOLS)}))
    return pd.concat(rows, ignore_index=True)


def bench_segments():
    segs = [s for s in bench_load() if s["main"]]
    cue = pd.read_parquet(V.CACHE / "bench_cue.parquet")
    for s in segs:
        x = cue[(cue.dataset == s["dataset"]) & (cue.video == s["video"])].sort_values("frame")
        s.update(mad=x.mad.values, fpx=x.fpx.values, g_b=s["g"]["b"], g_bp=s["g"]["bprime"],
                 video_uid=s["uid"], assumed_fps=(s["dataset"] in ("CDnet2014", "LASIESTA") and s["video"] not in LOWFR))
    return segs


# ------------------------------------------------------------------ evaluation of one axis
def evaluate(axis, bu, mu, rule="v2", fine=False, order=("R", "D", "C")):
    bw = bu[(bu.policy == "P2")][["video_uid", "key", "n"]]
    if rule == "v1":
        mp = ST.build_strata_map(bw)
    else:
        mp = cells_v2(bw, order, fine)
    bu = bu.assign(cell=bu.key.map(lambda k: mp.get(k, "U")))
    mu = mu.assign(cell=mu.key.map(lambda k: mp.get(k, "U")))
    bench = Bench(bu)
    tw = mu[mu.policy == "P2"][["block_id", "cam_tag", "session", "w_start", "n", "E", "n_ev", "cell"]].reset_index(drop=True)
    tgt = Target(tw, bench.cells)
    Stot = bench.S.sum(0)
    out = []
    for u in [x for x in units(tw, PDEP) if x["level"] in ("camera", "session", "all")]:
        btc = bootstrap_unit(bench, tgt, u, "cond")
        Fs = tgt.wvec(u["blocks"])[:-1]
        om = mu[mu.block_id.isin(u["blocks"]) & (mu.cell != "U")]
        omega_all = float(tgt.wvec(u["blocks_all"])[-1] / tgt.wvec(u["blocks_all"]).sum())
        for p, (pol, K) in enumerate(POLICIES):
            y = om[(om.policy == pol) & (om.K == K)]
            name = PNAME[(pol, K)]
            for j, met in enumerate(MET):
                num, den = METRICS[met]
                nu, de = Bench.nu_de(Stot, p, met)
                ok = Fs > 0
                M = transport_point(nu[ok], de[ok], Fs[ok]) if ok.any() else np.nan
                naive = bench.naive(p, met)
                D = float(y[den].sum())
                mobs = float(y[num].sum()) / D if D > 0 else np.nan
                vb = float(np.nanvar(btc[:, p, j], ddof=1)) if np.isfinite(btc[:, p, j]).sum() > 1 else np.nan
                vo = v_obs(bench, tgt, u, p, met, M, de) if np.isfinite(M) else np.nan
                half = TCRIT * np.sqrt(vb + (vo if np.isfinite(vo) else 0)) if np.isfinite(vb) else np.nan
                en, et = naive - mobs, M - mobs
                defined = np.isfinite(en) and np.isfinite(et)
                if u["level"] == "all":
                    role = "summary"
                elif not defined:
                    role = "undefined"
                elif (pol in ("P2", "P3")) and met == "frame_recall":
                    role = "expected_FAIL"
                elif pol == "P1" and met in ("processed_frac", "frame_recall"):
                    role = "deterministic_by_construction"
                else:
                    role = "H3_denominator"
                out.append(dict(axis=axis, level=u["level"], unit=u["unit"], policy=name, metric=met, role=role,
                                naive=naive, transport=M, M_obs=mobs, err_naive=en, err_transport=et,
                                cell_PASS=bool(abs(et) <= 0.5 * abs(en)) if defined else None,
                                PI_half=half, PI_covers=bool(abs(et) <= half) if defined and np.isfinite(half) else None,
                                omega_unit_all=omega_all, n_cells_bench=len(bench.cells),
                                bench_frac_U=float(bench.w_B.get("U", 0.0))))
    return pd.DataFrame(out)


def main():
    check_lock()
    only = sys.argv[1:]
    bsegs = bench_segments()
    msegs = meva_load()
    bid, mid = ["video_uid"], ["block_id", "cam_tag", "session"]

    def axis_data(gk_b="g_bp", gk_m="g_bp", win=300, norm=1800, keyf=keys_standard, cuts=None, bsegs_=None):
        bu = run_side(bsegs_ or bsegs, gk_b, win, norm, keyf, bid, cuts)
        mu = run_side(msegs, gk_m, win, norm, keyf, mid, cuts)
        return bu, mu

    axes = {
        "main": lambda: evaluate("main", *axis_data()),
        "b_raw": lambda: evaluate("b_raw", *axis_data("g_b", "g_b")),
        "rule_v1": lambda: evaluate("rule_v1", *axis_data(), rule="v1"),
        "D_fine": lambda: evaluate("D_fine", *axis_data(keyf=keys_dfine), fine=True),
        "window_1800": lambda: evaluate("window_1800", *axis_data(win=1800)),
        "norm_300": lambda: evaluate("norm_300", *axis_data(norm=300)),
    }

    def cue_axis():
        lm = np.concatenate([m_baseline(s["mad"])[np.maximum(np.arange(0, len(s["mad"]), 300), 2).clip(max=len(s["mad"]) - 1)]
                             for s in bsegs])
        cuts = np.nanpercentile(lm, [100 / 3, 200 / 3])
        bu, mu = axis_data(keyf=keys_cue, cuts=cuts)
        r = evaluate("cue_baseline", bu, mu, order=("R", "D", "C", "B"))
        return r.assign(note=f"tercile cuts log10 m = {cuts[0]:.4f}, {cuts[1]:.4f}")
    axes["cue_baseline"] = cue_axis

    def fps_axis(fps):
        def g_alt(s):
            return ST.merge_gaps(s["g_b"], ST.k_merge(fps)) if s["assumed_fps"] else s["g_bp"]
        bu = run_side(bsegs, g_alt, 300, 1800, keys_standard, bid)
        mu = run_side(msegs, "g_bp", 300, 1800, keys_standard, mid)
        return evaluate(f"fps_{fps}", bu, mu)
    axes["fps_15"] = lambda: fps_axis(15)
    axes["fps_30"] = lambda: fps_axis(30)

    res = []
    prev = RES / "E6_cells.csv"
    done = pd.read_csv(prev) if prev.exists() and only else None
    for name, f in axes.items():
        if only and name not in only:
            continue
        print("axis", name, flush=True)
        res.append(f())
    # route: the 6 P-1 clips with laptop detections (expected identical g)
    if not only or "route_laptop" in only:
        from paths import MEVA_DIR
        lap = MEVA_DIR / "detections_laptop_ov_gpu_fp32x"
        stems = {p.stem for p in lap.glob("*.json")}
        nfr, ndiff = 0, 0
        import json as _j
        for r in msegs:                                       # laptop detections only for the 6 stems
            if not stems & set(r["stems"]):
                continue
            off = 0
            for st in r["stems"]:
                n = int(np.load(MEVA_DIR / "cue" / f"{st}.npz")["n_decoded"])
                if st in stems:
                    fr = _j.load(open(lap / f"{st}.json"))["frames"]
                    gb = np.array([any(b[5] == 0 and b[4] >= 0.7 for b in f["boxes"]) for f in fr], np.int8)
                    ndiff += int((gb != r["g_b"][off:off + n]).sum())
                    nfr += n
                off += n
        same = ndiff == 0
        (RES / "E6_route_laptop.json").write_text(json.dumps(dict(
            clips=sorted(stems), frames=nfr, g_b_frames_differing=ndiff, identical=same,
            consequence="E6 route_laptop == main (identical g(b) and therefore g(b'), windows, policies)"
            if same else "differences found -- rerun required"), indent=1))
        print("route_laptop identical:", same, nfr, ndiff)
        if same and res and (res[0].axis == "main").all():
            res.append(res[0].assign(axis="route_laptop"))
    out = pd.concat(res, ignore_index=True)
    if done is not None:
        out = pd.concat([done[~done.axis.isin(out.axis.unique())], out], ignore_index=True)
    out.to_csv(RES / "E6_cells.csv", index=False, float_format="%.6g")
    # check: main axis reproduces E3
    e3 = pd.read_csv(RES / "E3_transport_cells.csv")
    m = out[out.axis == "main"].merge(e3, on=["level", "unit", "policy", "metric"], suffixes=("", "_e3"))
    if len(m):
        dmax = float(np.nanmax(np.abs(np.r_[m.transport - m.transport_e3, m.M_obs - m.M_obs_e3, m.naive - m.naive_e3])))
        print(f"main axis vs E3: {len(m)} cells, max |diff| = {dmax:.2e}")
        assert dmax < 2e-5, dmax                    # stored E3/prereg values are rounded to 6 digits
    summ = []
    for ax, x in out.groupby("axis", sort=False):
        d = x[(x.role == "H3_denominator") & x.cell_PASS.notna()]
        summ.append(dict(axis=ax, n_denominator=len(d), n_pass=int(d.cell_PASS.sum()), pass_rate=d.cell_PASS.mean(),
                         median_error_ratio=float((d.err_transport.abs() / d.err_naive.abs()).median()),
                         PI_coverage=d.PI_covers.mean(), omega_all=float(x[x.level == "all"].omega_unit_all.iloc[0]),
                         n_cells_bench=int(x.n_cells_bench.iloc[0]), bench_frac_U=float(x.bench_frac_U.iloc[0]),
                         H3_criterion_met=bool(d.cell_PASS.mean() >= 0.8)))
    s = pd.DataFrame(summ)
    s.to_csv(RES / "E6_summary.csv", index=False, float_format="%.4f")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
