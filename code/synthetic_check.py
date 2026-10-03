"""A6: numerical check of the theory on synthetic data with a KNOWN generator (seed 42).

Generator: discrete-time alternating renewal process (on = event, off = idle).
  on  ~ 0.5 Geometric(1/4)  +  0.5 round(LogNormal(log 40, 0.9))           (shared by B and T)
  off ~ round(LogNormal(log 80 * a_cam * b_sess, 1.0)) capped at 1700       ("short idle")
  target only: with prob 0.4 an off is Uniform{300..1700} ("medium idle")  -> different profile w_T
  quiet scenario (A2 violated): with prob q an off is round(LogNormal(log 20000, 0.5))  -> R >= 1800 (U)
  Benchmark: 121 clips, L ~ LogUniform(250, 3000), cut from a stationary stream, a=b=1.
  Target: C cameras x S sessions, 36,000 frames each (= 1 h at 10 fps), camera effect a_c and
  session effect b_s ~ LogNormal(0, 0.3) on the idle scale (two-way cluster structure).
  Cue: d_t = exp(0.4 Z) + g_t * Bern(0.7) * (2 + Exp(1)); normalised causally exactly as STRATA_SPEC §5.
Policies (local minimal versions, NOT code/policies.py):
  P1 periodic K=10, phase-averaged in closed form (frame 0 always processed; event caught w.p. min(1, d/K)).
  P2 threshold tau on normalised cue + backstop 15 (frame 0 processed); tau set once so that pooled
     benchmark event recall ~ 0.90.
Strata: code/strata.py (window_table, build_strata_map) -- the pipeline's own implementation.
Estimator: post-stratified ratio (Theorem 1a). Intervals:
  TW  : cluster bootstrap over benchmark videos x two-way pigeonhole bootstrap (cameras x sessions)
        of the target profile w_T; percentile CI and SE-based CI with t_{min(C,S)-1} critical value.
  COND: benchmark bootstrap only (w_T fixed at the observed profile) + a predictive term for the
        within-stratum noise of the observed target metric (normal PI).
Truths: M*_T  = transported estimand with population quantities (A1-free: checks inference only),
        M_T   = population target metric (checks A1 + inference; A1 holds only approximately here),
        M_obs = the metric observed on the replicate's own target sample (what H3 compares with).
Also: (i) Manski bounds under A2 failure, width vs omega; (iii) Lemma 3 KM vs naive; (iv) P1 recall.
Outputs: results/synthetic_check.json, results/fig_synthetic_coverage.png
Usage: python code/synthetic_check.py [R] [B]     (defaults R=200, B=500)
"""
from __future__ import annotations
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata as ST                       # noqa: E402
from flip_radius import ratio_bounds_lp, r_star_linear, min_linear_in_ball  # noqa: E402

SEED = 42
REPO = Path(__file__).resolve().parents[1]
K1, KR, WIN = 10, 15, 300
METRICS = ["proc", "frec", "erec", "fo", "prec"]      # (num, den) in window_sums
NUMDEN = {"proc": ("P", "n"), "frec": ("TP", "E"), "erec": ("caught", "ev"), "fo": ("fo", "n"), "prec": ("TP", "P")}
POL = ["P1", "P2"]
T_LEN = 36_000


# ------------------------------------------------------------------ generator
def draw_on(rng, n):
    geo = rng.geometric(0.25, n)
    ln = np.maximum(1, np.round(rng.lognormal(np.log(40), 0.9, n))).astype(int)
    return np.where(rng.random(n) < 0.5, geo, ln)


def draw_off(rng, n, scale=1.0, medium=0.0, quiet=0.0):
    off = np.clip(np.round(rng.lognormal(np.log(80 * scale), 1.0, n)), 1, 1700).astype(int)
    u = rng.random(n)
    off = np.where(u < medium, rng.integers(300, 1701, n), off)
    if quiet > 0:
        q = np.maximum(1, np.round(rng.lognormal(np.log(20000), 0.5, n))).astype(int)
        off = np.where(rng.random(n) < quiet, q, off)
    return off


def stream(rng, length, burn=4000, **kw):
    tot, parts = length + burn, []
    acc = 0
    while acc < tot:
        m = 200
        on, off = draw_on(rng, m), draw_off(rng, m, **kw)
        seg = np.empty(2 * m, int); seg[0::2] = off; seg[1::2] = on
        parts.append(seg); acc += seg.sum()
    durs = np.concatenate(parts)
    vals = np.tile([0, 1], len(durs) // 2)
    g = np.repeat(vals, durs)[:tot]
    return g[burn:].astype(np.int8)


def cue(rng, g):
    d = np.exp(0.4 * rng.standard_normal(len(g))) + g * (rng.random(len(g)) < 0.7) * (2 + rng.exponential(1, len(g)))
    s = pd.Series(d)
    med = s.rolling(1800, min_periods=1).median().shift(1)
    med.iloc[:2] = s.iloc[:2]
    dn = (s / np.maximum(med, 0.1)).to_numpy()
    dn[0] = np.nan
    return dn


# ------------------------------------------------------------------ policies -> per-window sums
def window_sums(g, dn, tau):
    n = len(g)
    nw = (n + WIN - 1) // WIN
    wi = np.arange(n) // WIN
    s, L, cens = ST.events(g)
    out = {}
    # ---- P1 (closed form, phase-averaged)
    p = np.full(n, 1.0 / K1); p[0] = 1.0
    caught = np.where(s == 0, 1.0, np.minimum(1.0, L / K1))
    fo = p * (1 - g)                       # K>=2: every processed idle frame is its own run
    out["P1"] = dict(P=np.bincount(wi, p, nw), TP=np.bincount(wi, p * g, nw),
                     caught=np.bincount(s // WIN, caught, nw), fo=np.bincount(wi, fo, nw))
    # ---- P2 (threshold + backstop)
    op = np.nan_to_num(dn, nan=0.0) > tau
    op[0] = True
    last = np.maximum.accumulate(np.where(op, np.arange(n), -1))
    pr = ((np.arange(n) - last) % (KR + 1)) == 0
    cp = np.r_[0, np.cumsum(pr)]
    c2 = (cp[s + L] - cp[s]) > 0
    dpr = np.diff(np.r_[0, pr.astype(np.int8), 0])
    rs, re = np.where(dpr == 1)[0], np.where(dpr == -1)[0]
    cg = np.r_[0, np.cumsum(g)]
    fo_run = (cg[re] - cg[rs]) == 0
    out["P2"] = dict(P=np.bincount(wi, pr, nw), TP=np.bincount(wi, pr * g, nw),
                     caught=np.bincount(s // WIN, c2, nw), fo=np.bincount(rs // WIN, fo_run, nw))
    base = dict(n=np.bincount(wi, None, nw), E=np.bincount(wi, g, nw), ev=np.bincount(s // WIN, None, nw))
    return base, out


def segment_table(rng, g, tau, **ids):
    dn = cue(rng, g)
    wt = ST.window_table(g)
    base, pol = window_sums(g, dn, tau)
    d = {"key": wt.key.to_numpy(), **base}
    for pn, dct in pol.items():
        for k, v in dct.items():
            d[f"{pn}_{k}"] = v
    for k, v in ids.items():
        d[k] = np.full(len(wt), v)
    return pd.DataFrame(d)


BENCH_MEDIUM = 0.15      # benchmark also has some medium idles, so that A2 holds in the main study


def bench_sample(rng, n_clips, tau, medium=BENCH_MEDIUM):
    L = np.exp(rng.uniform(np.log(250), np.log(3000), n_clips)).astype(int)
    return pd.concat([segment_table(rng, stream(rng, l, medium=medium), tau, video_uid=i) for i, l in enumerate(L)],
                     ignore_index=True)


def target_sample(rng, C, S, tau, quiet=0.0, a=None, b=None):
    a = rng.lognormal(0, 0.3, C) if a is None else a
    b = rng.lognormal(0, 0.3, S) if b is None else b
    rows = []
    for c in range(C):
        for s_ in range(S):
            g = stream(rng, T_LEN, scale=a[c] * b[s_], medium=0.4, quiet=quiet)
            rows.append(segment_table(rng, g, tau, cam=c, sess=s_))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ estimation
def cols(pol, m):
    num, den = NUMDEN[m]
    f = lambda x: x if x in ("n", "E", "ev") else f"{pol}_{x}"  # noqa: E731
    return f(num), f(den)


def direct(df, pol, m):
    nc, dc = cols(pol, m)
    return df[nc].sum() / df[dc].sum()


def build_cells(bw, rule):
    """Returns (target_map: raw key -> cell id or 'U', cells: cell id -> list of benchmark raw keys).
    rule 'pipeline': code/strata.build_strata_map (partition; unsupported strata pooled only among
    themselves). rule 'pooled' (= strata.build_strata_cells v2, order R -> D -> C): an unsupported stratum is mapped, at the first coarsening step whose
    cell is supported, to the cell of ALL benchmark raw strata sharing its coarsened key (supported
    siblings included; they also keep their own cell). Raw keys absent from the benchmark -> U."""
    raw = sorted(set(bw.key))
    if rule == "pipeline":
        mp = ST.build_strata_map(bw)
        cells = {}
        for k, c in mp.items():
            if c != "U":
                cells.setdefault(c, []).append(k)
        return mp, cells
    return ST.build_strata_cells(bw, "v2")   # STRATA_SPEC v2 pooled rule (order R -> D -> C)


def bench_arrays(bw, groups):
    """V x G arrays of frames and of every numerator/denominator column."""
    V = bw.video_uid.max() + 1
    gi = bw.grp.map({g: i for i, g in enumerate(groups)}).to_numpy()
    ok = ~pd.isna(gi)
    vi = bw.video_uid.to_numpy()[ok]; gi = gi[ok].astype(int)
    arr = {}
    for c in ["n", "E", "ev"] + [f"{p}_{k}" for p in POL for k in ("P", "TP", "caught", "fo")]:
        A = np.zeros((V, len(groups)))
        np.add.at(A, (vi, gi), bw[c].to_numpy()[ok])
        arr[c] = A
    return arr


def target_frames(tw, groups, C, S):
    A = np.zeros((C, S, len(groups)))
    gi = tw.grp.map({g: i for i, g in enumerate(groups)})
    ok = gi.notna().to_numpy()
    np.add.at(A, (tw.cam.to_numpy()[ok], tw.sess.to_numpy()[ok], gi[ok].astype(int).to_numpy()), tw.n.to_numpy()[ok])
    return A


def transport(w, arr_sum_frames, arr_num, arr_den, fallback=None):
    """w: (...,G) target frame weights; arr_*: (...,G) benchmark sums -> ratio estimate.
    A group absent from a bootstrap resample takes the full-sample per-frame mean (fallback)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        nu = arr_num / arr_sum_frames
        de = arr_den / arr_sum_frames
    if fallback is not None:
        nu = np.where(arr_sum_frames > 0, nu, fallback[0]); de = np.where(arr_sum_frames > 0, de, fallback[1])
    nu = np.nan_to_num(nu); de = np.nan_to_num(de)
    return (w * nu).sum(-1) / (w * de).sum(-1)


def cell_arrays(bw, cells, groups):
    raw = sorted(set(bw.key))
    bw = bw.assign(grp=bw.key)
    A = bench_arrays(bw, raw)                                   # V x raw
    Mm = np.zeros((len(raw), len(groups)))
    ri = {k: i for i, k in enumerate(raw)}
    for j, c in enumerate(groups):
        for k in cells[c]:
            Mm[ri[k], j] = 1
    return {k: v @ Mm for k, v in A.items()}


def one_rep(rng, tau, C, S, Bn, pop_raw, rule="pooled", data=None):
    bw, tw = data if data is not None else (bench_sample(rng, 121, tau), target_sample(rng, C, S, tau))
    mp, cells = build_cells(bw, rule)
    tw = tw.assign(grp=tw.key.map(lambda k: mp.get(k, "U")))
    groups = sorted(cells)
    arr = cell_arrays(bw, cells, groups)
    Tf = target_frames(tw, groups, C, S)
    omega = tw.n[tw.grp == "U"].sum() / tw.n.sum()
    w_obs = Tf.sum((0, 1))
    V = arr["n"].shape[0]
    Wv = rng.multinomial(V, np.ones(V) / V, Bn).astype(float)           # B x V
    mc = rng.multinomial(C, np.ones(C) / C, Bn).astype(float)           # B x C
    ms = rng.multinomial(S, np.ones(S) / S, Bn).astype(float)           # B x S
    w_tw = np.einsum("bc,bs,csg->bg", mc, ms, Tf)
    fr_b = Wv @ arr["n"]
    # population truths for this replicate's map
    pb, pt = pop_raw
    rawsum = pb.groupby("key").sum(numeric_only=True)
    pbs = pd.DataFrame({c: [rawsum.reindex(cells[g])[c].fillna(0).sum() for g in groups] for c in rawsum.columns},
                       index=groups)
    pt_g = pt.assign(grp=pt.key.map(lambda k: mp.get(k, "U")))
    pts = pt_g[pt_g.grp != "U"].groupby("grp").n.sum().reindex(groups).fillna(0).to_numpy()
    mp_pipe = ST.build_strata_map(bw) if rule != "pipeline" else mp
    res = {"omega": float(omega), "n_groups": len(groups),
           "omega_pipeline_rule": float(tw.n[tw.key.map(lambda k: mp_pipe.get(k, "U")) == "U"].sum() / tw.n.sum())}
    tcrit = stats.t.ppf(0.975, min(C, S) - 1)
    for pol in POL:
        for m in METRICS:
            nc, dc = cols(pol, m)
            est = transport(w_obs, arr["n"].sum(0), arr[nc].sum(0), arr[dc].sum(0))
            fb = (arr[nc].sum(0) / arr["n"].sum(0), arr[dc].sum(0) / arr["n"].sum(0))
            bt = transport(w_tw, fr_b, Wv @ arr[nc], Wv @ arr[dc], fb)
            bc = transport(w_obs[None], fr_b, Wv @ arr[nc], Wv @ arr[dc], fb)
            m_star = transport(pts, pbs["n"].to_numpy(), pbs[nc].to_numpy(), pbs[dc].to_numpy())
            m_obs = direct(tw, pol, m)
            # predictive within-stratum variance of the observed target metric (windows independent)
            zr = bw[nc] - est * bw[dc]
            s2 = np.array([zr[bw.key.isin(cells[g_])].var() for g_ in groups])
            s2 = np.nan_to_num(s2)
            nwin_t = tw[tw.grp != "U"].groupby("grp").size().reindex(groups).fillna(0).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                de_hat = (arr[dc].sum(0) / arr["n"].sum(0))
            Dhat = (w_obs * np.nan_to_num(de_hat)).sum()
            v_obs = (nwin_t * s2).sum() / Dhat ** 2
            sd_c = np.nanstd(bc)
            res[f"{pol}_{m}"] = dict(
                est=float(est), m_star=float(m_star), m_obs=float(m_obs),
                tw_pct=[float(np.nanpercentile(bt, 2.5)), float(np.nanpercentile(bt, 97.5))],
                tw_se=float(np.nanstd(bt)),
                cond_pct=[float(np.nanpercentile(bc, 2.5)), float(np.nanpercentile(bc, 97.5))],
                pi=[float(est - 1.96 * np.sqrt(sd_c ** 2 + v_obs)), float(est + 1.96 * np.sqrt(sd_c ** 2 + v_obs))],
                tcrit=float(tcrit))
    return res


def population(rng, tau, n_bench=2000, n_cells=300):
    t0 = time.time()
    pb = bench_sample(rng, n_bench, tau)
    cells = []
    for i in range(n_cells):
        a, b = rng.lognormal(0, 0.3), rng.lognormal(0, 0.3)
        g = stream(rng, T_LEN, scale=a * b, medium=0.4)
        cells.append(segment_table(rng, g, tau, cam=0, sess=i))
    pt = pd.concat(cells, ignore_index=True)
    keep = ["key", "n", "E", "ev"] + [f"{p}_{k}" for p in POL for k in ("P", "TP", "caught", "fo")]
    truth = {f"{p}_{m}": float(direct(pt, p, m)) for p in POL for m in METRICS}
    print(f"population built in {time.time() - t0:.0f}s: bench {len(pb)} win, target {len(pt)} win", flush=True)
    return (pb[keep], pt[keep]), truth


def pick_tau(rng):
    L = np.exp(rng.uniform(np.log(250), np.log(3000), 400)).astype(int)
    segs = [stream(rng, l, medium=BENCH_MEDIUM) for l in L]
    dns = [cue(rng, g) for g in segs]
    best = None
    for tau in np.arange(1.0, 6.01, 0.1):
        c = e = 0
        for g, dn in zip(segs, dns):
            _, pol = window_sums(g, dn, tau)
            c += pol["P2"]["caught"].sum(); e += len(ST.events(g)[0])
        rec = c / e
        if rec >= 0.90:
            best = (float(tau), float(rec))
    return best


# ------------------------------------------------------------------ extra checks
def check_bounds(rng, tau, q_list, reps=20):
    out = []
    for q in q_list:
        for r in range(reps):
            bw = bench_sample(rng, 121, tau)
            tw = target_sample(rng, 5, 4, tau, quiet=q)
            mp, cells = build_cells(bw, "pooled")
            tw["grp"] = tw.key.map(lambda k: mp.get(k, "U"))
            groups = sorted(cells)
            arr = cell_arrays(bw, cells, groups)
            tot = tw.n.sum()
            wS = tw[tw.grp != "U"].groupby("grp").n.sum().reindex(groups).fillna(0).to_numpy() / tot
            U = tw[tw.grp == "U"]
            omega = float(U.n.sum() / tot)
            bs = pd.DataFrame({k: v.sum(0) for k, v in arr.items()}, index=groups)
            row = dict(q=q, rep=r, omega=omega)
            eU = U.E.sum() / max(U.n.sum(), 1); evU = U.ev.sum() / max(U.n.sum(), 1)
            for pol in POL:
                for m in METRICS:
                    nc, dc = cols(pol, m)
                    N_S = float((wS * bs[nc] / bs.n).sum()); D_S = float((wS * bs[dc] / bs.n).sum())
                    lo_p = 18 / 300
                    I = np.eye(2)
                    if m == "proc":
                        A, b = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([1, -lo_p, 1, -1])
                    elif m == "frec":
                        A, b = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([eU, 0, eU, -eU])
                    elif m == "erec":
                        A, b = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([evU, 0, evU, -evU])
                    elif m == "fo":
                        A, b = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([0.5, 0, 1, -1])
                    else:   # precision: x = TP <= min(y, eU), lo_p <= y <= 1
                        A = np.array([[1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]]); b = np.array([0, eU, 0, 1, -lo_p])
                    # oracle: supported part taken from the target itself (checks the bound logic exactly)
                    Tsup = tw[tw.grp != "U"]
                    N_O = float(Tsup[nc].sum() / tot); D_O = float(Tsup[dc].sum() / tot)
                    if omega > 0:
                        lo, hi = ratio_bounds_lp(N_S, D_S, [omega], [(A, b)])
                        lo_o, hi_o = ratio_bounds_lp(N_O, D_O, [omega], [(A, b)])
                    else:
                        lo = hi = N_S / D_S
                        lo_o = hi_o = N_O / D_O
                    m_obs = direct(tw, pol, m)
                    row[f"{pol}_{m}"] = dict(lo=lo, hi=hi, lo_oracle=lo_o, hi_oracle=hi_o, obs=float(m_obs), width=hi - lo)
            out.append(row)
    return out


def check_lemma3(rng):
    big = stream(rng, 3_000_000)
    s, L, _ = ST.events(big)
    true_med, true_mean = float(np.median(L)), float(L.mean())
    res = []
    for Lw in (100, 300, 1000, 3000, 10000):
        n_clips = max(50, 600_000 // Lw)
        obs, cen_s, cen_e = [], [], []
        for _ in range(n_clips):
            g = stream(rng, Lw)
            s, d, _ = ST.events(g)
            obs.append(d); cen_s.append(s == 0); cen_e.append(s + d == Lw)
        d = np.concatenate(obs); cs = np.concatenate(cen_s); ce = np.concatenate(cen_e)
        km_all = km_median(d, cs | ce)                 # CP0 method: any boundary touch = censored
        km_drop = km_median(d[~cs], ce[~cs])           # drop onset-censored, end-censored = censored
        res.append(dict(L=Lw, n_events=int(len(d)), frac_censored=float((cs | ce).mean()),
                        naive_median_all=float(np.median(d)), naive_median_uncensored=float(np.median(d[~(cs | ce)])),
                        km_median_both_censored=km_all, km_median_drop_onset=km_drop,
                        naive_mean_all=float(d.mean()), true_median=true_med, true_mean=true_mean))
    return res


def km_median(dur, cens):
    dur = np.asarray(dur, float); obs = ~np.asarray(cens, bool)
    S = 1.0
    for t in np.unique(dur[obs]):
        nr = (dur >= t).sum(); dd = ((dur == t) & obs).sum()
        S *= 1 - dd / nr
        if S <= 0.5:
            return float(t)
    return None


def check_p1(rng):
    worst = 0.0
    for _ in range(2000):
        K = int(rng.integers(1, 31)); d = int(rng.integers(1, 91)); s = int(rng.integers(1, 1000))
        hit = np.mean([any(((t - ph) % K) == 0 for t in range(s, s + d)) for ph in range(K)])
        worst = max(worst, abs(hit - min(1, d / K)))
    return worst


def check_rstar(rng, tau):
    """r* (Prop. 2) on a synthetic stratum table for P1 vs P2 processed fraction, vs brute force."""
    bw = bench_sample(rng, 121, tau)
    mp = ST.build_strata_map(bw); bw["grp"] = ST.apply_map(bw.key, mp)
    t = bw[bw.grp != "U"].groupby("grp").sum(numeric_only=True)
    w_B = (t.n / t.n.sum()).to_numpy()
    delta = ((t.P1_P - t.P2_P) / t.n).to_numpy()
    r, w = r_star_linear(delta, w_B)
    v_before, _ = min_linear_in_ball(np.sign(w_B @ delta) * delta, w_B, max(r - 1e-4, 0))
    brute = np.inf
    H = len(w_B)
    for _ in range(20000):
        x = rng.dirichlet(np.ones(H) * 0.3)
        if np.sign(x @ delta) != np.sign(w_B @ delta):
            lo, hi = 0.0, 1.0
            for _k in range(40):          # smallest mix toward x that flips
                mid = (lo + hi) / 2
                y = w_B + mid * (x - w_B)
                (hi, lo) = (mid, lo) if np.sign(y @ delta) != np.sign(w_B @ delta) else (hi, mid)
            y = w_B + hi * (x - w_B)
            brute = min(brute, 0.5 * np.abs(y - w_B).sum())
    return dict(H=H, r_star=float(r), brute_force_min_found=float(brute), no_flip_just_below=bool(v_before > 0))


# ------------------------------------------------------------------ main
def summarise(reps, key_truth, truth_pop):
    out = {}
    for pol in POL:
        for m in METRICS:
            k = f"{pol}_{m}"
            X = [r[k] for r in reps]
            ms = np.array([x["m_star"] for x in X]); mo = np.array([x["m_obs"] for x in X])
            est = np.array([x["est"] for x in X]); tp = truth_pop[k]
            tw = np.array([x["tw_pct"] for x in X]); se = np.array([x["tw_se"] for x in X]); tc = X[0]["tcrit"]
            cp = np.array([x["cond_pct"] for x in X]); pi = np.array([x["pi"] for x in X])
            inside = lambda iv, v: np.mean((iv[:, 0] <= v) & (v <= iv[:, 1]))  # noqa: E731
            twt = np.c_[est - tc * se, est + tc * se]
            out[k] = dict(
                cov_TW_pct_mstar=inside(tw, ms), cov_TW_pct_Mpop=inside(tw, np.full_like(ms, tp)),
                cov_TW_t_mstar=inside(twt, ms), cov_TW_t_Mpop=inside(twt, np.full_like(ms, tp)),
                cov_COND_pct_obs=inside(cp, mo), cov_PI_obs=inside(pi, mo),
                a1_bias_mean=float(np.mean(ms) - tp), est_bias_vs_mstar=float(np.mean(est - ms)),
                mean_width_TW_pct=float(np.mean(tw[:, 1] - tw[:, 0])), M_pop=tp)
    return out


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    Bn = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    ckpt = REPO / "cache" / f"synthetic_ckpt_R{R}_B{Bn}.pkl"
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    if ckpt.exists():                       # resume: restores results so far AND the rng state
        st = pickle.load(open(ckpt, "rb"))
        res, pop_raw, truth, tau, done = st["res"], st["pop_raw"], st["truth"], st["tau"], st["done"]
        rng.bit_generator.state = st["rng"]
        print(f"resumed from {ckpt.name}: {({k: len(v) for k, v in done.items()})}", flush=True)
    else:
        res = dict(meta=dict(seed=SEED, R=R, B=Bn, strata_py_sha256=hashlib.sha256(
            (REPO / "code" / "strata.py").read_bytes()).hexdigest()))
        tau, rec = pick_tau(rng)
        res["meta"].update(tau=tau, bench_event_recall_at_tau=rec)
        print("tau", tau, rec, flush=True)
        pop_raw, truth = population(rng, tau)
        res["truth_population"] = truth
        done = {}

    def save():
        ckpt.parent.mkdir(exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        pickle.dump(dict(res=res, pop_raw=pop_raw, truth=truth, tau=tau, done=done,
                         rng=rng.bit_generator.state), open(tmp, "wb"))
        tmp.replace(ckpt)

    for name, C, S in (("5x4", 5, 4), ("5x3", 5, 3)):
        reps = done.setdefault(name, [])
        for r in range(len(reps), R):
            reps.append(one_rep(rng, tau, C, S, Bn, pop_raw))
            if (r + 1) % 10 == 0:
                save()
                print(f"{name} rep {r + 1}/{R}  {time.time() - t0:.0f}s (checkpoint)", flush=True)
        res[f"coverage_{name}"] = summarise(reps, "m_star", truth)
        res[f"omega_{name}"] = dict(mean=float(np.mean([x["omega"] for x in reps])),
                                    max=float(np.max([x["omega"] for x in reps])),
                                    mean_pipeline_rule=float(np.mean([x["omega_pipeline_rule"] for x in reps])),
                                    n_groups_mean=float(np.mean([x["n_groups"] for x in reps])))
    # one final run with B = 1000 (interval widths only)
    fin = one_rep(rng, tau, 5, 4, 1000, pop_raw)
    res["final_B1000_5x4"] = {k: v for k, v in fin.items() if isinstance(v, dict)}
    print("bounds...", flush=True)
    bd = check_bounds(rng, tau, [0.002, 0.005, 0.01, 0.02], reps=10)
    bsum = {}
    for pol in POL:
        for m in METRICS:
            k = f"{pol}_{m}"
            om = np.array([r["omega"] for r in bd]); wd = np.array([r[k]["width"] for r in bd])
            cont = np.mean([(r[k]["lo"] - 1e-3 <= r[k]["obs"] <= r[k]["hi"] + 1e-3) for r in bd])
            cont_o = np.mean([(r[k]["lo_oracle"] - 1e-9 <= r[k]["obs"] <= r[k]["hi_oracle"] + 1e-9) for r in bd])
            slope = float(np.sum(om * wd) / np.sum(om * om))
            bsum[k] = dict(containment_plugin=float(cont), containment_oracle=float(cont_o), width_over_omega=slope,
                           r2_width_vs_omega=float(np.corrcoef(om, wd)[0, 1] ** 2) if wd.std() > 0 else None)
    res["bounds_A2_violated"] = dict(summary=bsum, omega_by_q={q: float(np.mean([r["omega"] for r in bd if r["q"] == q]))
                                                               for q in (0.002, 0.005, 0.01, 0.02)})
    print("lemma3...", flush=True)
    res["lemma3"] = check_lemma3(rng)
    res["p1_formula_max_abs_err"] = check_p1(rng)
    res["r_star_check"] = check_rstar(rng, tau)
    res["meta"]["runtime_s"] = round(time.time() - t0)
    (REPO / "results" / "synthetic_check.json").write_text(json.dumps(res, indent=1, default=float))
    make_fig(res, bd)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("coverage")}, indent=1, default=float)[:6000])


def make_fig(res, bd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    keys = [f"{p}_{m}" for p in POL for m in METRICS]
    x = np.arange(len(keys))
    for j, (nm, col, lab) in enumerate((("coverage_5x4", "cov_TW_pct_mstar", "TW pct, 5x4 (M*)"),
                                         ("coverage_5x3", "cov_TW_pct_mstar", "TW pct, 5x3 (M*)"),
                                         ("coverage_5x3", "cov_TW_t_mstar", "TW t, 5x3 (M*)"),
                                         ("coverage_5x4", "cov_PI_obs", "PI, 5x4 (M_obs)"))):
        ax[0].bar(x + (j - 1.5) * 0.2, [res[nm][k][col] for k in keys], 0.2, label=lab)
    ax[0].axhline(0.95, color="k", lw=0.8); ax[0].axhline(0.93, color="r", lw=0.8, ls="--")
    ax[0].set_xticks(x); ax[0].set_xticklabels(keys, rotation=60, fontsize=7); ax[0].set_ylim(0.5, 1.0)
    ax[0].set_title("Coverage of 95% intervals (R reps)", loc="left", fontsize=10); ax[0].legend(fontsize=7)
    om = np.array([r["omega"] for r in bd])
    for k in ("P2_proc", "P2_fo", "P2_frec", "P2_prec"):
        ax[1].scatter(om, [r[k]["width"] for r in bd], s=12, label=k)
    ax[1].set_xlabel("omega (target mass without benchmark support)"); ax[1].set_ylabel("bound width")
    ax[1].set_title("Theorem 1(b): bound width vs omega", loc="left", fontsize=10); ax[1].legend(fontsize=7)
    l3 = pd.DataFrame(res["lemma3"])
    for c in ("naive_median_all", "naive_median_uncensored", "km_median_both_censored", "km_median_drop_onset"):
        ax[2].plot(l3.L, l3[c].astype(float), marker="o", label=c)
    ax[2].axhline(l3.true_median.iloc[0], color="k", ls="--", label="true median")
    ax[2].set_xscale("log"); ax[2].set_xlabel("clip length L (frames)"); ax[2].set_ylabel("median event duration")
    ax[2].set_title("Lemma 3: truncation bias and KM", loc="left", fontsize=10); ax[2].legend(fontsize=7)
    fig.savefig(REPO / "results" / "fig_synthetic_coverage.png", dpi=150)


if __name__ == "__main__":
    main()
