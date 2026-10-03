"""E7 -- mechanism analysis. POST-HOC, EXPLORATORY: not part of the pre-registration (prereg/PREREG_MEVA.json);
does not touch H1-H4 and changes no pre-registered quantity.

Question: why does transport work for the stateless P1 but not for the cue-state-dependent P2-P4?
  1.1 per supported v2 cell: distribution of the normalised cue d~ (benchmark vs MEVA S8 transportable part):
      quantiles 50/90/99, share d~ > tau2, share pi-hat > tau4, sub-threshold run lengths  -> E7_cue_by_stratum.csv
  1.2 within-stratum processed-fraction gap of P2/P3/P4 split by WHY a frame was processed
      (segment start / cue above threshold / hold W / backstop Kr = 15)                     -> E7_trigger_decomposition.csv
  1.3 normalisation memory 300 / 1800 / 18000 frames: within-cell distance benchmark vs MEVA of log10 d~
      (Wasserstein-1, TV on 60 bins), MEVA-mass weighted                                     -> E7_memory.csv
  1.4 fig_mechanism.pdf (two-column): left same-cell d~ distributions, right transport error by policy
  numbers -> results/numbers.json section 'e7'
Operating points and P4 coefficients are the frozen ones (results/operating_points.json).
"""
from __future__ import annotations
import variant as V
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

from bench_units import load_segments as bench_load
from meva_frames import load_segments as meva_load
from meva_outcomes import check_lock
from policies import p2, p3, p4, KR, HOLD
from run_policies import OPS, normalise, pihat

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
PDEP = {"G331_2018-03-07_165000"}
WIN = 300
TAU2, TAU3, TAU4 = OPS["P2"]["tau"], OPS["P3"]["tau"], OPS["P4"]["tau"]
REASONS = ["start", "cue", "hold", "backstop"]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#e6e5e1"
LABEL = "post-hoc, exploratory (not pre-registered)"


@njit(cache=True)
def _thresh_reason(score, tau, g, hold):
    """Same decision rule as policies._thresh; also returns why each processed frame was taken:
    1 segment start, 2 cue above threshold, 3 hold window, 4 backstop. Precedence when several apply:
    start > cue > hold > backstop (the cue is credited whenever it fired)."""
    n = len(score)
    p = np.zeros(n, np.int8)
    why = np.zeros(n, np.int8)
    gap, h = 0, 0
    for t in range(n):
        cue = score[t] > tau
        take = t == 0 or gap >= KR or h > 0 or cue
        if take:
            p[t] = 1
            if t == 0:
                why[t] = 1
            elif cue:
                why[t] = 2
            elif h > 0:
                why[t] = 3
            else:
                why[t] = 4
            gap = 0
            h = h - 1 if h > 0 else 0
            if hold > 0 and g[t] == 1:
                h = hold
        else:
            gap += 1
    return p, why


def reasons(score, tau, g, hold):
    s = np.nan_to_num(np.asarray(score, float), nan=np.inf)
    return _thresh_reason(s, float(tau), np.asarray(g, np.int8), hold)


def runs_below(dn, tau):
    """Lengths and start indices of maximal runs with d~ <= tau (frame 0 excluded)."""
    b = np.r_[0, (np.nan_to_num(dn[1:], nan=np.inf) <= tau).astype(np.int8), 0]
    d = np.diff(b)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s + 1, e - s


# ------------------------------------------------------------------ frame tables
def frame_table(side: str, norm_wins=(1800,)):
    """One row per frame with cell, segment id, d~ for each memory, pi-hat (main memory), and per-frame reasons."""
    if side == "bench":
        segs = [s for s in bench_load() if s["main"]]
        cue = pd.read_parquet(V.CACHE / "bench_cue.parquet")
        bu = pd.read_parquet(RES / "benchmark_units.parquet")
        bu = bu[(bu.definition == "bprime") & bu.main & (bu.policy == "P2")]
        cellmap = {(r.video_uid, r.win_idx): r.h_v2_main for r in bu.itertuples(index=False)}
    else:
        segs = [s for s in meva_load() if s["block_id"] not in PDEP]
        mu = pd.read_parquet(RES / "meva_units.parquet")
        mu = mu[mu.policy == "P2"]
        cellmap = {(r.block_id, r.seg, r.win_idx): r.cell for r in mu.itertuples(index=False)}
    out = []
    for i, s in enumerate(segs):
        if side == "bench":
            x = cue[(cue.dataset == s["dataset"]) & (cue.video == s["video"])].sort_values("frame")
            mad, fpx, g = x.mad.values, x.fpx.values, s["g"]["bprime"]
            key = lambda w, s=s: (s["uid"], w)  # noqa: E731
        else:
            mad, fpx, g = s["mad"], s["fpx"], s["g_bp"]
            key = lambda w, s=s: (s["block_id"], s["seg"], w)  # noqa: E731
        n = len(g)
        w = np.arange(n) // WIN
        cols = dict(seg=i, t=np.arange(n), win=w, g=g, cell=[cellmap[key(k)] for k in w])
        for nw in norm_wins:
            dn, fn = normalise(mad, fpx, nw)
            cols[f"dn{nw}"] = dn
            if nw == 1800:
                cols["pi"] = pihat(dn, fn)
                for name, sc, tau, hold, ref in (("P2", dn, TAU2, 0, p2), ("P3", dn, TAU3, HOLD, p3),
                                                 ("P4", cols["pi"], TAU4, 0, p4)):
                    p, why = reasons(sc, tau, g, hold)
                    assert np.array_equal(p, ref(sc, tau, g)), f"instrumented {name} differs from policies.py"
                    cols[f"why_{name}"] = why
                rs, rl = runs_below(dn, TAU2)
                rr = np.zeros(n, np.int32)
                rr[rs] = rl
                cols["run_start_len"] = rr
        out.append(pd.DataFrame(cols))
    f = pd.concat(out, ignore_index=True)
    f["side"] = side
    return f


# ------------------------------------------------------------------ 1.1
def cue_by_stratum(fb, fm):
    rows = []
    for cell in sorted(set(fb.cell) - {"U"}):
        for side, f in (("benchmark", fb), ("MEVA", fm)):
            x = f[(f.cell == cell)]
            d = x.dn1800.dropna()
            if len(x) == 0:
                rows.append(dict(cell=cell, side=side, frames=0))
                continue
            rl = x.run_start_len[x.run_start_len > 0]
            rows.append(dict(cell=cell, side=side, frames=len(x), q50=d.quantile(0.5), q90=d.quantile(0.9),
                             q99=d.quantile(0.99), share_dn_gt_tau2=float((d > TAU2).mean()),
                             share_pi_gt_tau4=float((x.pi > TAU4).mean()), n_subthreshold_runs=len(rl),
                             run_len_median=rl.median() if len(rl) else np.nan,
                             run_len_q90=rl.quantile(0.9) if len(rl) else np.nan,
                             share_runs_reaching_backstop=float((rl >= KR).mean()) if len(rl) else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 1.2
def trigger_decomposition(fb, fm):
    """Within-stratum term of the processed-fraction gap, sum_h w_T(h) [nu_T(h) - nu_B(h)], split by reason."""
    wT = fm[fm.cell != "U"].groupby("cell").size()
    wT = wT / wT.sum()
    rows = []
    for pol in ("P2", "P3", "P4"):
        col = f"why_{pol}"
        tot = {"benchmark": 0.0, "MEVA": 0.0}
        for r_i, r in enumerate(REASONS, start=1):
            nb = fb[fb.cell.isin(wT.index)].groupby("cell")[col].apply(lambda s, r_i=r_i: (s == r_i).mean())
            nm = fm[fm.cell.isin(wT.index)].groupby("cell")[col].apply(lambda s, r_i=r_i: (s == r_i).mean())
            tb = float((wT * nb.reindex(wT.index).fillna(0)).sum())
            tm = float((wT * nm.reindex(wT.index).fillna(0)).sum())
            rows.append(dict(policy=pol, reason=r, transported_rate=tb, MEVA_rate=tm, within_gap=tm - tb))
            tot["benchmark"] += tb
            tot["MEVA"] += tm
        rows.append(dict(policy=pol, reason="total", transported_rate=tot["benchmark"], MEVA_rate=tot["MEVA"],
                         within_gap=tot["MEVA"] - tot["benchmark"]))
    d = pd.DataFrame(rows)
    tot = d[d.reason == "total"].set_index("policy").within_gap
    d["share_of_within_gap"] = [r.within_gap / tot[r.policy] if r.reason != "total" else 1.0 for r in d.itertuples()]
    return d


# ------------------------------------------------------------------ 1.3
def memory_distance(fb, fm, mems=(300, 1800, 18000)):
    wT = fm[fm.cell != "U"].groupby("cell").size()
    wT = wT / wT.sum()
    bins = np.linspace(-2, 2, 61)
    rows = []
    for m in mems:
        col = f"dn{m}"
        acc_w, acc_tv, acc_ex = 0.0, 0.0, 0.0
        for cell, wt in wT.items():
            a = np.log10(np.clip(fb.loc[fb.cell == cell, col].dropna().values, 1e-2, 1e2))
            b = np.log10(np.clip(fm.loc[fm.cell == cell, col].dropna().values, 1e-2, 1e2))
            if len(a) == 0 or len(b) == 0:
                continue
            W = stats.wasserstein_distance(a, b)
            ha = np.histogram(a, bins)[0] / len(a)
            hb = np.histogram(b, bins)[0] / len(b)
            T = 0.5 * np.abs(ha - hb).sum()
            ex = abs((10 ** b > TAU2).mean() - (10 ** a > TAU2).mean())
            rows.append(dict(memory=m, cell=cell, w_T=wt, wasserstein_log10=W, TV_hist=T, abs_diff_share_gt_tau2=ex))
            acc_w += wt * W
            acc_tv += wt * T
            acc_ex += wt * ex
        rows.append(dict(memory=m, cell="WEIGHTED", w_T=1.0, wasserstein_log10=acc_w, TV_hist=acc_tv,
                         abs_diff_share_gt_tau2=acc_ex))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 1.4 figure
def figure(fb, fm, cell, e3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelcolor": INK, "pdf.fonttype": 42})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw=dict(width_ratios=[1.15, 1]))
    bins = np.linspace(-1.5, 1.5, 61)
    for f, col, lab in ((fb, BLUE, "benchmark"), (fm, ORANGE, "MEVA S8")):
        v = np.log10(np.clip(f.loc[f.cell == cell, "dn1800"].dropna().values, 10 ** -1.5, 10 ** 1.5))
        a1.hist(v, bins, density=True, histtype="step", lw=1.6, color=col, label=f"{lab} (n = {len(v):,})")
    a1.axvline(np.log10(TAU2), color=INK, lw=1, ls="--")
    a1.text(np.log10(TAU2) + 0.04, a1.get_ylim()[1] * 0.92, r"$\tau_2$", fontsize=8)
    a1.set_xlabel(r"$\log_{10}\tilde d_t$ (normalised frame-difference cue)")
    a1.set_ylabel("density")
    a1.set_title(f"(a) one stratum, {cell}", loc="left")
    a1.legend(frameon=False, fontsize=7, loc="upper right")
    # (b) median |transport error| / |naive error| per policy over the H3 denominator cells, bootstrap CI
    d = e3[(e3.role == "H3_denominator") & e3.cell_PASS.notna()].copy()
    d["ratio"] = d.abs_err_transport / d.abs_err_naive
    pols = ["P1_K2", "P1_K5", "P1_K10", "P1_K30", "P2", "P3", "P4"]
    rng = np.random.default_rng(42)
    med, lo, hi = [], [], []
    for p in pols:
        r = d[d.policy == p].ratio.values
        bs = [np.median(rng.choice(r, len(r))) for _ in range(1000)]
        med.append(np.median(r))
        lo.append(np.percentile(bs, 2.5))
        hi.append(np.percentile(bs, 97.5))
    x = np.arange(len(pols))
    cols = [BLUE] * 4 + [ORANGE] * 3
    a2.axhline(0.5, color=INK, lw=1)
    a2.axhline(1.0, color=MUTED, lw=1, ls="--")
    for i in range(len(pols)):
        a2.plot([x[i], x[i]], [lo[i], hi[i]], color=cols[i], lw=1.5)
        a2.plot(x[i], med[i], "o", ms=5, color=cols[i], mec="white", mew=0.8)
    a2.set_xticks(x, ["P1\nK=2", "P1\nK=5", "P1\nK=10", "P1\nK=30", "P2", "P3", "P4"])
    a2.set_ylabel(r"median $|e_{\rm transport}|/|e_{\rm naive}|$")
    a2.set_title("(b) transport error by policy (H3 cells)", loc="left")
    a2.text(6.35, 0.52, "0.5 (H3)", fontsize=7, ha="right", va="bottom", color=INK)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=7)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(RES / "fig_mechanism.pdf")
    fig.savefig(RES / "fig_mechanism.png", dpi=200)
    plt.close(fig)
    return dict(policies=pols, median_ratio=med, ci_lo=lo, ci_hi=hi)


def main():
    check_lock()
    fb = frame_table("bench", (300, 1800, 18000))
    fm = frame_table("meva", (300, 1800, 18000))
    c = cue_by_stratum(fb, fm)
    c.insert(0, "label", LABEL)
    c.to_csv(RES / "E7_cue_by_stratum.csv", index=False, float_format="%.5g")
    t = trigger_decomposition(fb, fm)
    t.insert(0, "label", LABEL)
    t.to_csv(RES / "E7_trigger_decomposition.csv", index=False, float_format="%.5g")
    m = memory_distance(fb, fm)
    m.insert(0, "label", LABEL)
    m.to_csv(RES / "E7_memory.csv", index=False, float_format="%.5g")
    # figure cell: supported cell with the largest MEVA transportable mass among partially active windows
    wT = fm[fm.cell != "U"].groupby("cell").size().sort_values(ascending=False)
    part = [k for k in wT.index if k.startswith("partial") or ":partial" in k]
    cell = part[0]
    e3 = pd.read_csv(RES / "E3_transport_cells.csv")
    fig = figure(fb, fm, cell, e3)
    # numbers
    mw = m[m.cell == "WEIGHTED"].set_index("memory")
    cc = c.pivot_table(index="cell", columns="side", values=["share_dn_gt_tau2", "q50", "share_runs_reaching_backstop"])
    sup = wT.index
    wsum = wT / wT.sum()
    exceed_b = float((cc["share_dn_gt_tau2"]["benchmark"].reindex(sup) * wsum).sum())
    exceed_m = float((cc["share_dn_gt_tau2"]["MEVA"].reindex(sup) * wsum).sum())
    tt = t.set_index(["policy", "reason"])
    e7 = dict(label=LABEL, figure_cell=cell,
              exceed_tau2_transported=exceed_b, exceed_tau2_MEVA=exceed_m,
              q50_cell_benchmark=float(cc["q50"]["benchmark"][cell]), q50_cell_MEVA=float(cc["q50"]["MEVA"][cell]),
              within_gap={p: float(tt.loc[(p, "total"), "within_gap"]) for p in ("P2", "P3", "P4")},
              within_gap_cue={p: float(tt.loc[(p, "cue"), "within_gap"]) for p in ("P2", "P3", "P4")},
              within_gap_backstop={p: float(tt.loc[(p, "backstop"), "within_gap"]) for p in ("P2", "P3", "P4")},
              within_gap_hold={p: float(tt.loc[(p, "hold"), "within_gap"]) for p in ("P2", "P3", "P4")},
              memory_wasserstein={int(k): float(v) for k, v in mw.wasserstein_log10.items()},
              memory_TV={int(k): float(v) for k, v in mw.TV_hist.items()},
              memory_exceed_diff={int(k): float(v) for k, v in mw.abs_diff_share_gt_tau2.items()},
              distance_reduced_by_some_memory=bool((mw.wasserstein_log10.drop(1800) < mw.wasserstein_log10[1800]).any()
                                                   or (mw.TV_hist.drop(1800) < mw.TV_hist[1800]).any()),
              fig_b=fig)
    nj = json.loads((V.NUMBERS).read_text())
    nj["e7"] = e7
    (V.NUMBERS).write_text(json.dumps(nj, indent=1, default=float))
    print(json.dumps({k: v for k, v in e7.items() if k != "fig_b"}, indent=1, default=float))
    print(t.round(4).to_string())
    print(m[m.cell == "WEIGHTED"].round(4).to_string())


if __name__ == "__main__":
    main()
