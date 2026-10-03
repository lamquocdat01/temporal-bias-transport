"""Verification E2-E5 against the frozen pre-registration (prereg/PREREG_MEVA.json, CP1).

Evaluates exactly the rules written in the pre-registration; nothing here is tuned on outcomes.
  E2  Kitagawa-Oaxaca (ratio form, benchmark reference) per cell: composition = transport - naive,
      within = M_obs - transport, share = composition / (M_obs - naive); H2 at the 'all' unit.
  E3  per cell: naive / transport errors vs M_obs (transportable part), error ratio, PI / CI coverage;
      H3 on the pre-registered denominator; separate tables for expected-FAIL, pooling-dependent,
      deterministic and undefined cells; sensitivity with the deterministic cells included.
  E4  Theorem 1(b) bounds on ALL frames of a unit: plug-in (pre-registered) and oracle (observed supported
      part + U boxes); the observed full value must lie in the oracle bounds (else bug).
  E5  r* (pre-registered) vs TV(w_B, w_T) per unit; observed flips; AUC + bootstrap CI over pair x metric;
      ROC figure.  Holm over p_H1..p_H4.
Outputs: results/E2_decomposition.csv, E3_transport_cells.csv, E3_summary.csv, E4_bounds.csv, E5_flips.csv,
         E5_auc.json, fig_E3_errors.png, fig_E4_bounds.png, fig_E5_roc.png, results/numbers.json (section 'dot3').
Requires the pre-registration lock (same check as meva_outcomes.py).
"""
from __future__ import annotations
import variant as V
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from meva_outcomes import check_lock
from policies import METRICS
from prereg_build import bench_table, target_windows
from meva_frames import load_segments
from transport import Bench, Target, units, bounds_unit, POLICIES, PNAME, MET, SEED, B, TCRIT, tv

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
INV = {v: k for k, v in PNAME.items()}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = stats.rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def git_first_add(rel: str) -> str:
    """Commit that added the registration in the development repository (the release ships it in
    spec/PREREG_COMMITS.json, because the release history starts later)."""
    import subprocess
    rec = V.PRE_DIR / "PREREG_COMMITS.json"
    if rec.exists():
        return json.loads(rec.read_text())[Path(rel).name]
    out = subprocess.run(["git", "-C", str(REPO), "log", "--diff-filter=A", "--format=%h", "--", rel],
                         capture_output=True, text=True).stdout.split()
    return out[-1] if out else ""


def holm(p: dict) -> dict:
    k = sorted(p, key=lambda x: p[x])
    m, out, run = len(k), {}, 0.0
    for i, h in enumerate(k):
        run = max(run, min(1.0, (m - i) * p[h]))
        out[h] = run
    return out


def main():
    h = check_lock()
    pre = json.loads(V.PRE_JSON.read_text(encoding="utf-8"))
    cells = pd.DataFrame(pre["cells"])
    obs = pd.read_csv(RES / "meva_observed.csv")
    mu = pd.read_parquet(RES / "meva_units.parquet")

    # ---------------------------------------------------------------- E3 / E2 per cell
    ot = obs[obs.scope == "transportable"].set_index(["level", "unit", "policy", "metric"])
    of = obs[obs.scope == "full"].set_index(["level", "unit", "policy", "metric"])
    rows = []
    for c in cells.itertuples(index=False):
        k = (c.level, c.unit, c.policy, c.metric)
        o = ot.loc[k] if k in ot.index else None
        m_obs = float(o.value) if o is not None and o.den > 0 else np.nan
        den_obs = float(o.den) if o is not None else 0.0
        en = c.naive - m_obs
        et = c.transport - m_obs if c.transport is not None else np.nan
        ratio = abs(et) / abs(en) if (np.isfinite(et) and np.isfinite(en) and en != 0) else np.nan
        tr = c.transport if c.transport is not None else np.nan
        comp, within = tr - c.naive, m_obs - tr
        share = comp / (m_obs - c.naive) if np.isfinite(m_obs) and m_obs != c.naive else np.nan

        def cov(iv):
            return bool(iv is not None and iv[0] is not None and iv[0] <= m_obs <= iv[1]) if np.isfinite(m_obs) else None
        role = c.role.split(" ")[0]
        defined = np.isfinite(m_obs) and np.isfinite(et)
        rows.append(dict(level=c.level, unit=c.unit, policy=c.policy, metric=c.metric, role=role,
                         H3_expected=c.H3_expected, naive=c.naive, transport=tr, M_obs=m_obs, den_obs=den_obs,
                         err_naive=en, err_transport=et, abs_err_naive=abs(en), abs_err_transport=abs(et),
                         error_ratio=ratio, cell_PASS=(bool(abs(et) <= 0.5 * abs(en)) if defined else None),
                         PI_lo=c.PI[0] if c.PI else np.nan, PI_hi=c.PI[1] if c.PI else np.nan,
                         PI_half=c.PI_half, PI_covers=cov(c.PI), CI_twoway_covers=cov(c.CI_twoway),
                         CI_block_covers=cov(c.CI_block), composition=comp, within=within, composition_share=share,
                         omega_unit_all=c.omega_v2_unit_all, frames_transportable=c.frames_transportable,
                         M_full_obs=float(of.loc[k].value) if k in of.index else np.nan,
                         manski_L=c.manski_full[0], manski_U=c.manski_full[1],
                         P2form_L=c.manski_simple_P2[0], P2form_U=c.manski_simple_P2[1]))
    e3 = pd.DataFrame(rows)
    e3.to_csv(RES / "E3_transport_cells.csv", index=False, float_format="%.6g")
    e3[["level", "unit", "policy", "metric", "naive", "transport", "M_obs", "composition", "within",
        "composition_share"]].to_csv(RES / "E2_decomposition.csv", index=False, float_format="%.6g")

    # ---------------------------------------------------------------- H3
    den = e3[e3.role == "H3_denominator"]
    undefined_obs = den[den.cell_PASS.isna()]
    dd = den[den.cell_PASS.notna()]
    k3, n3 = int(dd.cell_PASS.sum()), len(dd)
    rate3 = k3 / n3
    p3 = float(stats.binomtest(k3, n3, 0.5, alternative="greater").pvalue)
    sens = e3[e3.role.isin(["H3_denominator", "deterministic_by_construction"]) & e3.cell_PASS.notna()]
    summ = []

    def add(name, x):
        x = x[x.cell_PASS.notna()]
        summ.append(dict(group=name, n=len(x), n_pass=int(x.cell_PASS.sum()),
                         pass_rate=x.cell_PASS.mean() if len(x) else np.nan,
                         median_error_ratio=x.error_ratio.median(), PI_coverage=x.PI_covers.mean(),
                         CI_twoway_coverage=x.CI_twoway_covers.mean(), CI_block_coverage=x.CI_block_covers.mean(),
                         median_abs_err_naive=x.abs_err_naive.median(),
                         median_abs_err_transport=x.abs_err_transport.median()))
    add("H3 denominator", den)
    add("  camera units", den[den.level == "camera"])
    add("  session units", den[den.level == "session"])
    for m in MET:
        add(f"  metric {m}", den[den.metric == m])
    for pname in [PNAME[p] for p in POLICIES]:
        add(f"  policy {pname}", den[den.policy == pname])
    add("expected FAIL (D3.5)", e3[e3.role == "expected_FAIL"])
    add("pooling-dependent block (D3.2)", e3[e3.role == "pooling_dependent"])
    add("deterministic by construction", e3[e3.role == "deterministic_by_construction"])
    add("sensitivity: denominator + deterministic", sens)
    add("'all' unit (summary)", e3[e3.level == "all"])
    E3s = pd.DataFrame(summ)
    E3s.to_csv(RES / "E3_summary.csv", index=False, float_format="%.4f")
    efail = e3[e3.role == "expected_FAIL"]

    # ---------------------------------------------------------------- H2 at the 'all' unit
    al = e3[(e3.level == "all")]
    need = pre["H2"]["thresholds"]["majority"]
    h2_metric, pms = {}, {}
    for m in pre["H2"]["thresholds"]["metrics"]:
        x = al[(al.metric == m) & ~((al.policy.str.startswith("P1")) & (m == "processed_frac"))]
        A = (x.abs_err_naive > x.PI_half)
        Bc = (x.composition_share >= 0.5)
        ok = int((A & Bc).sum())
        se = x.PI_half / TCRIT
        pv = 2 * stats.t.sf(np.abs(x.err_naive / se), 2)
        pms[m] = float(np.sort(pv)[need[m] - 1])
        h2_metric[m] = dict(n_cells=len(x), n_A=int(A.sum()), n_B=int(Bc.sum()), n_A_and_B=ok, need=need[m],
                            PASS=ok >= need[m], p_m=pms[m],
                            cells=x[["policy", "naive", "transport", "M_obs", "err_naive", "PI_half",
                                     "composition_share"]].round(6).to_dict("records"))
    h2_pass = any(v["PASS"] for v in h2_metric.values())
    p2 = min(1.0, 3 * min(pms.values()))

    # ---------------------------------------------------------------- E4 bounds
    bench = bench_table()
    segs = None if V.MEVA_WINDOWS else load_segments()
    tw = target_windows(segs)
    pdep = set(pre["design"]["pooling_dependent_blocks"])
    U = units(tw, pdep)
    Stot = bench.S.sum(0)
    mu_pre = mu[[(p, int(k)) in POLICIES for p, k in zip(mu.policy, mu.K)]].copy()
    mu_pre["pname"] = [PNAME[(p, int(k))] for p, k in zip(mu_pre.policy, mu_pre.K)]
    e4 = []
    for u in U:
        twu = tw[tw.block_id.isin(u["blocks_all"])]
        mm = mu_pre[mu_pre.block_id.isin(u["blocks_all"])]
        for (pol, K) in POLICIES:
            name = PNAME[(pol, K)]
            y = mm[mm.pname == name]
            p = POLICIES.index((pol, K))
            for met in MET:
                num, dn = METRICS[met]
                nu, de = Bench.nu_de(Stot, p, met)
                ys = y[y.cell != "U"]
                L, Ub, om, omE = bounds_unit(nu, de, twu, bench.cells, pol, K, met)
                Lo, Uo, _, _ = bounds_unit(nu, de, twu, bench.cells, pol, K, met,
                                           observed_supp=(float(ys[num].sum()), float(ys[dn].sum())))
                full = float(y[num].sum() / y[dn].sum()) if y[dn].sum() > 0 else np.nan
                pc = cells[(cells.level == u["level"]) & (cells.unit == u["unit"]) & (cells.policy == name)
                           & (cells.metric == met)].iloc[0]
                e4.append(dict(level=u["level"], unit=u["unit"], policy=name, metric=met, omega=om, omega_E=omE,
                               M_full_obs=full, plugin_L=L, plugin_U=Ub, plugin_width=Ub - L,
                               plugin_contains=bool(L - 1e-12 <= full <= Ub + 1e-12) if np.isfinite(full) else None,
                               oracle_L=Lo, oracle_U=Uo, oracle_width=Uo - Lo,
                               oracle_contains=bool(Lo - 1e-9 <= full <= Uo + 1e-9) if np.isfinite(full) else None,
                               P2form_L=pc.manski_simple_P2[0], P2form_U=pc.manski_simple_P2[1],
                               P2form_contains=bool(pc.manski_simple_P2[0] <= full <= pc.manski_simple_P2[1])
                               if np.isfinite(full) else None,
                               prereg_plugin_match=bool(abs(L - pc.manski_full[0]) < 1e-5 and abs(Ub - pc.manski_full[1]) < 1e-5)))
    e4 = pd.DataFrame(e4)
    e4.to_csv(RES / "E4_bounds.csv", index=False, float_format="%.6g")
    n_oracle_viol = int((e4.oracle_contains == False).sum())  # noqa: E712
    assert e4.prereg_plugin_match.all(), "plug-in bounds differ from the pre-registered ones"

    # ---------------------------------------------------------------- E5 flips / AUC
    rs = pd.DataFrame(pre["H4"]["r_star_table"])
    rs = rs[rs.in_H4]
    tgt = Target(tw, bench.cells)
    flips = []
    for u in [x for x in U if x["level"] in ("camera", "session")]:
        F = tgt.wvec(u["blocks"])[:-1]
        wT = F / F.sum()
        TVu = tv(bench.w_B_supp, wT)
        for r in rs.itertuples(index=False):
            o1 = ot.loc[(u["level"], u["unit"], r.p1, r.metric)]
            o2 = ot.loc[(u["level"], u["unit"], r.p2, r.metric)]
            if not (o1.den > 0 and o2.den > 0):
                flips.append(dict(level=u["level"], unit=u["unit"], pair=r.pair, metric=r.metric, defined=False))
                continue
            d = float(o1.value - o2.value)
            rstar = np.inf if r.r_star == "inf" else float(r.r_star)
            flip = (np.sign(d) != r.sign_B) or d == 0
            flips.append(dict(level=u["level"], unit=u["unit"], pair=r.pair, metric=r.metric, defined=True,
                              delta_B=r.delta_B, delta_obs=d, r_star=rstar, TV=TVu,
                              score=0.0 if not np.isfinite(rstar) else (np.inf if rstar == 0 else TVu / rstar),  # r* = 0: benchmark tie -> limit +inf
                              margin=-np.inf if not np.isfinite(rstar) else TVu - rstar, flip=bool(flip)))
    e5 = pd.DataFrame(flips)
    e5.to_csv(RES / "E5_flips.csv", index=False, float_format="%.6g")
    ed = e5[e5.defined]
    nflip, nno = int(ed.flip.sum()), int((~ed.flip.astype(bool)).sum())
    A4 = auc(ed.score, ed.flip.astype(bool))
    A4m = auc(ed.margin.replace(-np.inf, -10), ed.flip.astype(bool))
    rng = np.random.default_rng(SEED)
    keys = ed.pair + "|" + ed.metric
    uk = keys.unique()
    groups = {k: ed[keys == k] for k in uk}
    boot = []
    for _ in range(B):
        pick = rng.choice(uk, len(uk))
        x = pd.concat([groups[k] for k in pick])
        boot.append(auc(x.score, x.flip.astype(bool)))
    boot = np.array(boot)
    bv = boot[np.isfinite(boot)]
    p4 = float((np.sum(bv <= 0.5) + 1) / (len(bv) + 1))
    inconclusive = nflip < 5 or nno < 5
    h4_pass = (not inconclusive) and A4 >= 0.75
    e5j = dict(n_evaluations=len(e5), n_defined=len(ed), n_flips=nflip, n_nonflips=nno, AUC=A4,
               AUC_ci=list(np.percentile(bv, [2.5, 97.5])), p=p4, AUC_margin_sensitivity=A4m,
               inconclusive=inconclusive, PASS=h4_pass,
               flips_by_metric=ed.groupby("metric").flip.agg(["sum", "size"]).astype(int).to_dict("index"))
    (RES / "E5_auc.json").write_text(json.dumps(e5j, indent=1, default=float))

    # ---------------------------------------------------------------- Holm + numbers
    p1 = pre["H1"]["p_value"]
    ph = holm({"H1": p1, "H2": p2, "H3": p3, "H4": p4})
    verdict = {"H1": pre["H1"]["predicted"], "H2": "PASS" if h2_pass else "FAIL",
               "H3": "PASS" if rate3 >= 0.8 else "FAIL",
               "H4": "INCONCLUSIVE" if inconclusive else ("PASS" if h4_pass else "FAIL")}
    for k in verdict:
        if verdict[k] == "PASS" and ph[k] >= 0.05:
            verdict[k] = "PASS (not significant after Holm)"
    branch = ("H3 PASS -> 'transport correct on the transportable part + bounds for the rest + census'"
              if rate3 >= 0.8 else
              "H3 FAIL -> 'bounds + census are the contribution, transport is an evidenced negative lesson'")
    numbers = dict(
        prereg_sha256=h, prereg_commit=git_first_add(V.PRE_REL), n_cells=len(cells),
        H1=dict(verdict=verdict["H1"], p=p1, p_holm=ph["H1"], TV={k: v["TV"] for k, v in pre["H1"]["definitions"].items()},
                duty_in_ci={k: v["duty_MEVA_in_benchmark_ci"] for k, v in pre["H1"]["definitions"].items()}),
        H2=dict(verdict=verdict["H2"], p=p2, p_holm=ph["H2"],
                by_metric={m: {k: v for k, v in d.items() if k != "cells"} for m, d in h2_metric.items()}),
        H3=dict(verdict=verdict["H3"], n_denominator_prereg=len(den), n_evaluated=n3, n_undefined_observed=len(undefined_obs),
                n_pass=k3, pass_rate=rate3, threshold=0.8, p=p3, p_holm=ph["H3"],
                PI_coverage=float(dd.PI_covers.mean()), CI_twoway_coverage=float(dd.CI_twoway_covers.mean()),
                CI_block_coverage=float(dd.CI_block_covers.mean()),
                median_error_ratio=float(dd.error_ratio.median()),
                sensitivity_with_deterministic=dict(n=len(sens), pass_rate=float(sens.cell_PASS.mean())),
                expected_FAIL=dict(n=int(efail.cell_PASS.notna().sum()), n_pass=int(efail.cell_PASS.sum()),
                                   predicted_fail_confirmed=int((efail.cell_PASS == False).sum())),  # noqa: E712
                pooling_dependent=dict(n=int((e3.role == "pooling_dependent").sum()),
                                       n_pass=int(e3[e3.role == "pooling_dependent"].cell_PASS.sum()))),
        H4=dict(verdict=verdict["H4"], **{k: v for k, v in e5j.items() if k != "flips_by_metric"}, p_holm=ph["H4"]),
        E4=dict(n=len(e4), oracle_violations=n_oracle_viol, plugin_containment=float(e4.plugin_contains.mean()),
                P2form_containment=float(e4.P2form_contains.mean()),
                median_plugin_width=float(e4.plugin_width.median())),
        CP3_branch=branch, omega_S8_v2=0.514, omega_S8_v1=0.615)
    nj = V.NUMBERS
    allnum = json.loads(nj.read_text()) if nj.exists() else {}
    allnum["dot3"] = numbers
    nj.write_text(json.dumps(allnum, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else float(o)))

    figures(e3, e4, ed)
    print(json.dumps({k: numbers[k]["verdict"] for k in ("H1", "H2", "H3", "H4")}, indent=1))
    print(f"H3 {k3}/{n3} = {rate3:.3f}; H4 AUC {A4:.3f} flips {nflip}/{len(ed)}; oracle violations {n_oracle_viol}")
    print(E3s.round(3).to_string())


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)


def figures(e3, e4, ed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK})
    # E3: |error naive| vs |error transport|, denominator cells, log-log, reference lines y = x and y = 0.5x
    d = e3[(e3.role == "H3_denominator") & e3.cell_PASS.notna()]
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=200)
    lo = max(1e-6, min(d.abs_err_naive[d.abs_err_naive > 0].min(), d.abs_err_transport[d.abs_err_transport > 0].min()))
    hi = max(d.abs_err_naive.max(), d.abs_err_transport.max()) * 1.5
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=MUTED, lw=1, ls="--")
    ax.plot(xs, 0.5 * xs, color=INK, lw=1)
    for ok, col, lab in ((True, BLUE, "giảm ≥ 50 % (đạt)"), (False, ORANGE, "không đạt")):
        x = d[d.cell_PASS == ok]
        ax.scatter(x.abs_err_naive.clip(lower=lo), x.abs_err_transport.clip(lower=lo), s=14, color=col,
                   edgecolor="white", linewidth=0.6, label=f"{lab} (n = {len(x)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|sai số ngây thơ|"); ax.set_ylabel("|sai số transport|")
    ax.set_title("E3 — ô mẫu số H3 (đường liền: y = 0,5x)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout(); fig.savefig(RES / "fig_E3_errors.png"); plt.close(fig)
    # E4: plug-in bound width vs omega (camera/session units, processed fraction and false-open)
    x = e4[e4.level.isin(["camera", "session"])]
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=200)
    for met, col, lab in (("processed_frac", BLUE, "processed fraction"), ("false_open_per_frame", ORANGE, "false-open / khung")):
        y = x[x.metric == met]
        ax.scatter(y.omega, y.plugin_width, s=14, color=col, edgecolor="white", linewidth=0.6, label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("ω (khối khung MEVA ngoài hỗ trợ, v2)"); ax.set_ylabel("độ rộng cận ĐL1(b)")
    ax.set_title("E4 — độ rộng cận theo ω (đường đứt: độ rộng = ω)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout(); fig.savefig(RES / "fig_E4_bounds.png"); plt.close(fig)
    # E5: ROC
    y = ed.flip.astype(bool).values
    s = ed.score.values
    thr = np.r_[np.inf, np.unique(s)[::-1]]
    tpr = [(s[y] >= t).mean() if y.any() else np.nan for t in thr]
    fpr = [(s[~y] >= t).mean() for t in thr]
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.plot(fpr, tpr, color=BLUE, lw=2, drawstyle="steps-post")
    ax.set_xlabel("tỉ lệ dương giả"); ax.set_ylabel("tỉ lệ dương thật")
    ax.set_title(f"E5 — ROC: TV/r* → lật quan sát (AUC = {auc(s, y):.3f})", loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(RES / "fig_E5_roc.png"); plt.close(fig)


if __name__ == "__main__":
    main()
