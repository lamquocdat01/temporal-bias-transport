"""Build the MEVA pre-registration (CP1): prereg/PREREG_MEVA.json + prereg/PREREG_MEVA.md.

Inputs allowed for the H2-H4 predictions (docs/PREREG_DECISIONS.md, D3.7): results/benchmark_units.parquet,
results/operating_points.json, results/strata_map_benchmark.csv, results/w_B_strata.csv, and the MEVA (b') PROFILE
(window strata of the S8 frames, g(b') only: frames, windows, event frames, event onsets per window). No policy is run
on MEVA here; no (c) row and no (b) / (a) MEVA row enters H2-H4 (asserted). H1 is "confirmatory-of-profile" and uses
the census profiles (a), (b'), (c) by design (prompt §1.3).

Usage: python code/prereg_build.py      (writes the two files; hashing + commit are separate steps)
"""
from __future__ import annotations
import variant as V
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import flip_radius as fr
from meva_frames import load_segments
from strata import window_table, events, build_strata_cells
from transport import (Bench, Target, units, bootstrap_unit, v_obs, bounds_unit, simple_bounds, transport_point,
                       POLICIES, PNAME, MET, TCRIT, DF, B, SEED, FPS_T, tv, pidx)

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
PRE = REPO / "prereg"
DEF, RULE, ARM = "bprime", "v2", "main"
EXPECTED_FAIL = {("P2", "frame_recall"), ("P3", "frame_recall")}            # D3.5
# P1 averaged over phases processes every frame with probability exactly 1/K (frame 0 of a segment always): its
# processed fraction and frame recall are fixed by construction up to segment-start frames (no content dependence)
DETERMINISTIC = {(PNAME[p], m) for p in POLICIES if p[0] == "P1" for m in ("processed_frac", "frame_recall")}
H2_METRICS = ("processed_frac", "event_recall", "false_open_per_frame")
METRIC_LABEL = {"processed_frac": "processed fraction", "frame_recall": "frame recall", "event_recall": "event recall",
                "false_open_per_frame": "false-open / khung", "precision": "precision"}


def path_of(p: str) -> Path:
    """Input path of the current variant (results/<variant>/..., cache/<variant>/...)."""
    if p.startswith("results/"):
        return V.RES / p[len("results/"):]
    if p in ("cache/meva_windows.parquet",):
        return V.CACHE / p[len("cache/"):]
    return REPO / p


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def r6(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r6(v) for v in x]
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


# ------------------------------------------------------------------ inputs
def bench_table(definition=DEF, rule=RULE) -> Bench:
    u = pd.read_parquet(RES / "benchmark_units.parquet")
    u = u[(u.definition == definition) & u.main].copy()
    u["cell"] = u[f"h_{rule}_main"]
    return Bench(u)


def target_windows(segs, g_key="g_bp", definition=DEF, rule=RULE) -> pd.DataFrame:
    """Profile-only window table of the S8 target: strata key, cell, frames, event frames, event onsets."""
    smap = pd.read_csv(RES / "strata_map_benchmark.csv")
    smap = smap[(smap.definition == definition) & (smap.rule == rule) & (smap.arm == ARM)]
    mp = dict(zip(smap.key, smap.h))
    if segs is None:                                   # profile-only reproduction: released window table
        tw = pd.read_parquet(V.MEVA_WINDOWS)
        tw["cell"] = tw.key.map(lambda k: mp.get(k, "U"))
        return tw
    rows = []
    for s in segs:
        g = s[g_key]
        wt = window_table(g)
        st, L, _ = events(g)
        nw = len(wt)
        E = np.bincount(np.arange(len(g)) // 300, weights=g, minlength=nw)
        nev = np.bincount(st // 300, minlength=nw) if len(st) else np.zeros(nw)
        rows.append(wt.assign(block_id=s["block_id"], cam_tag=s["cam_tag"], session=s["session"], seg=s["seg"],
                              E=E, n_ev=nev, definition=definition))
    tw = pd.concat(rows, ignore_index=True)
    tw["cell"] = tw.key.map(lambda k: mp.get(k, "U"))
    return tw


def pooling_dependent() -> set:
    om = pd.read_csv(RES / "omega_bprime_S8.csv")
    pdep = om[(om.level == "cell") & (om.pi_pool > 0.5)]
    return set(pdep.unit)


# ------------------------------------------------------------------ H1 (census, confirmatory-of-profile)
def h1_section():
    from meva_profile import bench_windows_a
    tw = pd.read_parquet(V.CACHE / "meva_windows.parquet")
    tw = tw[tw.in_S8]
    wB = pd.read_csv(RES / "w_B_strata.csv")
    wB = wB[(wB.rule == "v2") & (wB.arm == "main")]
    ba = bench_windows_a()
    mpa, _ = build_strata_cells(ba, "v2")
    fa = ba.assign(h=ba.key.map(mpa)).groupby("h").n.sum()
    wb = {"a": fa / fa.sum(), "bprime": wB[wB.definition == "bprime"].set_index("h").w_B,
          "c": wB[wB.definition == "bprime"].set_index("h").w_B}
    e1b = pd.read_csv(RES / "E1_profile_benchmark.csv")
    e1b = e1b[e1b.dataset != "VIRAT-P"]
    e1m = pd.read_csv(RES / "E1_profile_meva.csv")
    rng = np.random.default_rng(SEED)
    out = {}
    for d in ("a", "bprime", "c"):
        x = tw[tw.definition == d]
        f = x.groupby("h_v2_main").n.sum()
        w = f / f.sum()
        idx = wb[d].index.union(w.index)
        tv0 = tv(wb[d].reindex(idx, fill_value=0).values, w.reindex(idx, fill_value=0).values)
        fb = x.groupby(["cam_tag", "session", "h_v2_main"]).n.sum()
        cams, sss = sorted(x.cam_tag.unique()), sorted(x.session.unique())
        pairs = set(fb.index.droplevel(2))
        bt = []
        for _ in range(B):
            cs, ss = rng.choice(cams, len(cams)), rng.choice(sss, len(sss))
            parts = [fb.loc[(c, s)] for c in cs for s in ss if (c, s) in pairs]
            if not parts:
                continue
            ff = pd.concat(parts).groupby(level=0).sum()
            ww = ff / ff.sum()
            ii = wb[d].index.union(ww.index)
            bt.append(tv(wb[d].reindex(ii, fill_value=0).values, ww.reindex(ii, fill_value=0).values))
        bt = np.array(bt)
        col = {"a": "a", "bprime": "bp", "c": "bp"}[d]
        duty_v, nfr = e1b[f"duty_cycle_{col}"].values, e1b.n_frames.values
        dboot = []
        for _ in range(B):
            k = rng.integers(0, len(duty_v), len(duty_v))
            dboot.append(np.average(duty_v[k], weights=nfr[k]))
        mrow = e1m[(e1m.level == "all_S8") & (e1m.definition == d)].iloc[0]
        ci = np.percentile(dboot, [2.5, 97.5])
        out[d] = dict(TV=r6(tv0), TV_ci_twoway=r6(np.percentile(bt, [2.5, 97.5])),
                      p_TV_le_0_3=r6((np.sum(bt <= 0.3) + 1) / (len(bt) + 1)),
                      duty_MEVA_S8=r6(mrow.duty_cycle),
                      duty_benchmark=r6(np.average(duty_v, weights=nfr)), duty_benchmark_ci=r6(ci),
                      duty_MEVA_in_benchmark_ci=bool(ci[0] <= mrow.duty_cycle <= ci[1]),
                      bench_definition={"a": "(a)", "bprime": "(b')", "c": "(b') [cross-definition]"}[d],
                      meets_0_3=bool(tv0 >= 0.3))
    n_meet = sum(v["meets_0_3"] for v in out.values())
    n_ref = sum(v["duty_MEVA_in_benchmark_ci"] for v in out.values())
    verdict = "PASS" if (n_meet >= 2 and n_ref < 2) else "FAIL"
    return dict(status="confirmatory-of-profile (observed before pre-registration; not a blind prediction)",
                criterion="TV(w_B, w_T) >= 0.3 in >= 2 of 3 definitions {(a), (b'), (c)}; (c) census only, mapped on "
                          "the (b') benchmark cells. Refuted (PLAN §3) if the MEVA duty cycle lies inside the benchmark "
                          "95 % CI (cluster bootstrap over the 121 videos, frame-weighted) for >= 2 of 3 definitions. "
                          "H1 = PASS iff the TV criterion holds AND the refutation clause does not fire.",
                definitions=out, n_definitions_meeting=n_meet, n_definitions_refuting=n_ref,
                tv_component="PASS" if n_meet >= 2 else "FAIL",
                refutation_component="fires" if n_ref >= 2 else "does not fire",
                predicted=verdict,
                note="Both components are profile facts already seen. The TV part holds (3/3) while the duty-cycle "
                     "refutation clause fires for (a) and (b'): benchmark and MEVA differ in temporal COMPOSITION "
                     "(strata: idle-run lengths, censoring) far more than in duty cycle. Flag for Dat: the PLAN wording "
                     "makes H1 FAIL on duty cycle although TV >= 0.3 in every definition.",
                p_value_rule="p_H1 = second-smallest of the three one-sided two-way bootstrap p-values P(TV* <= 0.3) "
                             "(>= 2 of 3 must hold).",
                p_value=r6(sorted(v["p_TV_le_0_3"] for v in out.values())[1]))


# ------------------------------------------------------------------ H4 r*
def rstar_table(bench: Bench) -> list[dict]:
    Stot = bench.S.sum(0)
    wB = bench.w_B_supp
    rows = []
    for i in range(len(POLICIES)):
        for j in range(i + 1, len(POLICIES)):
            for met in MET:
                nu1, de1 = Bench.nu_de(Stot, i, met)
                nu2, de2 = Bench.nu_de(Stot, j, met)
                M1, M2 = (wB @ nu1) / (wB @ de1), (wB @ nu2) / (wB @ de2)
                info = {}
                if met in ("processed_frac", "false_open_per_frame"):
                    r, _ = fr.r_star_linear(nu1 - nu2, wB)
                    method, rl = "linear exact", r
                elif met in ("frame_recall", "event_recall"):
                    assert np.allclose(de1, de2)
                    r, _ = fr.r_star_ratio_shared_den(nu1, nu2, de1, wB)
                    method, rl = "shared-denominator exact", r
                else:
                    r, _, info = fr.r_star_ratio_general(nu1, de1, nu2, de2, wB)
                    # dense refinement of the outer 1-D search between / around the two benchmark ratios: every rho(t)
                    # is an exact LP and attained, so the minimum stays a valid (attained) value
                    lo, hi = min(M1, M2), max(M1, M2)
                    pad = max(hi - lo, 1e-3)
                    a1, b1, a2, b2 = (nu1, de1, nu2, de2) if M1 >= M2 else (nu2, de2, nu1, de1)   # _rho: policy 1 ahead
                    for t in np.linspace(lo - 5 * pad, hi + 5 * pad, 801):
                        r = min(r, fr._rho(t, a1, b1, a2, b2, wB, None)[0])
                    assert r >= fr.r_star_ratio_lower(nu1, de1, nu2, de2, wB) - 1e-6
                    rl = fr.r_star_ratio_lower(nu1, de1, nu2, de2, wB)
                    method = "general LP (attained upper value; certified lower bound alongside)"
                # P1 vs P1: ordering is K-monotone in every window (except precision) -> cannot flip; excluded from H4
                in_h4 = not (POLICIES[i][0] == "P1" and POLICIES[j][0] == "P1" and met != "precision")
                rows.append(dict(in_H4=in_h4, pair=f"{PNAME[POLICIES[i]]} vs {PNAME[POLICIES[j]]}", p1=PNAME[POLICIES[i]],
                                 p2=PNAME[POLICIES[j]], metric=met, M1_B=r6(M1), M2_B=r6(M2),
                                 delta_B=r6(M1 - M2), sign_B=int(np.sign(M1 - M2)), r_star=r6(r) if np.isfinite(r) else "inf",
                                 r_star_lower=r6(rl) if np.isfinite(rl) else "inf", method=method))
    return rows


# ------------------------------------------------------------------ main
def main():
    bench = bench_table()
    segs = load_segments()
    tw = target_windows(segs)
    assert set(tw.definition) == {DEF}, "H2-H4 predictions must use the (b') profile only"
    pdep = pooling_dependent()
    assert pdep == {"G331_2018-03-07_165000"}, pdep
    tgt = Target(tw, bench.cells)
    U = units(tw, pdep)
    Stot = bench.S.sum(0)

    cells_out = []
    for u in U:
        btc = bootstrap_unit(bench, tgt, u, "cond")
        bt2 = bootstrap_unit(bench, tgt, u, "twoway")
        btb = bootstrap_unit(bench, tgt, u, "block")
        F = tgt.wvec(u["blocks"])
        Fs = F[:-1]
        tw_all = tw[tw.block_id.isin(u["blocks_all"])]
        F_all = tgt.wvec(u["blocks_all"])
        om_all = F_all[-1] / F_all.sum()
        om_tr = F[-1] / F.sum()
        contains_pd = any(b in pdep for b in u["blocks_all"])
        for p, (pol, K) in enumerate(POLICIES):
            for j, met in enumerate(MET):
                nu, de = Bench.nu_de(Stot, p, met)
                ok = Fs > 0
                M = transport_point(nu[ok], de[ok], Fs[ok])
                naive = bench.naive(p, met)
                vb = float(np.nanvar(btc[:, p, j], ddof=1)) if np.isfinite(btc[:, p, j]).sum() > 1 else float("nan")
                vo = v_obs(bench, tgt, u, p, met, M, de)
                half = TCRIT * np.sqrt(vb + (vo if np.isfinite(vo) else 0.0))
                L, Ub, om, omE = bounds_unit(nu, de, tw_all, bench.cells, pol, K, met)
                # simple P-2 form uses the plug-in on the supported part of ALL frames of the unit
                Fa = F_all[:-1]
                M_supp_all = transport_point(nu[Fa > 0], de[Fa > 0], Fa[Fa > 0])
                sL, sU = simple_bounds(M_supp_all, om_all, met)
                key = (pol, met)
                name = PNAME[(pol, K)]
                if not np.isfinite(M):
                    role, expect = "undefined (no transported denominator mass in the transportable part)", "n/a"
                elif u["level"] == "pooling_dependent_block":
                    role, expect = "pooling_dependent (reported separately, D3.2)", "n/a"
                elif u["level"] == "all":
                    role, expect = "summary (H2, not in H3 denominator)", "n/a"
                elif key in EXPECTED_FAIL:
                    role, expect = "expected_FAIL (D3.5, reported separately)", "FAIL"
                elif (name, met) in DETERMINISTIC:
                    role, expect = "deterministic_by_construction (reported separately)", "n/a"
                else:
                    role, expect = "H3_denominator", "PASS"
                c = dict(level=u["level"], unit=u["unit"], policy=name, metric=met,
                         transportable_blocks=u["blocks"], all_blocks=u["blocks_all"],
                         frames_transportable=int(Fs.sum()), frames_all=int(F_all.sum()),
                         omega_v2_unit_all=r6(om_all), omega_v2_unit_transportable_blocks=r6(om_tr),
                         contains_pooling_dependent_block=contains_pd,
                         naive=r6(naive), transport=r6(M), predicted_shift=r6(M - naive),
                         v_boot=r6(vb), v_obs=r6(vo), PI=r6([M - half, M + half]), PI_half=r6(half),
                         CI_cond=r6(np.nanpercentile(btc[:, p, j], [2.5, 97.5])) if np.isfinite(M) else None,
                         CI_twoway=r6(np.nanpercentile(bt2[:, p, j], [2.5, 97.5])) if np.isfinite(M) else None,
                         CI_block=r6(np.nanpercentile(btb[:, p, j], [2.5, 97.5])) if np.isfinite(M) else None,
                         n_boot_valid=dict(cond=int(np.isfinite(btc[:, p, j]).sum()),
                                           twoway=int(np.isfinite(bt2[:, p, j]).sum()),
                                           block=int(np.isfinite(btb[:, p, j]).sum())),
                         manski_full=r6([L, Ub]), manski_width=r6(Ub - L), omega_E=r6(omE),
                         manski_simple_P2=r6([sL, sU]), role=role, H3_expected=expect)
                if met == "false_open_per_frame":
                    s = FPS_T * 3600
                    c.update(per_hour=dict(naive=r6(naive * s), transport=r6(M * s), PI=r6([(M - half) * s, (M + half) * s]),
                                           manski_full=r6([L * s, Ub * s]), manski_simple_P2=r6([sL * s, sU * s])))
                cells_out.append(c)
        print(f"{u['level']:>24s} {u['unit']:<24s} frames {int(Fs.sum()):>8d}  omega_all {om_all:.3f}")

    rs = rstar_table(bench)
    h1 = h1_section()
    n_den = sum(c["role"] == "H3_denominator" for c in cells_out)
    seen = (REPO / "docs" / "PREREG_DECISIONS.md").read_text(encoding="utf-8")
    seen = seen[seen.index("**Dòng \"đã xem\" — cập nhật 2026-10-01"):].strip()
    inputs = ["results/benchmark_units.parquet", "results/operating_points.json", "results/strata_map_benchmark.csv",
              "results/w_B_strata.csv", "results/m_B_strata.csv", "results/omega_bprime_S8.csv",
              "results/meva_wT_strata.csv", "results/E1_profile_meva.csv", "results/E1_profile_benchmark.csv",
              "results/meva_S8_blocks.csv", "results/meva_manifest.csv", "docs/STRATA_SPEC.md",
              "docs/PREREG_DECISIONS.md", "docs/MEVA_detector_plan.md", "code/transport.py", "code/prereg_build.py",
              "code/meva_frames.py", "code/policies.py", "code/strata.py", "code/flip_radius.py", "code/cue.py",
              "code/meva_outcomes.py", "cache/meva_windows.parquet", "cache/meva_s8_frames.pkl"]
    pre = dict(
        title=("PREREG_MEVA_corrected — Đề tài 09, AMENDMENT 01 (LASIESTA temporal order; NOT blind: original outcomes seen)"
               if V.CORRECTED else "PREREG_MEVA — Đề tài 09, CP1 pre-registration (MEVA S8, definition (b'), STRATA_SPEC v2)"),
        variant=V.VARIANT,
        created=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        repo_head_at_build=git("rev-parse", "HEAD"),
        input_sha256={V.rel(path_of(p)): sha(path_of(p)) for p in inputs},
        decisions="docs/PREREG_DECISIONS.md: Q-A..Q-E, 27-09 round 2, P-1/P-2 (30-09), D3.1-D3.7 (01-10)",
        design=dict(
            target="MEVA S8 (plan 1101322c...; 10 blocks, 5 cameras x 3 sessions, 1,062,229 frames), detector route "
                   "cuda_fp32_kaggle (gate tier event)",
            event_definition="(b') on BOTH sides: person conf >= 0.7, idle gaps <= round(0.2 s x fps) merged inside a "
                             "segment (MEVA 30 fps: k = 6)",
            strata="STRATA_SPEC v2 (pooled R -> D -> C), cells from results/strata_map_benchmark.csv (bprime, v2, main)",
            policies={PNAME[p]: (f"periodic K = {p[1]}, averaged over phases" if p[0] == "P1" else p[0]) for p in POLICIES},
            operating_points="results/operating_points.json (frozen; NOT re-selected)",
            metrics={m: METRIC_LABEL[m] for m in MET},
            false_open_per_hour="false-open / frame x 30 fps x 3600 (MEVA header fps)",
            units="camera (5) and session (3) of S8; 'all' = summary; the pooling-dependent block is its own unit",
            transportable_part="frames in cells != U (rule v2) of non-pooling-dependent blocks",
            naive="pooled ratio over all windows of the 121 main benchmark videos (incl. benchmark U windows)",
            transport="sum_h w_T(h) nu_B(h) / sum_h w_T(h) delta_B(h) over supported cells (Theorem 1(a))",
            PI=f"for the OBSERVED value M_obs: transport +- t(0.975, df={DF}) = {TCRIT:.4f} x sqrt(v_boot + v_obs); "
               "v_boot = variance of the benchmark-video cluster bootstrap with w_T fixed at the observed profile; "
               "v_obs = sum_h nwin_T(h) Var_B(N - M Q | h) / Dhat^2 (Dhat transported) -- the form validated in "
               f"synthetic_check.py (COND), with 1.96 replaced by t_2 (Q-D/D3.6). B = {B}, seed {SEED}",
            CI_secondary="percentile CI with the unit's blocks resampled jointly with benchmark videos (block bootstrap, "
                         "D3.6); also CI_twoway for M_T(w_T) (camera x session two-way for 'all'; for camera / session "
                         "units the one available dimension = the unit's blocks) and CI_cond (benchmark only)",
            bounds="Theorem 1(b) closed forms on ALL frames of the unit (incl. U and the pooling-dependent block): "
                   "supported cells by plug-in, U windows by per-window logical boxes (processed >= floor(n/16)/n for "
                   "P2-P4, exact phase-averaged count for P1; false-open <= ceil(n/2)/n, <= processed for P1; recall "
                   "numerator in [0, e]; precision closed form). Also the PREREG_DECISIONS P-2 form "
                   "[(1-w) M, (1-w) M + w] (width = omega).",
            observed="M_obs = sum N / sum Q over the unit's transportable windows (computed only after unlock)",
            E4_check="the observed value on ALL frames of a unit must lie inside the ORACLE Theorem 1(b) bounds "
                     "(observed supported part + U boxes) -- a violation is a bug; containment of the observed value in "
                     "the pre-registered plug-in bounds is a reported result (it can fail through A1 only)",
            H3_cell_rule="cell PASS iff |transport - M_obs| <= 0.5 |naive - M_obs|; cell undefined if the observed "
                         "denominator is 0",
            pooling_dependent_blocks=sorted(pdep),
            df=DF, t_crit=TCRIT, B=B, seed=SEED),
        H1=h1,
        H2=dict(criterion="At the 'all' unit (transportable part of S8): for policy x metric with metric in "
                          "{processed fraction, event recall, false-open/frame}, deterministic-by-construction cells "
                          "(P1 x processed) excluded: (A) |naive - M_obs| > PI half-width and (B) composition share = "
                          "(transport - naive) / (M_obs - naive) >= 0.5 (Kitagawa-Oaxaca, ratio form, benchmark "
                          "reference). H2 PASS if, for >= 1 of the 3 metrics, (A) and (B) hold for a majority of its "
                          "eligible policy cells (processed: P2-P4, need 2 of 3; event recall and false-open: 7 cells, "
                          "need 4 of 7).",
                thresholds=dict(composition_share=0.5, majority={"processed_frac": 2, "event_recall": 4,
                                                                 "false_open_per_frame": 4},
                                metrics=list(H2_METRICS)),
                p_value_rule="per policy x metric: two-sided t test of bias / (PI_half / t_crit), df = 2; per metric "
                             "p_m = k-th smallest p (k = required majority); p_H2 = min(1, 3 x min_m p_m) (Bonferroni "
                             "over the 3 metrics)",
                predicted_direction="not predicted numerically beyond the per-cell naive / transport / PI below"),
        H3=dict(criterion="Binding (D3.3): |transport - M_obs| <= 0.5 |naive - M_obs| in >= 80 % of the H3 denominator "
                          "cells. Coverage of the PI is reported only.",
                denominator_rule="camera and session units x 7 policies x 5 metrics, transportable part only; EXCLUDED "
                                 "and listed: cells whose transport is undefined from the profile (no transported "
                                 "denominator mass, e.g. recall where every supported window is idle); also EXCLUDED "
                                 "and reported separately: (i) the pooling-dependent block (D3.2) [its frames are removed "
                                 "from camera P-4 and session 03-07]; (ii) expected-FAIL cells P2/P3 x frame recall (D3.5); "
                                 "(iii) deterministic-by-construction cells P1(K) x {processed fraction, frame recall} "
                                 "(phase-averaged P1 processes each frame with probability exactly 1/K; naive, transport "
                                 "and observed all equal 1/K up to segment-start frames, so the error ratio is a "
                                 "clip-boundary artefact ~1e-4) -- added by Claude at CP1, flagged for Dat's review; H3 is "
                                 "also reported with (iii) included as a sensitivity row; (iv) any cell whose observed "
                                 "denominator is 0 (undefined), listed.",
                n_denominator=n_den, threshold_pass_rate=0.8, threshold_error_ratio=0.5,
                p_value_rule="exact one-sided binomial P(X >= k | n, 0.5) on the denominator cells (descriptive: "
                             "cells are not independent)",
                expected_FAIL_cells=sorted(f"{p} x {m}" for p, m in EXPECTED_FAIL)),
        H4=dict(criterion="AUC(score -> observed flip) >= 0.75 over pair x metric x unit (camera and session units); "
                          "score = TV(w_B, w_T) / r*, with w_B, w_T renormalised over the supported cells (the ball "
                          "of flip_radius.py) and w_T from the unit's transportable part; r* = inf -> score 0. Flip "
                          "observed = sign of Delta on the unit's transportable part differs from sign_B (tie = flip).",
                r_star="LP flip_radius.py on the benchmark supported cells (NOT the closed form)",
                exclusion="P1 vs P1 pairs for processed, frame recall, event recall, false-open (K-monotone ordering in "
                          "every window: cannot flip; would only add guaranteed true negatives); kept for precision",
                n_pairs=len(rs) // len(MET), n_pair_metric=sum(r["in_H4"] for r in rs), n_units=8,
                n_evaluations=sum(r["in_H4"] for r in rs) * 8,
                ci_rule=f"bootstrap over pair x metric clusters, B = {B}, seed {SEED}; p = P(AUC* <= 0.5)",
                min_class_rule="if < 5 flips or < 5 non-flips, H4 = INCONCLUSIVE (reported as not PASS)",
                sensitivity="score = TV - r* (margin), reported only",
                r_star_table=rs),
        multiplicity="Holm over p_H1..p_H4 at alpha = 0.05. Binding verdicts are the criteria above on point values; "
                     "Holm-adjusted p is reported alongside (a criterion met without Holm significance is labelled "
                     "'PASS (not significant after Holm)').",
        E6=dict(note="sensitivity only; does not change H1-H4. Each axis repeats E3 (transport, PI, error ratio) with "
                     "operating points and P4 coefficients frozen.",
                axes={"b_raw": "(b) raw instead of (b') on both sides (benchmark cells (b), v2)",
                      "rule_v1": "STRATA_SPEC v1 partition merge instead of v2 pooled",
                      "D_fine": "D classes <=2 / 3-5 / 6-10 / 11-30 / 31-100 / >100 frames (support rule unchanged, v2)",
                      "cue_baseline": "4th stratum component: tercile (benchmark cut points) of log10 of the causal cue "
                                      "median m_t at the window's first frame",
                      "window_1800": "u = 1800-frame windows (R classes unchanged)",
                      "norm_300": "cue normalisation median over 300 instead of 1800 previous frames (both sides)",
                      "fps_15": "benchmark videos of unknown fps assumed 15 fps (k_merge 3) instead of 25",
                      "fps_30": "benchmark videos of unknown fps assumed 30 fps (k_merge 6) instead of 25",
                      "route_laptop": "6 P-1 clips with the laptop ov_gpu_fp32x detections instead of Kaggle "
                                      "(g(b') already shown identical on 54,005 frames -> expected identical)"}),
        forbidden=["changing operating points (results/operating_points.json)", "changing the MEVA selection / S8",
                   "changing STRATA_SPEC (v2 main, v1 sensitivity)", "adding or removing cells after unlock",
                   "using (c) or seen (b') numbers to calibrate H2-H4", "extending MEVA based on results (D3.6)"],
        amendments=("prereg/AMENDMENT_01.md (sha256 in prereg/AMENDMENT_01.md.sha256): LASIESTA re-indexed to temporal order; "
                    "everything else identical; original = prereg/PREREG_MEVA.json 209f2e22..." if V.CORRECTED else
                    "none; any change after hashing goes to prereg/AMENDMENT_nn.md (reason, time, what outcome was seen), "
                   "hashed and committed; results reported before and after"),
        seen_before_prereg=(seen + ("\n\nAMENDMENT 01: the original-set MEVA outcomes (H1-H4 FAIL, H3 81/192, "
                                    "H4 AUC 0.739) were seen before these corrected predictions were built." if V.CORRECTED else "")),
        cells=cells_out)
    PRE.mkdir(exist_ok=True)
    V.PRE_JSON.write_text(json.dumps(pre, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                          newline="\n")
    write_md(pre)
    print(f"cells {len(cells_out)}, H3 denominator {n_den}, r* rows {len(rs)}; H1 {h1['predicted']} "
          f"({h1['n_definitions_meeting']}/3)")


def fmt(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def iv(x, d=4):
    return "—" if x is None or x[0] is None else f"[{fmt(x[0], d)}; {fmt(x[1], d)}]"


def write_md(pre: dict):
    L = []
    A = L.append
    A(f"# {pre['title']}\n")
    A(f"Tạo lúc **{pre['created']}** · HEAD lúc dựng `{pre['repo_head_at_build'][:10]}` · bản máy đọc: "
      f"`{V.PRE_REL}` (file này là bản người đọc, sinh tự động từ cùng dữ liệu). "
      "Băm SHA-256 của cả hai file nằm ở `prereg/*.sha256`.\n")
    A("**Luật:** sau khi băm + commit, KHÔNG sửa file này. Mọi thay đổi bắt buộc → `prereg/AMENDMENT_nn.md` (lý do, "
      "thời điểm, đã thấy kết cục nào chưa), băm, commit; báo cả kết quả trước và sau.\n")
    d = pre["design"]
    A("## 1. Thiết kế (đối tượng của pre-registration)\n")
    for k in ("target", "event_definition", "strata", "operating_points", "units", "transportable_part", "naive",
              "transport", "PI", "CI_secondary", "bounds", "observed", "false_open_per_hour"):
        A(f"- **{k}**: {d[k]}")
    A(f"- **chính sách**: " + "; ".join(f"{k} = {v}" for k, v in d["policies"].items()))
    A(f"- **metric**: " + "; ".join(d["metrics"].values()))
    A(f"- **block pooling-dependent**: {', '.join(d['pooling_dependent_blocks'])}\n")
    h1 = pre["H1"]
    A("## 2. H1 — " + h1["status"] + "\n")
    A(h1["criterion"] + "\n")
    A("| định nghĩa | TV | CI hai chiều | p(TV* ≤ 0,3) | duty MEVA S8 | duty benchmark [CI 95 %] | trong CI? |")
    A("|---|---|---|---|---|---|---|")
    for k, v in h1["definitions"].items():
        A(f"| {k} → bench {v['bench_definition']} | {fmt(v['TV'])} | [{fmt(v['TV_ci_twoway'][0])}; {fmt(v['TV_ci_twoway'][1])}] "
          f"| {fmt(v['p_TV_le_0_3'], 4)} | {fmt(v['duty_MEVA_S8'])} | {fmt(v['duty_benchmark'])} "
          f"[{fmt(v['duty_benchmark_ci'][0])}; {fmt(v['duty_benchmark_ci'][1])}] | {'có' if v['duty_MEVA_in_benchmark_ci'] else 'không'} |")
    A(f"\nThành phần TV: **{h1['tv_component']}** ({h1['n_definitions_meeting']}/3) · điều khoản bác bỏ (duty trong CI): "
      f"**{h1['refutation_component']}** ({h1['n_definitions_refuting']}/3) → H1 (đã quan sát được): **{h1['predicted']}**; "
      f"p_H1 (thành phần TV) = {fmt(h1['p_value'], 4)}.\n\n{h1['note']}\n")
    A("## 3. H2\n")
    A(pre["H2"]["criterion"] + "\n\n" + "p: " + pre["H2"]["p_value_rule"] + "\n")
    A("## 4. H3\n")
    h3 = pre["H3"]
    A(h3["criterion"] + "\n\n" + "Mẫu số: " + h3["denominator_rule"] + "\n")
    A(f"**Số ô mẫu số H3 = {h3['n_denominator']}** (ngưỡng tỉ lệ đạt 0,80; tỉ số sai số 0,5). p: {h3['p_value_rule']}.\n")
    A("## 5. H4\n")
    A(pre["H4"]["criterion"] + "\n")
    A(f"r\\*: {pre['H4']['r_star']}. Số cặp chính sách = {pre['H4']['n_pairs']}, cặp × metric = {pre['H4']['n_pair_metric']}, "
      f"× {pre['H4']['n_units']} đơn vị = {pre['H4']['n_evaluations']} lần đánh giá. {pre['H4']['min_class_rule']}.\n")
    A(f"Loại khỏi H4: {pre['H4']['exclusion']}.\n")
    A("| cặp | metric | Δ_B | r\\* | r\\* cận dưới | vào H4 |")
    A("|---|---|---|---|---|---|")
    for r in pre["H4"]["r_star_table"]:
        A(f"| {r['pair']} | {r['metric']} | {fmt(r['delta_B'], 4)} | {fmt(r['r_star'], 4)} | {fmt(r['r_star_lower'], 4)} "
          f"| {'có' if r['in_H4'] else 'không'} |")
    A("\n## 6. Bội số\n\n" + pre["multiplicity"] + "\n")
    A("## 7. Dự đoán theo ô (ước lượng ngây thơ, transport, PI 95 % t₂, CI phụ, cận Manski toàn MEVA)\n")
    A("Cột: ngây thơ · transport · PI · CI block · cận ĐL1(b) toàn đơn vị · cận P-2 · ω toàn đơn vị · vai trò · H3 dự kiến.\n")
    cells = pd.DataFrame(pre["cells"])
    for (lvl, unit), x in cells.groupby(["level", "unit"], sort=False):
        A(f"### {lvl} — {unit} (khung transportable {x.frames_transportable.iloc[0]:,}; ω toàn đơn vị {fmt(x.omega_v2_unit_all.iloc[0])})\n")
        A("| chính sách | metric | ngây thơ | transport | PI | CI block | cận ĐL1(b) | cận P-2 | vai trò | H3 dk |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for c in x.itertuples(index=False):
            A(f"| {c.policy} | {c.metric} | {fmt(c.naive, 4)} | {fmt(c.transport, 4)} | {iv(c.PI)} | {iv(c.CI_block)} "
              f"| {iv(c.manski_full)} | {iv(c.manski_simple_P2)} | {c.role.split(' ')[0]} | {c.H3_expected} |")
        A("")
    A("## 8. E6 (độ nhạy, không vào H1–H4)\n")
    A(pre["E6"]["note"] + "\n")
    for k, v in pre["E6"]["axes"].items():
        A(f"- `{k}`: {v}")
    A("\n## 9. Cấm\n")
    for f in pre["forbidden"]:
        A(f"- {f}")
    A("\n## 10. Những gì đã xem trước pre-registration (chép từ PREREG_DECISIONS)\n")
    A(pre["seen_before_prereg"] + "\n")
    A("## 11. SHA-256 đầu vào\n")
    for k, v in pre["input_sha256"].items():
        A(f"- `{k}` `{v}`")
    V.PRE_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
