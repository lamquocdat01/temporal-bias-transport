"""results/numbers.json + result tables -> paper/numbers.tex (macros) and paper/tab_*.tex (tables).

C6: every number printed in paper/main.tex and paper/supp.tex comes from here; nothing is typed by hand.
Adds section 'paper' (all macro values) to results/numbers.json. Usage: python code/numbers_tex.py
"""
from __future__ import annotations
import variant as V
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
PAP = Path(os.environ.get("NINE_PAPER_OUT") or REPO / "paper")
POLS = ["P1_K2", "P1_K5", "P1_K10", "P1_K30", "P2", "P3", "P4"]
PLAB = {"P1_K2": "P1, $K{=}2$", "P1_K5": "P1, $K{=}5$", "P1_K10": "P1, $K{=}10$", "P1_K30": "P1, $K{=}30$",
        "P2": "P2", "P3": "P3", "P4": "P4"}
MLAB = {"processed_frac": "processed fraction", "frame_recall": "frame recall", "event_recall": "event recall",
        "false_open_per_frame": "false opens / frame", "precision": "precision"}


def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def i(x):
    return f"{int(x):,}".replace(",", "{,}")


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


ORIG_SHA = (V.PRE_DIR / "PREREG_MEVA.json.sha256").read_text().split()[0]
ORIG_KEYS = ("HoneVerdict", "HtwoVerdict", "HthreeVerdict", "HfourVerdict", "HonepHolm", "HtwopHolm", "HthreepHolm",
             "HfourpHolm", "HthreeN", "HthreePass", "HthreeRate", "HthreeMedRatio", "HthreePIcov", "HthreeSensRate",
             "PoneRateLo", "PoneRateHi", "StatefulRateLo", "StatefulRateHi", "PoneMedRatio", "StatefulMedRatio",
             "HfourAUC", "HfourAUClo", "HfourAUChi", "HfourFlips", "HfourN", "HfourAUCmargin", "HtwoCompMin",
             "HtwoCompMax", "EfourOracleViol", "EfourPluginCont", "EfourPtwoCont", "EsixMinRate", "EsixMaxRate",
             "TauTwo", "TauThree", "TauFour", "OmegaVtwo", "OmegaVone", "TVa", "TVbp", "TVc", "BenchEventsBp",
             "BenchIdleMedian", "BenchIdleMax", "CensusRlongB", "CensusCensB", "SevExceedT", "SevExceedM",
             "SevCuePtwo", "SevBackPtwo", "SevGapPtwo", "SevTVthreeh", "SevTVeighteenh", "SevTVeighteenk")

E6LAB = {"main": "main (pre-registered)", "b_raw": "raw (b), no gap closing", "rule_v1": "pooling rule v1",
         "D_fine": "finer duration classes", "cue_baseline": "+ cue-baseline tercile", "window_1800": "1800-frame windows",
         "norm_300": "normaliser memory 300", "fps_15": "unknown fps $=15$", "fps_30": "unknown fps $=30$",
         "route_laptop": "detector route (6 clips)"}



def amendment_counts() -> dict:
    """Event counts of definition (a) before / after AMENDMENT 01; computed from the corpus when it is available,
    otherwise read from results/corrected/amendment_counts.json (profile-only reproduction)."""
    out = V.RES / "amendment_counts.json"
    if V.PROC.exists():
        ev = pd.read_csv(V.PROC / "events_durations.csv")
        evc = V.events_durations()
        cnt = dict(LASIESTA_events_a_orig=int((ev.dataset == "LASIESTA").sum()),
                   LASIESTA_events_a_corrected=int((evc.dataset == "LASIESTA").sum()),
                   corpus_events_a_orig=int((ev.dataset != "VIRAT").sum()),
                   corpus_events_a_corrected=int((evc.dataset != "VIRAT").sum()))
        out.write_text(json.dumps(cnt, indent=1))
        return cnt
    return json.loads(out.read_text())


def main():
    nj = json.loads((V.NUMBERS).read_text())
    d3, e7 = nj["dot3"], nj["e7"]
    pre = json.loads(V.PRE_JSON.read_text(encoding="utf-8"))
    vf = json.loads((RES / "viratp_flicker.json").read_text())
    op = json.loads((RES / "operating_points.json").read_text())
    e1b = pd.read_csv(RES / "E1_profile_benchmark.csv")
    e1m = pd.read_csv(RES / "E1_profile_meva.csv")
    e3 = pd.read_csv(RES / "E3_transport_cells.csv")
    e3s = pd.read_csv(RES / "E3_summary.csv").set_index("group")
    e4 = pd.read_csv(RES / "E4_bounds.csv")
    e5 = json.loads((RES / "E5_auc.json").read_text())
    e6 = pd.read_csv(RES / "E6_summary.csv").set_index("axis")
    om = pd.read_csv(RES / "omega_bprime_S8.csv")
    gate = json.loads((RES / "kaggle_gate_b01.json").read_text())
    s8 = pd.read_csv(RES / "meva_S8_blocks.csv")
    bm = e1b[e1b.dataset != "VIRAT-P"]
    ms8 = e1m[(e1m.level == "block") & e1m.in_S8 & (e1m.definition == "bprime")]
    allS8 = e1m[e1m.level == "all_S8"].set_index("definition")
    h1d = pre["H1"]["definitions"]
    den = e3[(e3.role == "H3_denominator") & e3.cell_PASS.notna()]
    pol_rate = den.groupby("policy").cell_PASS.mean()
    p1r = pol_rate[[p for p in POLS if p.startswith("P1")]]
    sr = pol_rate[["P2", "P3", "P4"]]
    m = {
        # event definition evidence (VIRAT-P)
        "VPshortShare": f"{100 * vf['share_of_events_len_1_3']:.1f}",
        "VPshortNoGTtime": f"{100 * vf['short_temporal']['frac_unsupported']:.1f}",
        "VPshortNoGTspace": f"{100 * vf['short_spatial']['frac_unsupported']:.1f}",
        "VPlongNoGTtime": f"{100 * vf['long_temporal']['frac_unsupported']:.1f}",
        "VPlongNoGTspace": f"{100 * vf['long_spatial']['frac_unsupported']:.1f}",
        "VPiou": f"{vf['iou_threshold']:.1f}", "VPeventsRaw": i(vf["events_total"]), "VPeventsMerged": i(306),
        "VPvideosGT": str(vf["videos_with_gt"]),
        # benchmark
        "BenchVideos": str(len(bm)), "BenchCDnet": str((bm.dataset == "CDnet2014").sum()),
        "BenchLASIESTA": str((bm.dataset == "LASIESTA").sum()), "BenchBMC": str((bm.dataset == "BMC").sum()),
        "BenchFrames": i(bm.n_frames.sum()), "BenchEventsBp": i(bm.n_events_bp.sum()),
        "BenchMeanFrames": i(round(bm.n_frames.mean())),
        "BenchIdleMedian": i(bm.longest_idle_frames_bp.median()), "BenchIdleMax": i(bm.longest_idle_frames_bp.max()),
        "VPFrames": i(e1b[e1b.dataset == "VIRAT-P"].n_frames.sum()), "VPVideos": str((e1b.dataset == "VIRAT-P").sum()),
        "BenchUshare": f(float(e6.loc["main", "bench_frac_U"])), "BenchCells": str(int(e6.loc["main", "n_cells_bench"])),
        "PfourFitVideos": str(len(op["P4"]["fit_videos"])),
        # MEVA
        "SEightBlocks": str(len(s8)), "SEightClips": str(int(s8.n_clips.sum())), "SEightHours": f"{s8.hours.sum():.2f}",
        "SEightFrames": i(allS8.loc["bprime", "n_frames"]), "SEightEventsBp": i(ms8.n_events.sum()),
        "SEightIdleMinBlock": i(ms8.longest_idle_frames[ms8.n_events > 5].min()),
        "SEightIdleMax": i(ms8.longest_idle_frames.max()), "SEightIdleMaxMin": f"{ms8.longest_idle_frames.max() / 30 / 60:.0f}",
        "SEightCams": "5", "SEightSessions": str(pre["design"]["df"] + 1), "SelHours": "20.83",
        "DutyMEVAa": f(h1d["a"]["duty_MEVA_S8"]), "DutyMEVAbp": f(h1d["bprime"]["duty_MEVA_S8"]),
        "DutyMEVAc": f(h1d["c"]["duty_MEVA_S8"]),
        "DutyBencha": f(h1d["a"]["duty_benchmark"]), "DutyBenchbp": f(h1d["bprime"]["duty_benchmark"]),
        "DutyBenchaLo": f(h1d["a"]["duty_benchmark_ci"][0]), "DutyBenchaHi": f(h1d["a"]["duty_benchmark_ci"][1]),
        "DutyBenchbpLo": f(h1d["bprime"]["duty_benchmark_ci"][0]), "DutyBenchbpHi": f(h1d["bprime"]["duty_benchmark_ci"][1]),
        "TVa": f(h1d["a"]["TV"]), "TVbp": f(h1d["bprime"]["TV"]), "TVc": f(h1d["c"]["TV"]),
        "TVmin": f(min(h1d[k]["TV"] for k in h1d), 2),
        "TVaLo": f(h1d["a"]["TV_ci_twoway"][0]), "TVaHi": f(h1d["a"]["TV_ci_twoway"][1]),
        "TVbpLo": f(h1d["bprime"]["TV_ci_twoway"][0]), "TVbpHi": f(h1d["bprime"]["TV_ci_twoway"][1]),
        "TVcLo": f(h1d["c"]["TV_ci_twoway"][0]), "TVcHi": f(h1d["c"]["TV_ci_twoway"][1]),
        "OmegaVtwo": f(om[om.level == "all"].omega_v2.iloc[0]), "OmegaVone": f(om[om.level == "all"].omega_v1.iloc[0]),
        "PiPoolAll": f(om[om.level == "all"].pi_pool.iloc[0]),
        "PiPoolPD": f(om[om.pooling_dependent].pi_pool.iloc[0]),
        "PiPoolNext": f(om[(om.level == "cell") & ~om.pooling_dependent].pi_pool.max()),
        "PDframes": i(om[om.pooling_dependent].n_frames.iloc[0]),
        "OmegaBlockMin": f(om[om.level == "cell"].omega_v2.min()), "OmegaBlockMax": f(om[om.level == "cell"].omega_v2.max()),
        # detector gate
        "GateMaxDconf": r"10^{%d}" % round(np.log10(gate["measured"]["max_dconf"])),
        "GateFrames": str(gate["measured"]["n_frames"]), "GateEvents": str(gate["measured"]["events_b_same_bounds"]),
        "SEightIdleMedian": i(ms8.longest_idle_frames.median()),
        "CensusRlongB": pct(nj["census"]["R_ge1800_benchmark"], 1), "CensusRlongM": pct(nj["census"]["R_ge1800_MEVA"]),
        "CensusCensB": pct(nj["census"]["R_censored_benchmark"]), "CensusCensM": pct(nj["census"]["R_censored_MEVA"], 1),
        "CensusIdleB": pct(nj["census"]["idle_benchmark"]), "CensusIdleM": pct(nj["census"]["idle_MEVA"]),
        # pre-registration
        "PIdf": str(pre["design"]["df"]), "PItcrit": f"{pre['design']['t_crit']:.3f}",
        # blind pre-registration (original) is always the cited registration; AMENDMENT 01 has its own macros
        "PreregSHAshort": ORIG_SHA[:12], "PreregSHA": ORIG_SHA, "Bboot": str(pre["design"]["B"]),
        "PreregCells": str(len(pre["cells"])), "PreregTime": "2026-10-01 15:36 (UTC+7)", "PreregCommit": "8a2902b",
        "TauTwo": f(op["P2"]["tau"]), "TauThree": f(op["P3"]["tau"]), "TauFour": f(op["P4"]["tau"]),
        "TargetRecall": f(op["target_event_recall"], 2),
        # H1-H4
        "HoneVerdict": d3["H1"]["verdict"], "HtwoVerdict": d3["H2"]["verdict"], "HthreeVerdict": d3["H3"]["verdict"],
        "HfourVerdict": d3["H4"]["verdict"],
        "HonepHolm": f(d3["H1"]["p_holm"]), "HtwopHolm": f(d3["H2"]["p_holm"]), "HthreepHolm": f(d3["H3"]["p_holm"]),
        "HfourpHolm": f(d3["H4"]["p_holm"]), "HtwoP": f(d3["H2"]["p"]),
        "HthreeN": str(d3["H3"]["n_evaluated"]), "HthreePass": str(d3["H3"]["n_pass"]),
        "HthreeRate": f(d3["H3"]["pass_rate"]), "HthreeMedRatio": f(d3["H3"]["median_error_ratio"]),
        "HthreePIcov": f(d3["H3"]["PI_coverage"]), "HthreeCIblockcov": f(d3["H3"]["CI_block_coverage"]),
        "HthreeSensRate": f(d3["H3"]["sensitivity_with_deterministic"]["pass_rate"]),
        "HthreeSensN": str(d3["H3"]["sensitivity_with_deterministic"]["n"]),
        "HthreeExpFailConfirmed": str(d3["H3"]["expected_FAIL"]["predicted_fail_confirmed"]),
        "HthreeExpFailN": str(d3["H3"]["expected_FAIL"]["n"]),
        "HthreePDpass": str(d3["H3"]["pooling_dependent"]["n_pass"]), "HthreePDn": str(d3["H3"]["pooling_dependent"]["n"]),
        "PoneRateLo": pct(p1r.min()), "PoneRateHi": pct(p1r.max()), "StatefulRateLo": pct(sr.min()),
        "StatefulRateHi": pct(sr.max()),
        "PoneMedRatio": f(float(den[den.policy.str.startswith("P1")].error_ratio.median()), 2),
        "StatefulMedRatio": f(float(den[~den.policy.str.startswith("P1")].error_ratio.median()), 2),
        "HfourAUC": f(d3["H4"]["AUC"]), "HfourAUClo": f(d3["H4"]["AUC_ci"][0]), "HfourAUChi": f(d3["H4"]["AUC_ci"][1]),
        "HfourFlips": str(d3["H4"]["n_flips"]), "HfourN": str(d3["H4"]["n_defined"]),
        "HfourNprereg": str(pre["H4"]["n_evaluations"]), "HfourAUCmargin": f(d3["H4"]["AUC_margin_sensitivity"]),
        "HtwoCompMax": f(float(e3[(e3.level == "all") & e3.policy.isin(["P2", "P3", "P4"]) &
                                  e3.metric.isin(["processed_frac", "event_recall", "false_open_per_frame"])
                                  ].composition_share.max()), 2),
        "HtwoCompMin": f(float(e3[(e3.level == "all") & e3.policy.isin(["P2", "P3", "P4"]) &
                                  e3.metric.isin(["processed_frac", "event_recall", "false_open_per_frame"])
                                  ].composition_share.min()), 2),
        "EfourOracleViol": str(d3["E4"]["oracle_violations"]), "EfourN": str(d3["E4"]["n"]),
        "EfourPluginCont": f(d3["E4"]["plugin_containment"]), "EfourPtwoCont": f(d3["E4"]["P2form_containment"]),
        "EsixMinRate": f(e6.pass_rate.min()), "EsixMaxRate": f(e6.pass_rate.max()),
        "EsixAxes": str(len(e6)),
        # E7
        "SevExceedT": f(e7["exceed_tau2_transported"]), "SevExceedM": f(e7["exceed_tau2_MEVA"]),
        "SevGapPtwo": f(e7["within_gap"]["P2"]), "SevGapPthree": f(e7["within_gap"]["P3"]),
        "SevGapPfour": f(e7["within_gap"]["P4"]),
        "SevCuePtwo": f(e7["within_gap_cue"]["P2"]), "SevBackPtwo": f(e7["within_gap_backstop"]["P2"]),
        "SevCuePfour": f(e7["within_gap_cue"]["P4"]), "SevBackPfour": f(e7["within_gap_backstop"]["P4"]),
        "SevHoldPthree": f(e7["within_gap_hold"]["P3"]),
        "SevTVthreeh": f(e7["memory_TV"]["300"]), "SevTVeighteenh": f(e7["memory_TV"]["1800"]),
        "SevTVeighteenk": f(e7["memory_TV"]["18000"]),
        "SevWthreeh": f(e7["memory_wasserstein"]["300"]), "SevWeighteenh": f(e7["memory_wasserstein"]["1800"]),
        "SevWeighteenk": f(e7["memory_wasserstein"]["18000"]),
        "SevCell": e7["figure_cell"].replace("|", r"\,|\,").replace("_", r"\_"),
    }
    syn = json.loads((RES / "synthetic_check.json").read_text())
    covs = [syn[g][c]["cov_TW_t_mstar"] for g in ("coverage_5x4", "coverage_5x3") for c in syn[g] if c != "P1_proc"]
    m.update(SynR=str(syn["meta"]["R"]), SynCovTmin=f(min(covs), 2),
             SynPtwoFrecBias=f(syn["coverage_5x4"]["P2_frec"]["a1_bias_mean"]),
             SynPtwoFrecCovPop=f(syn["coverage_5x4"]["P2_frec"]["cov_TW_t_Mpop"], 2),
             SynPtwoFrecPI=f(syn["coverage_5x4"]["P2_frec"]["cov_PI_obs"], 2),
             SynOmegaVtwo=f(syn["omega_5x4"]["mean"]), SynOmegaVone=f(syn["omega_5x4"]["mean_pipeline_rule"]),
             RouteFrames=i(54005))
    if V.CORRECTED:
        # AMENDMENT 01 (LASIESTA temporal order): registration of the corrected predictions + original-set values
        cnt = amendment_counts()
        m.update(AmendSHA=(V.PRE_DIR / "AMENDMENT_01.md.sha256").read_text().split()[0],
                 AmendPreregSHA=d3["prereg_sha256"], AmendPreregSHAshort=d3["prereg_sha256"][:12],
                 AmendCommit=d3["prereg_commit"], AmendTime="2026-10-03 16:07 (UTC+7)",
                 AmendLASevOld=i(cnt["LASIESTA_events_a_orig"]), AmendLASevNew=i(cnt["LASIESTA_events_a_corrected"]),
                 AmendCorpusOld=i(cnt["corpus_events_a_orig"]), AmendCorpusNew=i(cnt["corpus_events_a_corrected"]))
        orig = json.loads((V.RES_ORIG / "numbers.json").read_text())["paper"]
        for k in ORIG_KEYS:
            m["Orig" + k] = orig[k]
    nj["paper"] = m
    (V.NUMBERS).write_text(json.dumps(nj, indent=1, default=float))
    lines = ["% generated by code/numbers_tex.py from results/numbers.json -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAP / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    tables(d3, e3, e3s, e4, e6, om, h1d, pre, e5)
    print(f"{len(m)} macros -> paper/numbers.tex; tables -> paper/tab_*.tex")


def tables(d3, e3, e3s, e4, e6, om, h1d, pre, e5):
    W = lambda name, s: (PAP / name).write_text("% generated by code/numbers_tex.py -- do not edit\n" + s,  # noqa: E731
                                                encoding="utf-8", newline="\n")
    # ---- H1-H4
    def obs(d, e):
        h1, bm = d["H1"], d["H2"]["by_metric"]
        return {"H1": f"TV {f(h1['TV']['a'])}/{f(h1['TV']['bprime'])}/{f(h1['TV']['c'])}; duty in CI for (a),(b$'$)",
                "H2": f"both conditions met in {sum(x['n_A_and_B'] for x in bm.values())} cells (processed "
                      f"{bm['processed_frac']['n_A']}/{bm['processed_frac']['n_cells']} meet bias only)",
                "H3": f"{d['H3']['n_pass']}/{d['H3']['n_evaluated']} $=$ {f(d['H3']['pass_rate'])}",
                "H4": f"{f(e['AUC'])} [{f(e['AUC_ci'][0])}, {f(e['AUC_ci'][1])}]; {e['n_flips']}/{e['n_defined']} flips"}
    crit = {"H1": r"TV $\ge0.3$ in $\ge2/3$ definitions; refuted if MEVA duty cycle in benchmark 95\% CI",
            "H2": r"naive bias $>$ PI half-width and composition share $\ge0.5$ in a majority of policies, $\ge1$ metric",
            "H3": rf"$|e_{{\rm tr}}|\le0.5|e_{{\rm naive}}|$ in $\ge80\%$ of {d3['H3']['n_evaluated']} cells",
            "H4": r"AUC(TV$/r^*\to$ flip) $\ge0.75$"}
    oc = obs(d3, e5)
    if V.CORRECTED:                                    # AMENDMENT 01: corrected (main) next to the blind original run
        do = json.loads((V.RES_ORIG / "numbers.json").read_text())["dot3"]
        eo = json.loads((V.RES_ORIG / "E5_auc.json").read_text())
        oo = obs(do, eo)
        t = [r"\begin{tabular}{@{}lp{4.6cm}p{4.3cm}ccp{3.9cm}c@{}}", r"\toprule",
             r"& & \multicolumn{3}{c}{Corrected run (Amendment 01, main)} & \multicolumn{2}{c}{Original blind run}\\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-7}",
             r"& Pre-registered criterion & Observed & Verdict & $p_{\rm Holm}$ & Observed & Verdict\\", r"\midrule"]
        for h in ("H1", "H2", "H3", "H4"):
            t.append(rf"{h} & {crit[h]} & {oc[h]} & {d3[h]['verdict']} & {f(d3[h]['p_holm'])} & {oo[h]} & {do[h]['verdict']}\\")
    else:
        t = [r"\begin{tabular}{@{}lp{7.2cm}p{6.0cm}cc@{}}", r"\toprule",
             r"& Pre-registered criterion & Observed & Verdict & $p_{\rm Holm}$\\", r"\midrule"]
        for h in ("H1", "H2", "H3", "H4"):
            t.append(rf"{h} & {crit[h]} & {oc[h]} & {d3[h]['verdict']} & {f(d3[h]['p_holm'])}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    W("tab_hyp.tex", "\n".join(t) + "\n")
    if V.CORRECTED:
        # ---- supplement: E6 summary, original vs corrected (AMENDMENT 01)
        e6o = pd.read_csv(V.RES_ORIG / "E6_summary.csv").set_index("axis")
        t = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"& & \multicolumn{2}{c}{H3 rate} & \multicolumn{2}{c}{med.\ ratio}\\ \cmidrule(lr){3-4}\cmidrule(l){5-6}",
             r"Axis & cells & orig. & corr. & orig. & corr.\\", r"\midrule"]
        for ax, r in e6.iterrows():
            o = e6o.loc[ax]
            t.append(rf"{E6LAB.get(ax, ax)} & {int(r.n_denominator)} & {f(o.pass_rate)} & {f(r.pass_rate)} & "
                     rf"{f(o.median_error_ratio)} & {f(r.median_error_ratio)}\\")
        t += [r"\bottomrule", r"\end{tabular}"]
        W("tab_supp_e6.tex", "\n".join(t) + "\n")
    # ---- E3 by policy (H3 denominator)
    den = e3[(e3.role == "H3_denominator") & e3.cell_PASS.notna()]
    t = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Policy & cells & pass & rate & med.\ ratio & med.\ $|e_{\rm naive}|$ & PI cov.\\", r"\midrule"]
    for p in POLS:
        x = den[den.policy == p]
        t.append(rf"{PLAB[p]} & {len(x)} & {int(x.cell_PASS.sum())} & {f(x.cell_PASS.mean(), 2)} & "
                 rf"{f(x.error_ratio.median(), 2)} & {f(x.abs_err_naive.median())} & {f(x.PI_covers.mean(), 2)}\\")
        if p == "P1_K30":
            t.append(r"\addlinespace")
    t += [r"\midrule", rf"All & {len(den)} & {int(den.cell_PASS.sum())} & {f(den.cell_PASS.mean(), 2)} & "
          rf"{f(den.error_ratio.median(), 2)} & {f(den.abs_err_naive.median())} & {f(den.PI_covers.mean(), 2)}\\",
          r"\bottomrule", r"\end{tabular}"]
    W("tab_e3_policy.tex", "\n".join(t) + "\n")
    # ---- E3 by metric
    t = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Metric & cells & rate & med.\ ratio\\", r"\midrule"]
    for k, lab in MLAB.items():
        x = den[den.metric == k]
        t.append(rf"{lab} & {len(x)} & {f(x.cell_PASS.mean(), 2)} & {f(x.error_ratio.median(), 2)}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    W("tab_e3_metric.tex", "\n".join(t) + "\n")
    # ---- E6
    lab = E6LAB
    t = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Axis & cells & rate & med.\ ratio & $\omega$\\", r"\midrule"]
    for ax, r in e6.iterrows():
        t.append(rf"{lab[ax]} & {int(r.n_denominator)} & {f(r.pass_rate, 2)} & {f(r.median_error_ratio, 2)} & {f(r.omega_all, 2)}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    W("tab_e6.tex", "\n".join(t) + "\n")
    # ---- omega per camera x session
    c = om[om.level == "cell"].sort_values(["cam_tag", "session"])
    t = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Camera & Session & frames & $\omega_{v2}$ & $\omega_{v1}$ & $\pi_{\rm pool}$\\", r"\midrule"]
    for r in c.itertuples(index=False):
        mark = r"$^\dagger$" if r.pooling_dependent else ""
        t.append(rf"{r.cam_tag} & {r.session.replace('2018-', '').replace('_', ' ')} & {i(r.n_frames)} & {f(r.omega_v2)} & {f(r.omega_v1)} & {f(r.pi_pool)}{mark}\\")
    a = om[om.level == "all"].iloc[0]
    t += [r"\midrule", rf"\multicolumn{{2}}{{@{{}}l}}{{S8 total}} & {i(a.n_frames)} & {f(a.omega_v2)} & {f(a.omega_v1)} & {f(a.pi_pool)}\\",
          r"\bottomrule", r"\end{tabular}"]
    W("tab_omega.tex", "\n".join(t) + "\n")
    # ---- E4 containment by metric
    x = e4[e4.level.isin(["camera", "session"])]
    g = x.groupby("metric").agg(pl=("plugin_contains", "mean"), orc=("oracle_contains", "mean"),
                                p2=("P2form_contains", "mean"), w=("plugin_width", "median"), n=("metric", "size"))
    t = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Metric & cells & plug-in & oracle & width-$\omega$ form & med.\ width\\", r"\midrule"]
    for k in MLAB:
        r = g.loc[k]
        t.append(rf"{MLAB[k]} & {int(r.n)} & {f(r.pl, 2)} & {f(r.orc, 2)} & {f(r.p2, 2)} & {f(r.w)}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    W("tab_e4.tex", "\n".join(t) + "\n")
    # ---- supplement: all 350 cells (compact longtable)
    t = [r"\begin{longtable}{@{}llllrrrrrc@{}}", r"\toprule",
         r"Level & Unit & Policy & Metric & naive & transp. & obs. & ratio & PI cov. & role\\", r"\midrule\endhead"]
    mab = {"processed_frac": "proc", "frame_recall": "frec", "event_recall": "erec", "false_open_per_frame": "fo/fr",
           "precision": "prec"}
    rab = {"H3_denominator": "D", "expected_FAIL": "F", "deterministic_by_construction": "Z", "undefined": "U",
           "summary": "S", "pooling_dependent": "P"}
    for r in e3.itertuples(index=False):
        unit = str(r.unit).replace("2018-", "").replace("_", " ").replace("G331 03-07 165000", "P-4/03-07")
        t.append(rf"{r.level.split('_')[0]} & {unit} & {r.policy.replace('_', ' ')} & {mab[r.metric]} & {f(r.naive)} & "
                 rf"{f(r.transport)} & {f(r.M_obs)} & {f(r.error_ratio, 2)} & "
                 rf"{'' if pd.isna(r.PI_covers) else ('y' if r.PI_covers else 'n')} & {rab[r.role]}\\")
    t += [r"\bottomrule", r"\end{longtable}"]
    W("tab_supp_e3_all.tex", "\n".join(t) + "\n")
    # ---- supplement: E7 (exploratory) trigger decomposition and memory distances
    td = pd.read_csv(RES / "E7_trigger_decomposition.csv")
    t = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"Policy & reason & transported & footage & gap\\", r"\midrule"]
    for r in td.itertuples(index=False):
        if r.reason == "start":
            continue
        t.append(rf"{r.policy} & {r.reason} & {f(r.transported_rate)} & {f(r.MEVA_rate)} & {f(r.within_gap)}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    W("tab_supp_e7_trigger.tex", "\n".join(t) + "\n")
    me = pd.read_csv(RES / "E7_memory.csv")
    me = me[me.cell == "WEIGHTED"]
    t = [r"\begin{tabular}{@{}rrrr@{}}", r"\toprule", r"memory (frames) & Wasserstein & TV & $|\Delta\Pr(\tilde d>\tau_2)|$\\",
         r"\midrule"]
    for r in me.itertuples(index=False):
        t.append(rf"{i(r.memory)} & {f(r.wasserstein_log10)} & {f(r.TV_hist)} & {f(r.abs_diff_share_gt_tau2)}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    W("tab_supp_e7_memory.tex", "\n".join(t) + "\n")
    # ---- supplement: r* table (H4)
    rs = pd.DataFrame(pre["H4"]["r_star_table"])
    rs = rs[rs.in_H4]
    t = [r"\begin{longtable}{@{}llrrr@{}}", r"\toprule", r"Pair & Metric & $\Delta_B$ & $r^*$ & certified lower\\",
         r"\midrule\endhead"]
    for r in rs.itertuples(index=False):
        rv = r"$\infty$" if r.r_star == "inf" else f(float(r.r_star))
        rl = r"$\infty$" if r.r_star_lower == "inf" else f(float(r.r_star_lower))
        t.append(rf"{r.pair.replace('_', ' ')} & {mab[r.metric]} & {f(r.delta_B, 4)} & {rv} & {rl}\\")
    t += [r"\bottomrule", r"\end{longtable}"]
    W("tab_supp_rstar.tex", "\n".join(t) + "\n")


if __name__ == "__main__":
    main()
