"""AMENDMENT 01: original (blind pre-registered) vs corrected (LASIESTA temporal order) side by side.

Reads results/orig/ and results/corrected/ only. Writes results/corrected/AMENDMENT_01_compare.{json,csv} and
docs/AMENDMENT_01_RESULTS.md. Usage: python code/amendment_compare.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = {"orig": REPO / "results" / "orig", "corrected": REPO / "results" / "corrected"}
NJ = {"orig": "numbers.json", "corrected": "numbers_corrected.json"}
PRE = {"orig": "PREREG_MEVA.json", "corrected": "PREREG_MEVA_corrected.json"}


def f3(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}".replace(".", ",")


def main():
    N = {v: json.loads((R[v] / NJ[v]).read_text()) for v in R}
    D = {v: N[v]["dot3"] for v in R}
    sha = {v: (REPO / "prereg" / f"{PRE[v]}.sha256").read_text().split()[0] for v in R}
    rows = []
    for v in R:
        d = D[v]
        h1, h2, h3, h4 = d["H1"], d["H2"], d["H3"], d["H4"]
        rows += [
            dict(variant=v, H="H1", threshold="TV >= 0.3 in >= 2/3 defs; refuted if duty in CI for >= 2/3",
                 observed=f"TV a {h1['TV']['a']:.3f} / b' {h1['TV']['bprime']:.3f} / c {h1['TV']['c']:.3f}; duty in CI "
                          f"{sum(h1['duty_in_ci'].values())}/3", verdict=h1["verdict"], p=h1["p"], p_holm=h1["p_holm"]),
            dict(variant=v, H="H2", threshold=">= 1 metric with (A) and (B) in a majority of policy cells",
                 observed="; ".join(f"{m}: A&B {x['n_A_and_B']}/{x['n_cells']} (need {x['need']})"
                                    for m, x in h2["by_metric"].items()),
                 verdict=h2["verdict"], p=h2["p"], p_holm=h2["p_holm"]),
            dict(variant=v, H="H3", threshold="pass rate >= 0.80 of 192 cells",
                 observed=f"{h3['n_pass']}/{h3['n_evaluated']} = {h3['pass_rate']:.3f}; median error ratio "
                          f"{h3['median_error_ratio']:.3f}; PI coverage {h3['PI_coverage']:.3f}",
                 verdict=h3["verdict"], p=h3["p"], p_holm=h3["p_holm"]),
            dict(variant=v, H="H4", threshold="AUC(TV/r* -> flip) >= 0.75",
                 observed=f"AUC {h4['AUC']:.3f} [{h4['AUC_ci'][0]:.3f}, {h4['AUC_ci'][1]:.3f}]; flips {h4['n_flips']}/"
                          f"{h4['n_defined']}; margin-score AUC {h4['AUC_margin_sensitivity']:.3f}",
                 verdict=h4["verdict"], p=h4["p"], p_holm=h4["p_holm"])]
    H = pd.DataFrame(rows)
    H.to_csv(R["corrected"] / "AMENDMENT_01_compare.csv", index=False)

    # H3 by policy / metric
    pol = {}
    for v in R:
        s = pd.read_csv(R[v] / "E3_summary.csv").set_index("group")
        pol[v] = s
    groups = [g for g in pol["orig"].index if g.strip().startswith(("policy", "metric")) or g == "H3 denominator"]
    P = pd.DataFrame({f"{c}_{v}": pol[v].loc[groups, c] for v in R for c in ("n", "n_pass", "pass_rate", "median_error_ratio")})
    P.index = [g.strip() for g in groups]
    # E6, E7, omega, operating points
    e6 = {v: pd.read_csv(R[v] / "E6_summary.csv").set_index("axis")[["pass_rate", "median_error_ratio", "omega_all"]] for v in R}
    E6 = e6["orig"].join(e6["corrected"], lsuffix="_orig", rsuffix="_corrected")
    op = {v: json.loads((R[v] / "operating_points.json").read_text()) for v in R}
    om = {v: pd.read_csv(R[v] / "omega_bprime_S8.csv") for v in R}
    om_max = float((om["orig"].set_index(["level", "unit"])[["omega_v2", "omega_v1", "pi_pool"]]
                    - om["corrected"].set_index(["level", "unit"])[["omega_v2", "omega_v1", "pi_pool"]]).abs().max().max())
    pdep = {v: sorted(om[v][(om[v].level == "cell") & (om[v].pi_pool > 0.5)].unit) for v in R}
    e7 = {v: N[v]["e7"] for v in R}
    e4 = {v: D[v]["E4"] for v in R}
    out = dict(prereg_sha256=sha, H=rows, H3_by_group=P.reset_index().to_dict("records"),
               E6=E6.reset_index().to_dict("records"),
               operating_points={v: dict(tau2=op[v]["P2"]["tau"], tau3=op[v]["P3"]["tau"], tau4=op[v]["P4"]["tau"],
                                         P1_recall={k: op[v]["P1"]["recall_by_K"][k] for k in ("2", "5", "10", "30")})
                                 for v in R},
               omega_bprime_S8_max_abs_diff=om_max, pooling_dependent=pdep,
               omega_S8_v2={v: D[v]["omega_S8_v2"] for v in R}, omega_S8_v1={v: D[v]["omega_S8_v1"] for v in R},
               E4=e4, E7={v: {k: e7[v][k] for k in ("exceed_tau2_transported", "exceed_tau2_MEVA", "within_gap",
                                                     "within_gap_cue", "within_gap_backstop", "memory_TV")} for v in R},
               verdict_differs=[r["H"] for r in rows if r["variant"] == "orig"
                                and r["verdict"] != next(x["verdict"] for x in rows if x["variant"] == "corrected" and x["H"] == r["H"])])
    (R["corrected"] / "AMENDMENT_01_compare.json").write_text(json.dumps(out, indent=1, default=float))

    L = ["# AMENDMENT 01 — kết quả: bộ gốc (pre-registered mù) và bộ corrected (thứ tự khung LASIESTA đúng)\n",
         f"Sinh tự động bởi `code/amendment_compare.py` từ `results/orig/` và `results/corrected/`. "
         f"Pre-registration: gốc `{sha['orig'][:8]}…`, corrected `{sha['corrected'][:8]}…`. Theo quy tắc trong "
         "`prereg/AMENDMENT_01.md` §6, bộ corrected là kết luận chính; bộ gốc được báo song song.\n",
         "## 1. H1–H4\n", "| H | ngưỡng | quan sát — gốc | quan sát — corrected | kết luận gốc / corrected | p Holm gốc / corrected |",
         "|---|---|---|---|---|---|"]
    for h in ("H1", "H2", "H3", "H4"):
        o = H[(H.variant == "orig") & (H.H == h)].iloc[0]
        c = H[(H.variant == "corrected") & (H.H == h)].iloc[0]
        L.append(f"| {h} | {o.threshold} | {o.observed} | {c.observed} | **{o.verdict} / {c.verdict}** | "
                 f"{f3(o.p_holm, 4)} / {f3(c.p_holm, 4)} |")
    L.append("\nKết luận khác nhau giữa hai bộ: " + (", ".join(out["verdict_differs"]) or "**không có** (cả bốn H giống nhau ở hai bộ)") + ".\n")
    L += ["## 2. Ô H3 đạt theo chính sách / metric\n", "| nhóm | n | đạt gốc | đạt corrected | tỉ lệ gốc | tỉ lệ corrected | trung vị tỉ số sai số gốc / corrected |",
          "|---|---|---|---|---|---|---|"]
    for g, r in P.iterrows():
        L.append(f"| {g} | {int(r.n_corrected)} | {int(r.n_pass_orig)} | {int(r.n_pass_corrected)} | {f3(r.pass_rate_orig)} | "
                 f"{f3(r.pass_rate_corrected)} | {f3(r.median_error_ratio_orig)} / {f3(r.median_error_ratio_corrected)} |")
    L += ["\n## 3. Điểm vận hành, ω, cận\n", "| đại lượng | gốc | corrected |", "|---|---|---|"]
    for k in ("tau2", "tau3", "tau4"):
        L.append(f"| {k} | {op['orig'][k.replace('tau', 'P')]['tau']:.4f} | {op['corrected'][k.replace('tau', 'P')]['tau']:.4f} |")
    L.append(f"| ω (b′) S8 v2 / v1 | {f3(D['orig']['omega_S8_v2'])} / {f3(D['orig']['omega_S8_v1'])} | "
             f"{f3(D['corrected']['omega_S8_v2'])} / {f3(D['corrected']['omega_S8_v1'])} |")
    L.append(f"| max \\|Δ\\| ω_v2, ω_v1, π_pool trên mọi ô camera × phiên | — | {om_max:.2g} |")
    L.append(f"| block pooling-dependent | {', '.join(pdep['orig'])} | {', '.join(pdep['corrected'])} |")
    L.append(f"| E4: vi phạm cận oracle | {e4['orig'].get('oracle_violations', e4['orig'])} | {e4['corrected'].get('oracle_violations', e4['corrected'])} |")
    L += ["\n## 4. E6 (tỉ lệ đạt H3-ô theo trục)\n", "| trục | gốc | corrected | trung vị tỉ số sai số gốc / corrected |", "|---|---|---|---|"]
    for a, r in E6.iterrows():
        L.append(f"| {a} | {f3(r.pass_rate_orig)} | {f3(r.pass_rate_corrected)} | {f3(r.median_error_ratio_orig)} / {f3(r.median_error_ratio_corrected)} |")
    L += ["\n## 5. E7 (khám phá, không thuộc pre-registration)\n", "| đại lượng | gốc | corrected |", "|---|---|---|"]
    for k in ("exceed_tau2_transported", "exceed_tau2_MEVA"):
        L.append(f"| {k} | {f3(e7['orig'][k])} | {f3(e7['corrected'][k])} |")
    for k in ("within_gap", "within_gap_cue", "within_gap_backstop", "memory_TV"):
        a, b = e7["orig"][k], e7["corrected"][k]
        fmt = (lambda x: " / ".join(f"{p}: {f3(y)}" for p, y in x.items())) if isinstance(a, dict) else f3
        L.append(f"| {k} | {fmt(a)} | {fmt(b)} |")
    (REPO / "docs" / "AMENDMENT_01_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written docs/AMENDMENT_01_RESULTS.md")


if __name__ == "__main__":
    main()
