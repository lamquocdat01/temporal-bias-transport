"""Dot 5 P0.3 (cont.): shadow naive / transport estimates with the LASIESTA-corrected benchmark.

Needs cache/lasiesta_audit_segs.pkl from code/lasiesta_audit.py. Writes only results/lasiesta_audit/.
  - benchmark windows of the 7 pre-registered policy cells, old (reproduces results/E3) and corrected
    (corrected cue, g, re-searched operating points, corrected strata map);
  - per MEVA unit (camera, session, all; pooling-dependent block removed, D3.2) naive and transport, old vs new;
  - P1 cells: MEVA observed values do not depend on the benchmark -> exact new error ratio and H3 cell verdict;
    P2-P4: MEVA observed values depend on the frozen tau -> only the benchmark-side shift is reported here.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from policies import p1, p2, p3, p4, window_metrics, MCOLS
from strata import window_table
from transport import Bench, Target, units, transport_point, POLICIES, PNAME, MET
from prereg_build import target_windows
from meva_frames import load_segments

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = RES / "lasiesta_audit"


def bench_units(segs, op, mp):
    rows = []
    for s in segs:
        g = s["g"]["bprime"]
        wt = window_table(g)
        cell = wt.key.map(lambda k: mp.get(k, "U")).values
        pol = {("P2", 0): p2(s["dn"], op["tau2"][0], g), ("P3", 0): p3(s["dn"], op["tau3"][0], g),
               ("P4", 0): p4(s["pi"], op["tau4"][0], g)}
        for (name, K) in POLICIES:
            if name == "P1":
                m = np.mean([window_metrics(p1(len(g), K, f), g) for f in range(K)], axis=0)
            else:
                m = window_metrics(pol[(name, K)], g)
            rows.append(pd.DataFrame(m, columns=MCOLS).assign(video_uid=s["uid"], policy=name, K=K, cell=cell))
    return pd.concat(rows, ignore_index=True)


def main():
    d = pickle.load(open(REPO / "cache" / "lasiesta_audit_segs.pkl", "rb"))
    pre = json.loads((REPO / "prereg" / "PREREG_MEVA.json").read_text(encoding="utf-8"))
    pdep = set(pre["design"]["pooling_dependent_blocks"])
    e3 = pd.read_csv(RES / "E3_transport_cells.csv")
    tw_old = target_windows(load_segments())
    tw_new = tw_old.assign(cell=tw_old.key.map(lambda k: d["mp_new"].get(k, "U")))
    U = [u for u in units(tw_old, pdep) if u["level"] in ("camera", "session", "all")]
    out = []
    for tag, segs, op, tw in (("old", d["old"], d["op_old"], tw_old), ("new", d["new"], d["op_new"], tw_new)):
        bench = Bench(bench_units(segs, op, d[f"mp_{tag}"]))
        tgt = Target(tw, bench.cells)
        Stot = bench.S.sum(0)
        for u in U:
            F = tgt.wvec(u["blocks"])[:-1]
            for p, pk in enumerate(POLICIES):
                for met in MET:
                    nu, de = Bench.nu_de(Stot, p, met)
                    ok = F > 0
                    tr = transport_point(nu[ok], de[ok], F[ok]) if (ok.any() and F[ok] @ de[ok] > 0) else np.nan
                    out.append(dict(tag=tag, level=u["level"], unit=u["unit"], policy=PNAME[pk], metric=met,
                                    naive=bench.naive(p, met), transport=tr))
    o = pd.DataFrame(out).pivot_table(index=["level", "unit", "policy", "metric"], columns="tag",
                                      values=["naive", "transport"]).reset_index()
    o.columns = ["_".join(c).strip("_") for c in o.columns]
    m = o.merge(e3[["level", "unit", "policy", "metric", "role", "naive", "transport", "M_obs", "cell_PASS",
                    "error_ratio"]], on=["level", "unit", "policy", "metric"], how="left")
    chk = (m.naive_old - m.naive).abs().max(), (m.transport_old - m.transport).abs().max()
    print("reproduction of E3 (max |diff| naive, transport):", chk)
    assert chk[0] < 2e-6 and chk[1] < 2e-6, "old shadow (E3 csv stored with %.6g) does not reproduce E3_transport_cells.csv"
    m["d_naive"] = m.naive_new - m.naive_old
    m["d_transport"] = m.transport_new - m.transport_old
    p1c = m.policy.str.startswith("P1")
    m["error_ratio_new"] = np.where(p1c, (m.transport_new - m.M_obs).abs() / (m.naive_new - m.M_obs).abs(), np.nan)
    m["cell_PASS_new"] = np.where(p1c, m.error_ratio_new <= 0.5, None)
    m.to_csv(OUT / "transport_old_new.csv", index=False, float_format="%.6g")

    den = m[m.role == "H3_denominator"]
    dp1 = den[den.policy.str.startswith("P1") & den.cell_PASS.notna()]
    oth = den[~den.policy.str.startswith("P1")]
    summ = dict(
        H3_denominator_cells=int(len(den)), H3_pass_old_total=int(den.cell_PASS.sum()),
        P1_cells=int(len(dp1)), P1_pass_old=int(dp1.cell_PASS.sum()), P1_pass_new=int(dp1.cell_PASS_new.astype(bool).sum()),
        P1_flips_pass_to_fail=int((dp1.cell_PASS.astype(bool) & ~dp1.cell_PASS_new.astype(bool)).sum()),
        P1_flips_fail_to_pass=int((~dp1.cell_PASS.astype(bool) & dp1.cell_PASS_new.astype(bool)).sum()),
        P1_median_error_ratio_old=float(dp1.error_ratio.median()), P1_median_error_ratio_new=float(dp1.error_ratio_new.median()),
        P2_P4_cells=int(len(oth)), P2_P4_pass_old=int(oth.cell_PASS.sum()),
        P2_P4_median_abs_d_transport=float(oth.d_transport.abs().median()),
        P2_P4_max_abs_d_transport=float(oth.d_transport.abs().max()),
        P2_P4_median_abs_err_transport_old=float((oth.transport - oth.M_obs).abs().median()),
        P2_P4_median_abs_d_naive=float(oth.d_naive.abs().median()),
        P2_P4_cells_within_margin=int(((oth.transport - oth.M_obs).abs() - 0.5 * (oth.naive - oth.M_obs).abs()).abs()
                                      .le(oth.d_transport.abs() + 0.5 * oth.d_naive.abs()).sum()),
    )
    allu = m[m.level == "all"][["policy", "metric", "naive_old", "naive_new", "transport_old", "transport_new", "M_obs"]]
    summ["all_unit"] = allu.round(4).to_dict("records")
    (OUT / "transport_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "all_unit"}, indent=1, default=float))
    print(allu.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
