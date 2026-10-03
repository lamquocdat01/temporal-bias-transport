"""Census figure (descriptive, profile only): share of frames by recency class R (frames since the last event
frame, STRATA_SPEC v2 §3) and by activity state S, benchmark (121 videos) vs MEVA S8, definition (b').
Output: results/fig_census.pdf/.png, results/census_R_S.csv, numbers.json section 'census'.
"""
from __future__ import annotations
import variant as V
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#e6e5e1"
RORDER = ["lt300", "300-1800", "1800-18k", "gt18k", "censored"]
RLAB = ["<300", "300–1.8k", "1.8k–18k", ">18k", "unknown\n(segment start)"]


def shares():
    b = pd.read_parquet(RES / "benchmark_units.parquet")
    b = b[(b.definition == "bprime") & b.main & (b.policy == "P2")][["S", "R", "n"]]
    m = pd.read_parquet(V.CACHE / "meva_windows.parquet")
    m = m[(m.definition == "bprime") & m.in_S8][["S", "R", "n"]]
    rows = []
    for side, x in (("benchmark", b), ("MEVA S8", m)):
        r = x.R.where(~x.R.str.startswith("cens"), "censored")
        tot = x.n.sum()
        for k in RORDER:
            rows.append(dict(side=side, component="R", cls=k, share=float(x.n[r == k].sum() / tot)))
        for k in ("idle", "partial", "full"):
            rows.append(dict(side=side, component="S", cls=k, share=float(x.n[x.S == k].sum() / tot)))
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = shares()
    d.to_csv(RES / "census_R_S.csv", index=False, float_format="%.5f")
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    x = np.arange(len(RORDER))
    w = 0.38
    for j, (side, col) in enumerate((("benchmark", BLUE), ("MEVA S8", ORANGE))):
        v = d[(d.side == side) & (d.component == "R")].set_index("cls").share.reindex(RORDER).values
        ax.bar(x + (j - 0.5) * w, v, w * 0.92, color=col, label=side)
    ax.set_xticks(x, RLAB, fontsize=7)
    ax.set_ylabel("share of frames")
    ax.set_xlabel("frames since the last event frame (class $R$)", fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(RES / "fig_census.pdf")
    fig.savefig(RES / "fig_census.png", dpi=200)
    r = d[d.component == "R"].pivot(index="cls", columns="side", values="share")
    s = d[d.component == "S"].pivot(index="cls", columns="side", values="share")
    nj = json.loads((V.NUMBERS).read_text())
    nj["census"] = dict(R_ge1800_benchmark=float(r.loc[["1800-18k", "gt18k"], "benchmark"].sum()),
                        R_ge1800_MEVA=float(r.loc[["1800-18k", "gt18k"], "MEVA S8"].sum()),
                        R_censored_benchmark=float(r.loc["censored", "benchmark"]),
                        R_censored_MEVA=float(r.loc["censored", "MEVA S8"]),
                        idle_benchmark=float(s.loc["idle", "benchmark"]), idle_MEVA=float(s.loc["idle", "MEVA S8"]))
    (V.NUMBERS).write_text(json.dumps(nj, indent=1, default=float))
    print(r.round(3), s.round(3), sep="\n")


if __name__ == "__main__":
    main()
