"""pi_pool = omega_v1 - omega_v2 on the MEVA S8 (b') profile, per camera x session cell (= S8 block),
per camera, per session and overall; plus a diagnosis of one cell's dominant stratum under v2 pooling.

Inputs: results/meva_omega.csv (arm main, subset S8), cache/meva_windows.parquet (from meva_profile.py),
results/benchmark_units.parquet, results/strata_map_benchmark.csv, results/w_B_strata.csv.
Outputs: results/omega_bprime_S8.csv, results/omega_pool_diag_<block>.json.
Usage: python code/omega_pooling.py [block_id ...]   (default: G331_2018-03-07_165000 = P-4 x 03-07)
"""
from __future__ import annotations
import variant as V
import json
import sys
from pathlib import Path

import pandas as pd

from strata import build_strata_cells

REPO = Path(__file__).resolve().parents[1]
RES = V.RES
DEF, ARM = "bprime", "main"


def omega_table() -> pd.DataFrame:
    om = pd.read_csv(RES / "meva_omega.csv")
    om = om[(om.definition == DEF) & (om.arm == ARM) & (om.subset == "S8")]
    e1 = pd.read_csv(RES / "E1_profile_meva.csv")
    blk = e1[(e1.level == "block") & (e1.definition == DEF)].set_index("block_id")[["cam_tag", "session"]]
    w = om.pivot_table(index=["level", "unit"], columns="rule", values="omega").reset_index()
    fr = om[om.rule == "v2"].set_index(["level", "unit"]).frames
    w["n_frames"] = [int(fr.loc[(lv, u)]) for lv, u in zip(w.level, w.unit)]
    w["cam_tag"] = [blk.cam_tag.get(u, u if lv == "camera" else "") if lv in ("block", "camera") else ""
                    for lv, u in zip(w.level, w.unit)]
    w["session"] = [blk.session.get(u, "") if lv == "block" else (u if lv == "session" else "")
                    for lv, u in zip(w.level, w.unit)]
    w = w.rename(columns={"v2": "omega_v2", "v1": "omega_v1"})
    w["pi_pool"] = w.omega_v1 - w.omega_v2
    w["pooling_dependent"] = w.pi_pool > 0.5          # PREREG_DECISIONS P-1 / D3.2 (signed)
    order = {"block": 0, "camera": 1, "session": 2, "all": 3}
    w = w.assign(level=w.level.replace({"block": "cell"}), o=w.level.map(order)).sort_values(["o", "cam_tag", "session"])
    return w[["level", "unit", "cam_tag", "session", "omega_v2", "omega_v1", "pi_pool", "n_frames",
              "pooling_dependent"]].reset_index(drop=True)


def bench_windows() -> pd.DataFrame:
    b = pd.read_parquet(RES / "benchmark_units.parquet")
    b = b[(b.definition == DEF) & b.main]
    return b.drop_duplicates(["video_uid", "win_idx"])[["video_uid", "win_idx", "n", "key", "h_v2_main", "h_v1_main"]]


def diagnose(block: str, bw: pd.DataFrame) -> dict:
    win = pd.read_parquet(V.CACHE / "meva_windows.parquet")
    x = win[(win.definition == DEF) & (win.block_id == block)]
    tot = int(x.n.sum())
    by = (x.groupby(["key", "h_v2_main", "h_v1_main"]).n.agg(["sum", "size"])
          .rename(columns={"sum": "frames", "size": "n_win"}).sort_values("frames", ascending=False).reset_index())
    top = by.iloc[0]
    smap = pd.read_csv(RES / "strata_map_benchmark.csv")
    smap = smap[(smap.definition == DEF) & (smap.arm == ARM)]
    wB = pd.read_csv(RES / "w_B_strata.csv")
    wB = wB[(wB.definition == DEF) & (wB.arm == ARM)]
    mp2, cells2 = build_strata_cells(bw, "v2")
    assert all(mp2.get(k, "U") == h for k, h in zip(bw.key, bw.h_v2_main)), "v2 map not reproduced"
    stats = bw.groupby("key").agg(n_win=("key", "size"), n_vid=("video_uid", "nunique"), frames=("n", "sum"))

    def raw_support(k):
        return dict(n_win=int(stats.n_win.get(k, 0)), n_vid=int(stats.n_vid.get(k, 0)),
                    frames=int(stats.frames.get(k, 0)))

    cell = top.h_v2_main
    members = cells2.get(cell, [])
    cell_rows = bw[bw.key.isin(members)]
    tgt = dict(cell=cell, members=[dict(key=m, **raw_support(m)) for m in members],
               n_win=int(len(cell_rows)), n_vid=int(cell_rows.video_uid.nunique()),
               frames=int(cell_rows.n.sum()),
               w_B_v2=float(wB[(wB.rule == "v2") & (wB.h == cell)].w_B.sum()))
    S, D, R, C = top.key.split("|")
    return dict(block=block, definition=DEF, frames_block=tot,
                dominant=dict(key=top.key, S=S, D=D, R=R, C=C, frames=int(top.frames), n_win=int(top.n_win),
                              share=float(top.frames / tot), h_v2=top.h_v2_main, h_v1=top.h_v1_main),
                bench_support_raw=raw_support(top.key),
                v1_status=dict(h=top.h_v1_main,
                               in_strata_map=smap[(smap.rule == "v1") & (smap.key == top.key)].h.tolist()),
                v2_target_cell=tgt,
                strata_in_block=by.assign(share=by.frames / tot).to_dict("records"))


def main():
    t = omega_table()
    t.to_csv(RES / "omega_bprime_S8.csv", index=False, float_format="%.4f")
    print(t.round(3).to_string())
    bw = bench_windows()
    for block in (sys.argv[1:] or ["G331_2018-03-07_165000"]):
        d = diagnose(block, bw)
        (RES / f"omega_pool_diag_{block}.json").write_text(json.dumps(d, indent=1, default=str, ensure_ascii=False))
        print(json.dumps({k: v for k, v in d.items() if k != "strata_in_block"}, indent=1, default=str,
                         ensure_ascii=False))
        print(pd.DataFrame(d["strata_in_block"]).round(3).to_string())


if __name__ == "__main__":
    main()
