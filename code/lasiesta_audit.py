"""Dot 5 P0: LASIESTA frame-order audit (lexicographic vs numeric file order, finding of topic #17, 27-09-2026).

Shadow computation only: nothing in results/ outside results/lasiesta_audit/ is written.

  0.1 layer matrix  (a) detections_cpu (SCM dump), (b) cue (code/cue.py), (c) frame_scores / events_durations:
      first 10 files of each layer's order vs natural order on a sample video, plus empirical proofs
      (cue cache recomputed from lex order = exact; stored detections = fresh detector on lex file).
  0.3 before/after (re-index only, no detector re-run): events (a)/(b)/(b'), median duration, longest
      idle run, cue quantiles; 5 sample videos (seed 42) and all 48.
      Benchmark-level shadow: operating points (tau2..tau4, P1 recall by K), raw-stratum frame shares,
      TV(w_B old, w_B new), strata cells, omega and TV(w_B, w_T) on MEVA S8 with the corrected map.

Run: python code/lasiesta_audit.py [--no-detector]
Outputs: results/lasiesta_audit/{layer_check.json, videos_before_after.csv, shadow_benchmark.json}
"""
from __future__ import annotations
import glob as _glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cue import thumb, diff_stats, normalise, PROC, SCM_DIR, SCM_PATHS  # noqa: E402
from e1_profile import load_det, g_from_det, DET, PERSON_CONF  # noqa: E402
from strata import merge_gaps, k_merge, window_table, events, build_strata_cells  # noqa: E402
import bench_units as bu  # noqa: E402
from policies import p1, p2, p3, p4, p4_features  # noqa: E402

SEED = 42
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "lasiesta_audit"
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def lex_files(fdir: str) -> list[str]:
    """Exactly the file order of scm dump_p1_detections.iter_frames_dir (sorted(set(glob)))."""
    files = []
    for ext in EXTS:
        files += _glob.glob(os.path.join(fdir, ext)) + _glob.glob(os.path.join(fdir, ext.upper()))
    return sorted(set(files))


def file_num(fp: str) -> int:
    return int(re.search(r"-(\d+)\.[A-Za-z]+$", os.path.basename(fp)).group(1))


def lex_to_time(files: list[str]) -> np.ndarray:
    """t[i] = 0-based temporal index of the file at lex position i; asserts numbering is 1..n."""
    t = np.array([file_num(f) - 1 for f in files])
    assert np.array_equal(np.sort(t), np.arange(len(files))), "non-contiguous numbering"
    return t


def cue_raw(files: list[str]):
    mad, fpx, prev = [], [], None
    for fp in files:
        th = thumb(cv2.imread(fp))
        m, f = diff_stats(prev, th) if prev is not None else (np.nan, np.nan)
        mad.append(m), fpx.append(f)
        prev = th
    return np.array(mad), np.array(fpx)


def prof(g: np.ndarray) -> dict:
    s, L, cens = events(g)
    idle = events(1 - g)[1]
    return dict(n_ev=int(len(L)), med_dur=float(np.median(L)) if len(L) else np.nan,
                mean_dur=float(L.mean()) if len(L) else np.nan,
                longest_idle=int(idle.max()) if len(idle) else 0, duty=float(g.mean()))


def tv(a: pd.Series, b: pd.Series) -> float:
    k = a.index.union(b.index)
    return float(0.5 * (a.reindex(k, fill_value=0) - b.reindex(k, fill_value=0)).abs().sum())


def main(run_detector: bool = True):
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCM_DIR / "scm"))
    import dump_p1_detections as dp
    cfg = dp.load_paths(str(SCM_PATHS))
    recs = {r[0].split("/")[-1]: r for r in dp.collect_videos(cfg, ["LASIESTA"], None)}
    assert len(recs) == 48
    ev = pd.read_csv(PROC / "events_durations.csv")
    key_of = {v.split("__")[-1]: v for v in ev[ev.dataset == "LASIESTA"].video.unique()}
    fs = pd.read_parquet(PROC / "frame_scores.parquet")
    fs = fs[fs.dataset == "LASIESTA"]
    cue = pd.read_parquet(REPO / "cache" / "bench_cue.parquet")
    cue = cue[cue.dataset == "LASIESTA"]

    # ------------------------------------------------------------------ 0.1 layer matrix
    sample = "I_BS_01"
    files = lex_files(recs[sample][2])
    nat = sorted(files, key=file_num)
    t = lex_to_time(files)
    det = load_det(DET / "detections_cpu" / "LASIESTA" / f"{key_of[sample]}.json")
    fsv = fs[fs.video == sample].sort_values("frame")
    lc = dict(sample=sample, n_files=len(files),
              natural_first10=[os.path.basename(f) for f in nat[:10]],
              layer_a_detections_first10=[os.path.basename(f) for f in files[:10]],
              layer_b_cue_first10=[os.path.basename(f) for f in files[:10]],
              layer_c_frame_scores_first10=[f"frame {i} = {os.path.basename(files[i])}" for i in fsv.frame.values[:10]],
              n_det_frames=len(det), n_frame_scores=len(fsv),
              lex_equals_natural={}, code_path=dict(
                  a="scm/dump_p1_detections.py iter_frames_dir: files = sorted(set(glob)); idx = position",
                  b="code/cue.py bench_main -> dp.make_frame_iter -> same iter_frames_dir",
                  c="frame_scores.parquet / events_durations.csv built from detections_cpu frame index (Deep3)"))
    for v, r in sorted(recs.items()):
        fl = lex_files(r[2])
        lc["lex_equals_natural"][v] = bool(np.array_equal(lex_to_time(fl), np.arange(len(fl))))
    lc["n_videos_lex_equals_natural"] = int(sum(lc["lex_equals_natural"].values()))
    # (b) proof: cached cue == cue recomputed from lex order, != natural order (first 60 frames)
    cs = cue[cue.video == sample].sort_values("frame")
    m_lex, _ = cue_raw(files[:60])
    m_nat, _ = cue_raw(nat[:60])
    lc["cue_cache_vs_lex_maxabs"] = float(np.nanmax(np.abs(cs.mad.values[:60] - m_lex)))
    lc["cue_cache_vs_natural_maxabs"] = float(np.nanmax(np.abs(cs.mad.values[:60] - m_nat)))
    # (c) proof: frame_scores g_t(a) == has-box of detections at the same index
    ga = g_from_det(det, fsv.frame.values, None)
    lc["frame_scores_g_equals_det_hasbox"] = bool(np.array_equal(ga, fsv.g_t.values.astype(np.int8)))
    # (a) proof: fresh detector on lex file i reproduces stored det[i]; on natural file i it does not
    if run_detector:
        from ultralytics import YOLO
        model = YOLO(cfg["yolo_model_path"])
        chk = []
        for i in (1, 2, 3, 10, 50, 100):
            row = {"index": i, "lex_file": os.path.basename(files[i]), "natural_file": os.path.basename(nat[i])}
            for tag, fp in (("lex", files[i]), ("natural", nat[i])):
                r = model.predict(cv2.imread(fp), conf=cfg["p1_conf_thr"], imgsz=cfg["p1_imgsz"],
                                  device="cpu", verbose=False)[0]
                b = r.boxes
                got = np.c_[b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy()] if len(b) else np.zeros((0, 6))
                ref = np.array(det.get(i, []), float).reshape(-1, 6)
                same = got.shape == ref.shape and (len(ref) == 0 or (
                    np.abs(got[:, :4] - ref[:, :4]).max() <= 1 and np.abs(got[:, 4] - ref[:, 4]).max() <= 1e-3
                    and np.array_equal(got[:, 5], ref[:, 5])))
                row[f"match_{tag}"] = bool(same)
                row[f"n_boxes_{tag}"] = int(len(got))
            row["n_boxes_stored"] = len(det.get(i, []))
            chk.append(row)
        lc["detector_check"] = chk
    (OUT / "layer_check.json").write_text(json.dumps(lc, indent=2))
    print(json.dumps({k: v for k, v in lc.items() if k != "lex_equals_natural"}, indent=1))

    # ------------------------------------------------------------------ 0.3 per video before / after
    rng = np.random.default_rng(SEED)
    five = sorted(rng.choice(sorted(recs), 5, replace=False).tolist())
    rows, corr = [], {}
    for v in sorted(recs):
        fl = lex_files(recs[v][2])
        t = lex_to_time(fl)
        det = load_det(DET / "detections_cpu" / "LASIESTA" / f"{key_of[v]}.json")
        fr = np.arange(len(fl))
        assert np.array_equal(fs[fs.video == v].sort_values("frame").frame.values, fr)
        order = np.argsort(t)                          # lex positions in temporal order
        c_old = cue[cue.video == v].sort_values("frame")
        mad_n, fpx_n = cue_raw([fl[i] for i in order])
        dn_o, _ = normalise(c_old.mad.values, c_old.fpx.values)
        dn_n, fn_n = normalise(mad_n, fpx_n)
        k = k_merge(25.0)
        rec = dict(video=v, n_frames=len(fl), in_sample5=v in five)
        for dname, pc in (("a", None), ("b", PERSON_CONF)):
            g_old = g_from_det(det, fr, pc)
            g_new = g_old[order]
            gs = {dname: (g_old, g_new)}
            if dname == "b":
                gs["bprime"] = (merge_gaps(g_old, k), merge_gaps(g_new, k))
                corr[v] = dict(g_b=g_new, g_bp=gs["bprime"][1], dn=dn_n, fn=fn_n)
            for dd, (go, gn) in gs.items():
                for tag, g in (("old", go), ("new", gn)):
                    for kk, val in prof(g).items():
                        rec[f"{dd}_{kk}_{tag}"] = val
        for tag, dn in (("old", dn_o), ("new", dn_n)):
            q = np.nanpercentile(dn, [50, 90, 99])
            rec.update({f"dn_q50_{tag}": q[0], f"dn_q90_{tag}": q[1], f"dn_q99_{tag}": q[2]})
        rec["mad_mean_old"], rec["mad_mean_new"] = float(np.nanmean(c_old.mad)), float(np.nanmean(mad_n))
        rows.append(rec)
        print(v, rec["bprime_n_ev_old"], "->", rec["bprime_n_ev_new"], flush=True)
    vt = pd.DataFrame(rows)
    vt.to_csv(OUT / "videos_before_after.csv", index=False, float_format="%.6g")

    # ------------------------------------------------------------------ benchmark-level shadow
    segs = bu.load_segments()
    main_old = [s for s in segs if s["main"]]
    new = []
    for s in main_old:
        s2 = dict(s)
        if s["dataset"] == "LASIESTA":
            c = corr[s["video"]]
            s2.update(dn=c["dn"], fn=c["fn"], g={"b": c["g_b"], "bprime": c["g_bp"]})
        new.append(s2)

    def ops(ms):
        rng = np.random.default_rng(SEED)
        fit = []
        for ds in ("BMC", "CDnet2014", "LASIESTA"):
            u = sorted(s["uid"] for s in ms if s["dataset"] == ds)
            fit += list(rng.permutation(u)[: len(u) // 2])
        fit = set(fit)
        for s in ms:
            s["X"] = p4_features(s["dn"], s["fn"])
        clf = LogisticRegression(max_iter=2000).fit(np.vstack([s["X"] for s in ms if s["uid"] in fit]),
                                                    np.concatenate([s["g"]["bprime"] for s in ms if s["uid"] in fit]))
        for s in ms:
            s["pi"] = clf.predict_proba(s["X"])[:, 1]
        t2 = bu.search_tau(ms, lambda s, t: p2(s["dn"], t, s["g"]["bprime"]), 0.2, 200)
        t3 = bu.search_tau(ms, lambda s, t: p3(s["dn"], t, s["g"]["bprime"]), 0.2, 200)
        t4 = bu.search_tau(ms, lambda s, t: p4(s["pi"], t, s["g"]["bprime"]), 1e-4, 0.999)
        rk = {K: float(np.mean([bu.pooled_recall(ms, lambda s, K=K, f=f: p1(len(s["dn"]), K, f)) for f in range(K)]))
              for K in (2, 5, 10, 30)}
        n_ev = {ds: int(sum(len(events(s["g"]["bprime"])[1]) for s in ms if s["dataset"] == ds))
                for ds in ("BMC", "CDnet2014", "LASIESTA")}
        return dict(tau2=t2, tau3=t3, tau4=t4, P1_recall=rk, n_events_bprime=n_ev)

    old_list = [dict(s) for s in main_old]
    op_old, op_new = ops(old_list), ops(new)
    op_file = json.loads((REPO / "results" / "operating_points.json").read_text())
    assert abs(op_old["tau2"][0] - op_file["P2"]["tau"]) < 1e-9, "old shadow does not reproduce operating_points.json"

    def bwin(ms):
        return pd.concat([window_table(s["g"]["bprime"]).assign(video_uid=s["uid"], dataset=s["dataset"]) for s in ms],
                         ignore_index=True)
    bw_old, bw_new = bwin(main_old), bwin(new)
    f_old = bw_old.groupby("key").n.sum() / bw_old.n.sum()
    f_new = bw_new.groupby("key").n.sum() / bw_new.n.sum()
    las_o = bw_old[bw_old.dataset == "LASIESTA"]
    las_n = bw_new[bw_new.dataset == "LASIESTA"]
    mp_old, cells_old = build_strata_cells(bw_old, "v2")
    mp_new, cells_new = build_strata_cells(bw_new, "v2")
    wb_old = bw_old.assign(h=bw_old.key.map(mp_old)).groupby("h").n.sum()
    wb_new = bw_new.assign(h=bw_new.key.map(mp_new)).groupby("h").n.sum()
    wb_old, wb_new = wb_old / wb_old.sum(), wb_new / wb_new.sum()
    ref = pd.read_csv(REPO / "results" / "w_B_strata.csv")
    ref = ref[(ref.definition == "bprime") & (ref.rule == "v2") & (ref.arm == "main")].set_index("h").w_B
    assert tv(ref, wb_old) < 1e-5, "old shadow w_B differs from results/w_B_strata.csv"
    # MEVA side: windows (b'), S8, all level and per camera/session, with old vs corrected map
    mw = pd.read_parquet(REPO / "cache" / "meva_windows.parquet")
    mw = mw[(mw.definition == "bprime") & mw.in_S8]
    om = []
    for level, col in (("all", None), ("camera", "cam_tag"), ("session", "session")):
        for k_, y in (y_ for y_ in (mw.groupby(col) if col else [("ALL", mw)])):
            r = dict(level=level, unit=k_)
            for tag, mp, wb in (("old", mp_old, wb_old), ("new", mp_new, wb_new)):
                f = y.assign(h=y.key.map(lambda x, mp=mp: mp.get(x, "U"))).groupby("h").n.sum()
                w = f / f.sum()
                r[f"omega_{tag}"], r[f"TV_{tag}"] = float(w.get("U", 0.0)), tv(wb, w)
            om.append(r)
    om = pd.DataFrame(om)
    import pickle
    pickle.dump(dict(old=old_list, new=new, op_old=op_old, op_new=op_new, mp_old=mp_old, mp_new=mp_new),
                open(REPO / "cache" / "lasiesta_audit_segs.pkl", "wb"))
    om.to_csv(OUT / "omega_old_new.csv", index=False, float_format="%.5g")
    shadow = dict(
        operating_points_old=op_old, operating_points_new=op_new,
        frames_LASIESTA=int(las_o.n.sum()), frames_total=int(bw_old.n.sum()),
        TV_raw_strata_LASIESTA_old_new=tv(las_o.groupby("key").n.sum() / las_o.n.sum(),
                                          las_n.groupby("key").n.sum() / las_n.n.sum()),
        TV_raw_strata_benchmark_old_new=tv(f_old, f_new),
        TV_wB_cells_old_new=tv(wb_old, wb_new),
        n_cells_old=len(cells_old), n_cells_new=len(cells_new),
        cells_only_old=sorted(set(cells_old) - set(cells_new)), cells_only_new=sorted(set(cells_new) - set(cells_old)),
        benchmark_U_frames_old=float(wb_old.get("U", 0.0)), benchmark_U_frames_new=float(wb_new.get("U", 0.0)),
        LASIESTA_S_share_old=(las_o.groupby("S").n.sum() / las_o.n.sum()).to_dict(),
        LASIESTA_S_share_new=(las_n.groupby("S").n.sum() / las_n.n.sum()).to_dict(),
        LASIESTA_D_share_old=(las_o.groupby("D").n.sum() / las_o.n.sum()).to_dict(),
        LASIESTA_D_share_new=(las_n.groupby("D").n.sum() / las_n.n.sum()).to_dict(),
        omega_TV_all=om[om.level == "all"].iloc[0].to_dict(), sample5=five, seed=SEED)
    (OUT / "shadow_benchmark.json").write_text(json.dumps(shadow, indent=2, default=float))
    print(json.dumps(shadow, indent=1, default=float))
    print(om.round(4).to_string())


if __name__ == "__main__":
    main(run_detector="--no-detector" not in sys.argv)
