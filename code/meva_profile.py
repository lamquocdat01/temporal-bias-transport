"""MEVA temporal PROFILE (Dot 2 part C) -- no policy is run and no m_T(h) is computed here.

STRATA_SPEC v2. Per block (MEVA_selection v1), split into segments at gaps > 2 s (and, for (c), at
unlabeled clips):
  - E1 census under (a) has-box, (b) person >= 0.7, (b') = (b) with idle gaps <= round(0.2 s x fps)
    merged [(a)/(b)/(b') only for blocks fully covered by the detector], (c) MEVA activity labels;
  - windows/strata h(u) mapped with the BENCHMARK cells (results/strata_map_benchmark.csv), rules v2
    (pooled, main) and v1 (E6); w_T(h), omega = target frame mass in U, TV(w_B, w_T). Both omegas.
Levels: block, camera, session, all. Two-way (camera x session) bootstrap B=1000, seed 42, for the
'all' omega and TV; block bootstrap reported as sensitivity.

Outputs: results/E1_profile_meva.csv, results/meva_wT_strata.csv, results/meva_omega.csv,
         results/meva_processed_hours.json, cache/meva_windows.parquet
"""
from __future__ import annotations
import paths
import variant as V
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e1_profile import km_median
from strata import window_table, build_strata_cells, events, merge_gaps, k_merge
from paths import MEVA_DIR

SEED = 42
B = 1000
REPO = Path(__file__).resolve().parents[1]
RES = V.RES
DATA = MEVA_DIR
PROC = paths.BENCH_PROC
GAP_S = 2.0


def tsec(t: str) -> float:
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


def sessions(blocks: pd.DataFrame) -> dict:
    """Session = connected set of same-date blocks whose time intervals overlap (any camera)."""
    b = blocks.assign(a=blocks.start.map(tsec), e=blocks.end.map(tsec)).sort_values(["date", "a"])
    groups: list[list] = []
    for _, g in b.groupby("date", sort=True):
        cur_end = -1
        for r in g.itertuples(index=False):
            if r.a > cur_end:
                groups.append([])
            groups[-1].append(r)
            cur_end = max(cur_end, r.e)
    # session name = date + earliest start among its blocks
    return {r.block_id: f"{grp[0].date}_{min(x.start for x in grp)[:5]}" for grp in groups for r in grp}


def clip_frames(stem: str, key: str, man: pd.DataFrame) -> int:
    cz = DATA / "cue" / f"{stem}.npz"
    if cz.exists():
        return int(np.load(cz)["n_decoded"])
    dj = DATA / "detections" / f"{stem}.json"
    if dj.exists():
        return int(json.load(open(dj))["meta"]["n_frames"])
    return int(man.loc[key, "n_frames_header"])


def g_det(stem: str, n: int):
    p = DATA / "detections" / f"{stem}.json"
    if not p.exists():
        return None, None
    fr = json.load(open(p))["frames"]
    assert len(fr) == n, (stem, len(fr), n)
    ga = np.array([len(f["boxes"]) > 0 for f in fr], np.int8)
    gb = np.array([any(b[5] == 0 and b[4] >= 0.7 for b in f["boxes"]) for f in fr], np.int8)
    return ga, gb


def g_lab(stem: str, n: int):
    z = np.load(DATA / "labels" / f"{stem}.npz")
    if str(z["src"]) == "":
        return None
    g = z["g_c"]
    assert len(g) == n, (stem, len(g), n)
    return g


def profile_row(segs, fps):
    """E1 statistics over a list of g arrays (segments)."""
    n = sum(len(g) for g in segs)
    L_all, c_all, o_all, idle = [], [], [], []
    for g in segs:
        s, L, c = events(g)
        L_all += list(L)
        c_all += list(c)
        o_all += list(s == 0)
        _, Li, _ = events(1 - g)
        idle += list(Li)
    L_all, c_all, o_all = np.array(L_all), np.array(c_all, bool), np.array(o_all, bool)
    q = np.percentile(L_all, [10, 50, 90]) if len(L_all) else [np.nan] * 3
    li = max(idle) if idle else 0
    return dict(n_frames=n, hours=n / fps / 3600, duty_cycle=float(np.mean(np.concatenate(segs))),
                n_events=len(L_all), event_rate_per_min=len(L_all) / (n / fps / 60),
                dur_q10=q[0], dur_q50=q[1], dur_q90=q[2],
                frac_boundary=float(c_all.mean()) if len(c_all) else np.nan,
                median_KM=km_median(L_all, c_all) if len(L_all) else np.nan,
                median_KM_onset_in=km_median(L_all[~o_all], c_all[~o_all]) if (~o_all).any() else np.nan,
                longest_idle_frames=int(li), longest_idle_sec=li / fps, n_segments=len(segs))


def bench_windows_a():
    """Benchmark windows under definition (a) (g_t = has-box), 121 main videos."""
    fs = V.frame_scores()
    fs = fs[fs.dataset.isin(["CDnet2014", "LASIESTA", "BMC"])]
    rows = []
    for (d, v), x in fs.groupby(["dataset", "video"]):
        wt = window_table(x.sort_values("frame").g_t.values.astype(np.int8))
        rows.append(wt.assign(video_uid=f"{d}/{v}"))
    return pd.concat(rows, ignore_index=True)


def tv(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return 0.5 * float((p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())


def spec_guard():
    """Refuse to profile MEVA unless STRATA_SPEC.md matches its committed .sha256 (spec frozen first)."""
    import hashlib
    import subprocess
    spec, sha = REPO / "docs" / "STRATA_SPEC.md", REPO / "docs" / "STRATA_SPEC.md.sha256"
    if not sha.exists():
        raise SystemExit("REFUSED: docs/STRATA_SPEC.md.sha256 missing -- hash and commit the spec first.")
    h = hashlib.sha256(spec.read_bytes()).hexdigest()
    if h != sha.read_text().split()[0]:
        raise SystemExit("REFUSED: STRATA_SPEC.md differs from its recorded hash.")
    for f in ("docs/STRATA_SPEC.md", "docs/STRATA_SPEC.md.sha256"):
        if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", f]).returncode != 0 or            subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", f], capture_output=True).returncode != 0:
            raise SystemExit(f"REFUSED: {f} is not committed.")
    return h


def main():
    print("STRATA_SPEC sha256", spec_guard())
    sel = pd.read_csv(RES / "meva_selection.csv")
    sel = sel[sel.tier == "PRIMARY"]
    blocks = pd.read_csv(RES / "meva_selection_blocks.csv")
    blocks = blocks[blocks.tier == "PRIMARY"].copy()
    s8 = set(pd.read_csv(RES / "meva_S8_blocks.csv").block_id)
    sess = sessions(blocks)
    blocks["session"] = blocks.block_id.map(sess)
    man = pd.read_csv(RES / "meva_manifest.csv").set_index("key")

    # benchmark cells: (b'), (b) from bench_units (rules v2/v1, arms main/121+VP); (a) rebuilt here
    smap = pd.read_csv(RES / "strata_map_benchmark.csv")
    maps = {(d, r, a): dict(zip(g.key, g.h)) for (d, r, a), g in smap.groupby(["definition", "rule", "arm"])}
    wB = {(d, r, a): g.set_index("h").w_B
          for (d, r, a), g in pd.read_csv(RES / "w_B_strata.csv").groupby(["definition", "rule", "arm"])}
    ba = bench_windows_a()
    for rule in ("v2", "v1"):
        mp, _ = build_strata_cells(ba, rule)
        maps[("a", rule, "main")] = mp
        fa = ba.assign(h=ba.key.map(mp)).groupby("h").n.sum()
        wB[("a", rule, "main")] = fa / fa.sum()

    e1_rows, win_rows = [], []
    hours = {"a": 0.0, "b": 0.0, "bprime": 0.0, "c": 0.0, "cue": 0.0, "downloaded": 0.0}
    for blk in blocks.itertuples(index=False):
        clips = sel[sel.block_id == blk.block_id].sort_values(["date", "start"])
        if not all(k in man.index for k in clips.key):
            continue
        fps = float(man.loc[clips.key.iloc[0], "fps"])
        info = []
        for c in clips.itertuples(index=False):
            stem = Path(c.key).stem
            n = clip_frames(stem, c.key, man)
            ga, gb = g_det(stem, n)
            info.append(dict(stem=stem, n=n, a=ga, b=gb, c=g_lab(stem, n), start=tsec(c.start), end=tsec(c.end)))
            hours["downloaded"] += n / fps / 3600
            hours["cue"] += n / fps / 3600 if (DATA / "cue" / f"{stem}.npz").exists() else 0
        for d in ("a", "b", "bprime", "c"):
            src = "b" if d == "bprime" else d
            if src in "ab" and any(i[src] is None for i in info):
                continue                        # detector coverage is per whole block
            segs, cur, prev_end = [], [], None
            for i in info:
                if i[src] is None or (prev_end is not None and i["start"] - prev_end > GAP_S):
                    if cur:
                        segs.append(np.concatenate(cur))
                    cur = []
                if i[src] is not None:
                    cur.append(i[src])
                    hours[d] += i["n"] / fps / 3600
                prev_end = i["end"]
            if cur:
                segs.append(np.concatenate(cur))
            if d == "bprime":                   # gap merge within each segment (k = 6 at 30 fps)
                segs = [merge_gaps(g, k_merge(fps)) for g in segs]
            if not segs:
                continue
            e1_rows.append(dict(level="block", definition=d, cam_tag=blk.cam_tag, camera=blk.camera,
                                session=blk.session, block_id=blk.block_id, in_S8=blk.block_id in s8,
                                quiet=blk.quiet, fps=fps, **profile_row(segs, fps)))
            for si, g in enumerate(segs):
                wt = window_table(g)
                win_rows.append(wt.assign(definition=d, cam_tag=blk.cam_tag, session=blk.session,
                                          block_id=blk.block_id, in_S8=blk.block_id in s8, seg=si))
    e1 = pd.DataFrame(e1_rows)
    win = pd.concat(win_rows, ignore_index=True)
    for (d, rule, arm), mp in maps.items():
        sel_d = win.definition == d
        win.loc[sel_d, f"h_{rule}_{arm}"] = win.loc[sel_d, "key"].map(lambda k, mp=mp: mp.get(k, "U"))
    # (c) has no benchmark counterpart: mapped with the (b') benchmark cells, flagged cross-definition
    for rule in ("v2", "v1"):
        mp = maps[("bprime", rule, "main")]
        win.loc[win.definition == "c", f"h_{rule}_main"] = win.loc[win.definition == "c", "key"].map(
            lambda k, mp=mp: mp.get(k, "U"))
    V.CACHE.mkdir(exist_ok=True)
    win.to_parquet(V.CACHE / "meva_windows.parquet", index=False)

    # ---- aggregated E1 rows (camera / session / all), recomputed from the block rows' frames
    agg = []
    for d, x in e1.groupby("definition"):
        for level, col in (("camera", "cam_tag"), ("session", "session"), ("all", None), ("all_S8", None)):
            if level == "all_S8":               # Dot 3 P0.4: S8-only aggregate (comparable across definitions)
                x = x[x.in_S8]
            groups = x.groupby(col) if col else [("ALL", x)]
            for k, y in groups:
                w = y.n_frames
                agg.append(dict(level=level, definition=d, cam_tag=k if level == "camera" else "",
                                session=k if level == "session" else "", n_blocks=len(y),
                                n_frames=int(w.sum()), hours=float(y.hours.sum()),
                                duty_cycle=float(np.average(y.duty_cycle, weights=w)),
                                n_events=int(y.n_events.sum()),
                                event_rate_per_min=float(y.n_events.sum() / (y.hours.sum() * 60)),
                                longest_idle_sec=float(y.longest_idle_sec.max()),
                                median_block_duty=float(y.duty_cycle.median())))
    e1_all = pd.concat([e1, pd.DataFrame(agg)], ignore_index=True)
    e1_all.to_csv(RES / "E1_profile_meva.csv", index=False, float_format="%.6g")

    # ---- w_T(h), omega, TV by level; (b) main is the transport-relevant row
    rng = np.random.default_rng(SEED)
    om_rows, wt_rows = [], []
    # (target definition, rule, arm, benchmark definition); first row = main transport profile
    combos = [("bprime", "v2", "main", "bprime"), ("bprime", "v1", "main", "bprime"),
              ("bprime", "v2", "121+VP", "bprime"), ("b", "v2", "main", "b"), ("b", "v1", "main", "b"),
              ("a", "v2", "main", "a"), ("a", "v1", "main", "a"), ("c", "v2", "main", "bprime")]
    for d, rule, arm, bdef in combos:
        x = win[win.definition == d]
        if x.empty:
            continue
        col = f"h_{rule}_{arm}"
        wb = wB[(bdef, rule, arm)]
        for subset, xs in (("S8", x[x.in_S8]), ("all_available", x)):
            if xs.empty:
                continue
            for level, gcol in (("block", "block_id"), ("camera", "cam_tag"), ("session", "session"), ("all", None)):
                groups = xs.groupby(gcol) if gcol else [("ALL", xs)]
                for k, y in groups:
                    f = y.groupby(col).n.sum()
                    w = f / f.sum()
                    om_rows.append(dict(definition=d, bench_def=bdef, rule=rule, arm=arm, subset=subset,
                                        level=level, unit=k, frames=int(f.sum()), omega=float(w.get("U", 0.0)),
                                        TV=tv(wb, w)))
                    if level in ("camera", "all"):
                        wt_rows += [dict(definition=d, rule=rule, arm=arm, subset=subset, level=level, unit=k, h=h,
                                         frames=int(ff), w_T=float(ww), w_B=float(wb.get(h, 0.0)))
                                    for h, ff, ww in zip(f.index, f.values, w.values)]
            # two-way bootstrap (camera x session) and block bootstrap for the 'all' row
            cams, sss, blks = sorted(xs.cam_tag.unique()), sorted(xs.session.unique()), sorted(xs.block_id.unique())
            fb = xs.groupby(["cam_tag", "session", col]).n.sum()
            fbb = xs.groupby(["block_id", col]).n.sum()
            bo, bt, bbo, bbt = [], [], [], []
            for _ in range(B):
                cs = rng.choice(cams, len(cams))
                ss = rng.choice(sss, len(sss))
                parts = [fb.loc[(c, s)] for c in cs for s in ss if (c, s) in fb.index.droplevel(2)]
                if parts:
                    f = pd.concat(parts).groupby(level=0).sum()
                    w = f / f.sum()
                    bo.append(w.get("U", 0.0))
                    bt.append(tv(wb, w))
                bs = rng.choice(blks, len(blks))
                f = pd.concat([fbb.loc[b_] for b_ in bs]).groupby(level=0).sum()
                w = f / f.sum()
                bbo.append(w.get("U", 0.0))
                bbt.append(tv(wb, w))
            for r in om_rows:
                if (r["definition"], r["rule"], r["arm"], r["subset"], r["level"]) == (d, rule, arm, subset, "all"):
                    r.update(omega_ci2w=list(np.percentile(bo, [2.5, 97.5])), TV_ci2w=list(np.percentile(bt, [2.5, 97.5])),
                             omega_ciblk=list(np.percentile(bbo, [2.5, 97.5])), TV_ciblk=list(np.percentile(bbt, [2.5, 97.5])),
                             n_cam=len(cams), n_sess=len(sss), n_blocks=len(blks))
    pd.DataFrame(om_rows).to_csv(RES / "meva_omega.csv", index=False, float_format="%.5g")
    # Dot 3 P0.4: TV(w_B, w_T) and omega (rule v2, arm main) on the aggregated E1 rows
    om_all = pd.DataFrame(om_rows)
    om_all = om_all[(om_all.rule == "v2") & (om_all.arm == "main") & (om_all.level == "all")]
    for lvl, sub in (("all", "all_available"), ("all_S8", "S8")):
        for d in e1_all.definition.unique():
            r = om_all[(om_all.definition == d) & (om_all.subset == sub)]
            if len(r):
                m = (e1_all.level == lvl) & (e1_all.definition == d)
                e1_all.loc[m, "TV_v2"] = float(r.TV.iloc[0])
                e1_all.loc[m, "omega_v2"] = float(r.omega.iloc[0])
                e1_all.loc[m, "TV_bench_def"] = r.bench_def.iloc[0]
    e1_all.to_csv(RES / "E1_profile_meva.csv", index=False, float_format="%.6g")
    pd.DataFrame(wt_rows).to_csv(RES / "meva_wT_strata.csv", index=False, float_format="%.6g")
    (RES / "meva_processed_hours.json").write_text(json.dumps({k: round(v, 3) for k, v in hours.items()}, indent=1))
    print(json.dumps(hours, indent=1))
    om = pd.DataFrame(om_rows)
    print(om[om.level.isin(["camera", "session", "all"])][["definition", "rule", "arm", "subset", "level", "unit",
                                                             "frames", "omega", "TV"]].round(3).to_string())


if __name__ == "__main__":
    main()
