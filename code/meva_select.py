"""Deterministic, metadata-only selection of the MEVA 20 h primary set + reserve.

Inputs (metadata only):
    results/meva_inventory_labeled.csv  (from meva_inventory.py + meva_label_coverage.py)
    meva-data-repo/metadata/meva-clip-camera-and-time-table.txt   (official camera-set per clip)
    meva-data-repo/metadata/meva-camera-daily-status.txt          (official int/ext per camera-day)
Never reads video pixels, annotation contents, activity counts, or file sizes for ranking
(sizes are only summed for the GB report).

Outputs:
    results/meva_selection.csv          one row per clip (tier, rank, camera, block, key, ...)
    results/meva_selection_blocks.csv   one row per block
    docs/MEVA_selection.md              rule (verbatim RULE_TEXT, fixed before running) + lists
    docs/MEVA_selection.md.sha256
"""
import hashlib
import os

import numpy as np
import pandas as pd
from paths import MEVA_REPO

SEED = 42  # no randomness is used; kept for the record
np.random.seed(SEED)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
DOCS = os.path.join(REPO, "docs")
META = str(MEVA_REPO / "metadata")

# ---- pre-registered parameters (see RULE_TEXT) ----
GAP_MAX_S = 330          # tiny-gap tolerance: at most ONE missing 5-min slot between consecutive clips
OVERLAP_MIN_S = -30      # overlap tolerance
MIN_BLOCK_CLIPS = 6      # blocks shorter than 30 min are discarded
BLOCK_LABEL_MIN = 0.8    # primary blocks: >= 80 % of clips have a Kitware activities.yml
CAM_LABEL_MIN = 0.5      # camera eligibility: >= 50 % clips labeled
CAM_SET_FRAC = 0.9       # camera eligibility: >= 90 % clips in a numbered (fixed-view EO) camera set
CAM_MIN_DATES = 6
N_PRIMARY = 5
SITE_MAX = 2
H_MIN, H_MAX = 4.0, 4.5  # per-camera hours target for primary and reserve cameras
N_RESERVE_CAMS = 3
QUIET = lambda t: (t.hour < 10) or (t.hour >= 17)

RULE_TEXT = """## 1. Selection rule (pre-registered; written before the rule was applied)

Scope: `s3://mevadata-public-01/drops-123-r13/` only. Inputs are METADATA ONLY: clip file names
(date/start/end/site/camera), the existence (not the content) of a Kitware `*.activities.yml`
file for the clip (`kitware/` or `kitware-meva-training/`), the official clip table
(`metadata/meva-clip-camera-and-time-table.txt`, camera-set field) and the official
int/ext daily status. Video pixels, activity counts/types/levels and file sizes are NOT used
to rank or choose anything (sizes are summed only to report GB). No randomness is used
(seed 42 declared for completeness). Code: `code/meva_select.py`.

Feasibility fact that forced the block definition (checked from filenames only, before the
rule was written): in drops-123-r13 the longest gap-free run on ANY camera is 2.82 h
(2018-03-07 09:11-12:00); recording sessions per camera-day are 0.25-2.8 h. A single
4-5 h continuous window per camera therefore does not exist. The unit is thus a
per-camera set of continuous BLOCKS totalling 4.0-4.5 h.

R1 Camera eligibility. A camera is eligible iff (a) >= 90 % of its clips belong to a numbered
   camera set in the official clip table (excludes IR cameras G474/G475/G476/G479 and the
   'skip'/PTZ camera G339); (b) label coverage (fraction of clips with an activities.yml file)
   >= 0.50; (c) present on >= 6 recording dates.
R2 Blocks. Clips of one camera are sorted by start datetime. Consecutive clips i, i+1 are in
   the same block iff -30 s <= start(i+1) - end(i) <= 330 s ("tiny gap" = at most one missing
   5-min slot; every such gap is listed). Blocks with < 6 clips (< 30 min) are discarded.
   Block label coverage = fraction of its clips with an activities.yml file. A block is
   "quiet" iff >= 50 % of its clips start before 10:00 or at/after 17:00 local time.
R3 Camera ranking. Eligible cameras are ranked by (label coverage desc, labeled hours desc,
   camera id asc).
R4 Primary cameras (5). Pass 1: best-ranked camera of each site (admin, bus, hospital, school).
   Pass 2: the next-ranked camera overall such that its site has < 2 selected cameras.
   Throughout: at most ONE camera per official camera set (avoids near-duplicate views).
R5 Per-camera block selection. Candidate blocks = R2 blocks with block label coverage >= 0.80.
   Ranking key: (length in clips desc, block label coverage desc, date asc, start asc).
   Step a: take the best-ranked quiet candidate block if one exists (quiet-hour inclusion).
   Step b: walk the ranking and add each block whose addition keeps the camera total
   <= 4.5 h; stop as soon as the total is >= 4.0 h. A block that would overshoot 4.5 h is
   skipped (not truncated) and the walk continues. If a camera cannot reach 4.0 h it keeps
   what it has and this is reported.
R6 Reserve. Reserve cameras = the next 3 cameras in the R3 ranking that satisfy the
   one-camera-per-camera-set constraint w.r.t. all cameras already chosen (site cap not
   applied), each filled by R5; ranked in that order (RES-1, RES-2, RES-3). In addition, each
   primary camera's unselected R5 candidate blocks, in R5 order, form its "substitute" list.
R7 Expansion condition (pre-registered; the reserve is used ONLY if triggered):
   E1 clip-level: a primary clip fails integrity (download SHA mismatch, cv2 cannot open it,
      or decoded frame count < 95 % of duration x fps). If <= 5 % of that camera's selected
      clips fail, the whole failing block is replaced by that camera's first substitute block
      (whole block, to preserve continuity).
   E2 camera-level: if > 5 % of a primary camera's clips fail E1, the camera is replaced by
      the highest-ranked reserve camera from the same site, else by RES-1, RES-2, ... in order.
   E3 volume: if the usable primary total after E1/E2 is < 18.0 h, reserve cameras are
      added in order RES-1, RES-2, ... until >= 20.0 h.
   Nothing observed in the video content or annotations (activity counts, motion, scene
   busyness, detector outputs) can trigger expansion.
"""


def load():
    inv = pd.read_csv(os.path.join(RES, "meva_inventory_labeled.csv"))
    inv = inv[inv.is_video].copy()
    inv["t0"] = pd.to_datetime(inv.date + " " + inv.start)
    inv["t1"] = inv.t0 + pd.to_timedelta(inv.dur_s, unit="s")
    ct = pd.read_csv(os.path.join(META, "meva-clip-camera-and-time-table.txt"), sep=r"\s+", header=None,
                     usecols=[0, 3], names=["clip_id", "camera_set"])
    ct = ct.drop_duplicates("clip_id")
    inv = inv.merge(ct, on="clip_id", how="left")
    inv["camera_set"] = inv.camera_set.fillna("missing")
    st = pd.read_csv(os.path.join(META, "meva-camera-daily-status.txt"), sep="|")
    st.columns = ["camera", "date", "intext"]
    inv = inv.merge(st, on=["camera", "date"], how="left")
    return inv.sort_values(["camera", "t0", "key"]).reset_index(drop=True)


def cameras(inv):
    numbered = inv.camera_set.str.match(r"^\d+-\d+$")
    inv = inv.assign(numbered=numbered)
    g = inv.groupby("camera")
    c = pd.DataFrame({
        "site": g.site.agg(lambda s: s.mode().iloc[0]),
        "n_clips": g.size(),
        "frac_labeled": g.has_activity_labels.mean(),
        "labeled_hours": g.apply(lambda d: d.loc[d.has_activity_labels, "dur_s"].sum() / 3600, include_groups=False),
        "frac_numbered_set": g.numbered.mean(),
        "camera_set": g.camera_set.agg(lambda s: s[s.str.match(r"^\d+-\d+$")].mode().iloc[0]
                                        if s.str.match(r"^\d+-\d+$").any() else s.mode().iloc[0]),
        "intext": g.intext.agg(lambda s: s.mode().iloc[0] if s.notna().any() else ""),
        "n_dates": g.date.nunique(),
    }).reset_index()
    c["eligible"] = ((c.frac_numbered_set >= CAM_SET_FRAC) & (c.frac_labeled >= CAM_LABEL_MIN)
                     & (c.n_dates >= CAM_MIN_DATES))
    c = c.sort_values(["eligible", "frac_labeled", "labeled_hours", "camera"],
                      ascending=[False, False, False, True]).reset_index(drop=True)
    c["cam_rank"] = np.where(c.eligible, np.arange(1, len(c) + 1), np.nan)
    return c


def blocks(inv):
    rows, clip_block = [], {}
    for cam, g in inv.groupby("camera", sort=True):
        g = g.sort_values(["t0", "key"])
        gap = (g.t0 - g.t1.shift()).dt.total_seconds()
        new = gap.isna() | (gap > GAP_MAX_S) | (gap < OVERLAP_MIN_S)
        bid = new.cumsum()
        for b, gg in g.groupby(bid):
            gaps = gap.loc[gg.index[1:]]
            tiny = [(r.start, int(x)) for r, x in zip(gg.iloc[1:].itertuples(), gaps) if x > 60]
            block_id = f"{cam}_{gg.date.iloc[0]}_{gg.t0.iloc[0].strftime('%H%M%S')}"
            for k in gg.key:
                clip_block[k] = block_id
            rows.append(dict(block_id=block_id, camera=cam, site=gg.site.iloc[0], date=gg.date.iloc[0],
                             start=gg.t0.iloc[0].strftime("%H:%M:%S"), end=gg.t1.iloc[-1].strftime("%H:%M:%S"),
                             t0=gg.t0.iloc[0], n_clips=len(gg), hours=gg.dur_s.sum() / 3600,
                             label_cov=gg.has_activity_labels.mean(),
                             quiet=np.mean([QUIET(t) for t in gg.t0]) >= 0.5,
                             gb=gg.size_bytes.sum() / 1e9,
                             tiny_gaps=";".join(f"before {s} gap {x}s" for s, x in tiny)))
    b = pd.DataFrame(rows)
    return b, clip_block


def fill_camera(bc):
    """R5 for one camera's blocks dataframe. Returns (selected block ids, substitute ids)."""
    cand = bc[(bc.n_clips >= MIN_BLOCK_CLIPS) & (bc.label_cov >= BLOCK_LABEL_MIN)].copy()
    cand = cand.sort_values(["n_clips", "label_cov", "date", "start"],
                            ascending=[False, False, True, True])
    sel, tot = [], 0.0
    q = cand[cand.quiet]
    if len(q):
        r = q.iloc[0]
        if r.hours <= H_MAX:
            sel.append(r.block_id)
            tot += r.hours
    for r in cand.itertuples():
        if tot >= H_MIN:
            break
        if r.block_id in sel:
            continue
        if tot + r.hours <= H_MAX + 1e-9:
            sel.append(r.block_id)
            tot += r.hours
    subs = [x for x in cand.block_id if x not in sel]
    return sel, subs, tot


def main():
    inv = load()
    cams = cameras(inv)
    blk, clip_block = blocks(inv)
    inv["block_id"] = inv.key.map(clip_block)
    cams.to_csv(os.path.join(RES, "meva_selection_camera_ranking.csv"), index=False)

    el = cams[cams.eligible]
    primary, used_sets = [], set()
    for site in sorted(el.site.unique()):  # pass 1: best per site
        for r in el[el.site == site].itertuples():
            if r.camera_set not in used_sets:
                primary.append(r.camera); used_sets.add(r.camera_set); break
    primary = sorted(primary, key=lambda c: int(el.loc[el.camera == c, "cam_rank"].iloc[0]))
    for r in el.itertuples():  # pass 2
        if len(primary) >= N_PRIMARY:
            break
        n_site = sum(el.loc[el.camera == c, "site"].iloc[0] == r.site for c in primary)
        if r.camera not in primary and r.camera_set not in used_sets and n_site < SITE_MAX:
            primary.append(r.camera); used_sets.add(r.camera_set)
    reserve = []
    for r in el.itertuples():
        if len(reserve) >= N_RESERVE_CAMS:
            break
        if r.camera not in primary and r.camera_set not in used_sets:
            reserve.append(r.camera); used_sets.add(r.camera_set)

    blk_rows, clip_rows = [], []
    for tier, camlist in (("PRIMARY", primary), ("RESERVE", reserve)):
        for ci, cam in enumerate(camlist, 1):
            sel, subs, tot = fill_camera(blk[blk.camera == cam])
            tag = f"{'P' if tier == 'PRIMARY' else 'RES'}-{ci}"
            groups = [("selected", sel)] + ([("substitute", subs)] if tier == "PRIMARY" else [])
            for role, ids in groups:
                for bi, b_id in enumerate(ids, 1):
                    b = blk[blk.block_id == b_id].iloc[0]
                    blk_rows.append(dict(tier=tier if role == "selected" else "SUBSTITUTE", cam_tag=tag,
                                         block_rank=bi, **b.drop("t0").to_dict()))
                    for c in inv[inv.block_id == b_id].itertuples():
                        clip_rows.append(dict(tier=tier if role == "selected" else "SUBSTITUTE", cam_tag=tag,
                                              camera=cam, site=c.site, camera_set=c.camera_set, intext=c.intext,
                                              block_id=b_id, block_rank=bi, date=c.date, start=c.start,
                                              end=c.end, dur_s=c.dur_s, has_activity_labels=c.has_activity_labels,
                                              size_bytes=c.size_bytes, key=c.key))
    B = pd.DataFrame(blk_rows)
    C = pd.DataFrame(clip_rows)
    C["order"] = range(len(C))
    B.to_csv(os.path.join(RES, "meva_selection_blocks.csv"), index=False)
    C.drop(columns="order").to_csv(os.path.join(RES, "meva_selection.csv"), index=False)
    write_doc(cams, B, C, primary, reserve)


def fmt_h(x):
    return f"{x:.2f}"


def write_doc(cams, B, C, primary, reserve):
    L = ["# MEVA 20 h selection (Topic 09)", "",
         "Generated by `code/meva_select.py` (deterministic, metadata only). Source bucket prefix: "
         "`s3://mevadata-public-01/drops-123-r13/`. Machine-readable: `results/meva_selection.csv` (per clip), "
         "`results/meva_selection_blocks.csv` (per block), `results/meva_selection_camera_ranking.csv`.", "",
         RULE_TEXT, "## 2. Camera ranking (R1/R3)", "",
         "| rank | camera | site | camera_set | int/ext | label cov | labeled h | eligible |",
         "|---|---|---|---|---|---|---|---|"]
    for r in cams.itertuples():
        rk = "" if np.isnan(r.cam_rank) else int(r.cam_rank)
        L.append(f"| {rk} | {r.camera} | {r.site} | {r.camera_set} | {r.intext} | {r.frac_labeled:.3f} | "
                 f"{r.labeled_hours:.2f} | {r.eligible} |")
    for tier in ("PRIMARY", "RESERVE"):
        Ct = C[C.tier == tier]
        Bt = B[B.tier == tier]
        L += ["", f"## {'3' if tier == 'PRIMARY' else '4'}. {tier} list", "",
              f"Total: {Ct.camera.nunique()} cameras, {len(Ct)} clips, {Ct.dur_s.sum()/3600:.2f} h, "
              f"{Ct.size_bytes.sum()/1e9:.2f} GB (from inventory sizes), label coverage "
              f"{Ct.has_activity_labels.mean():.3f}, quiet-hour clips "
              f"{np.mean([QUIET(pd.Timestamp(s)) for s in Ct.start]):.3f}.", "",
              "| tag | camera | site | set | date | start-end | n clips | h | GB | label cov | quiet | tiny gaps |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in Bt.itertuples():
            L.append(f"| {r.cam_tag} | {r.camera} | {r.site} | "
                     f"{C.loc[C.block_id == r.block_id, 'camera_set'].iloc[0]} | {r.date} | {r.start}-{r.end} | "
                     f"{r.n_clips} | {r.hours:.2f} | {r.gb:.2f} | {r.label_cov:.2f} | {r.quiet} | {r.tiny_gaps or '-'} |")
        per = Ct.groupby(["cam_tag", "camera"]).agg(n=("key", "size"), h=("dur_s", lambda s: s.sum() / 3600),
                                                     gb=("size_bytes", lambda s: s.sum() / 1e9)).reset_index()
        L += ["", "Per camera: " + "; ".join(f"{r.cam_tag} {r.camera}: {r.n} clips, {r.h:.2f} h, {r.gb:.2f} GB"
                                              for r in per.itertuples()), ""]
        L += [f"<details><summary>{tier} keys ({len(Ct)})</summary>", "", "```"] + list(Ct.key) + ["```", "</details>"]
    S = B[B.tier == "SUBSTITUTE"]
    L += ["", "## 5. Substitute blocks for primary cameras (R6, used only under E1)", "",
          "| tag | camera | rank | date | start-end | n clips | h | label cov |", "|---|---|---|---|---|---|---|---|"]
    for r in S.itertuples():
        L.append(f"| {r.cam_tag} | {r.camera} | {r.block_rank} | {r.date} | {r.start}-{r.end} | {r.n_clips} | "
                 f"{r.hours:.2f} | {r.label_cov:.2f} |")
    L.append("")
    path = os.path.join(DOCS, "MEVA_selection.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{h}  MEVA_selection.md\n")
    P = C[C.tier == "PRIMARY"]; R = C[C.tier == "RESERVE"]
    print("primary:", primary, f"{P.dur_s.sum()/3600:.2f} h {P.size_bytes.sum()/1e9:.2f} GB")
    print("reserve:", reserve, f"{R.dur_s.sum()/3600:.2f} h {R.size_bytes.sum()/1e9:.2f} GB")
    print(B.drop(columns=["tiny_gaps"]).round(2).to_string())
    print("sha256", h)


if __name__ == "__main__":
    main()
