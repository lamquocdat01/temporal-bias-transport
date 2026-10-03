"""E1 census: temporal profile of each benchmark video (121 videos + VIRAT-P).

Two event definitions:
  (a) has-box, any class  -> g_t from frame_scores.parquet (verified against detections_cpu)
  (b) person (cls 0) conf >= 0.7 -> rebuilt from the original detections_cpu JSON
      (VIRAT-P: frame_scores_viratp.g_t, verified against detections_cpu_virat_full)
An event is a maximal run of consecutive frames with g=1.
Boundary-touching events (onset = first frame or end = last frame) are treated as
right-censored in the Kaplan-Meier estimate of the duration law.

Outputs (repo/results): E1_profile_benchmark.csv, E1_km_medians.csv, E1_summary.json,
fig_E1_profile.png.
"""
from __future__ import annotations
import paths
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from strata import merge_gaps, k_merge  # (b') = STRATA_SPEC v2
from paths import BMC_DIR, VIRAT_DIR
import variant as V

SEED = 42
np.random.seed(SEED)

PROC = paths.BENCH_PROC
DET = paths.SCM_DIR / "data"
BMC_MP4 = BMC_DIR
VIRAT_MP4 = VIRAT_DIR
REPO = Path(__file__).resolve().parents[1]
OUT = V.RES

BENCH = ["CDnet2014", "LASIESTA", "BMC"]
EXPECT_VIDEOS = {"CDnet2014": 53, "LASIESTA": 48, "BMC": 20}
EXPECT_EVENTS = V.EXPECT_EVENTS_A
EXPECT_FRAMES = 166_589
LOWFR = {"port_0_17fps": 0.17, "tunnelExit_0_35fps": 0.35, "turnpike_0_5fps": 0.5,
         "tramCrossroad_1fps": 1.0}
NOMINAL_FPS = 25.0          # used only where the dataset gives no frame rate
PERSON_CONF = 0.7


# --------------------------------------------------------------------------- helpers
def runs(g: np.ndarray, val: int = 1):
    """Return (starts, lengths) of maximal runs of value `val`."""
    b = (np.asarray(g) == val).astype(np.int8)
    d = np.diff(np.r_[0, b, 0])
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return s, e - s


def load_det(path: Path):
    return {fr["frame"]: fr.get("boxes", []) for fr in json.load(open(path))["frames"]}


def g_from_det(det: dict, frames: np.ndarray, person_conf: float | None):
    out = np.zeros(len(frames), dtype=np.int8)
    for i, f in enumerate(frames):
        boxes = det.get(int(f), [])
        if person_conf is None:
            out[i] = len(boxes) > 0
        else:
            out[i] = any(int(b[5]) == 0 and b[4] >= person_conf for b in boxes)
    return out


def video_fps(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return float(fps) if fps and fps > 0 else None


def km_median(dur: np.ndarray, censored: np.ndarray):
    """Kaplan-Meier median (smallest t with S(t) <= 0.5); None if never reached."""
    dur = np.asarray(dur, float)
    obs = ~np.asarray(censored, bool)
    s = 1.0
    for t in np.unique(dur[obs]):
        n_risk = (dur >= t).sum()
        d = ((dur == t) & obs).sum()
        s *= 1 - d / n_risk
        if s <= 0.5:
            return float(t)
    return None


def profile(g: np.ndarray, fps: float):
    n = len(g)
    s, L = runs(g, 1)
    _, idle = runs(g, 0)
    touch = (s == 0) | (s + L == n)
    onset_cut = s == 0
    q = np.percentile(L, [10, 50, 90]) if len(L) else [np.nan] * 3
    li = int(idle.max()) if len(idle) else 0
    return {
        "duty_cycle": float(g.mean()),
        "n_events": int(len(L)),
        "event_rate_per_min": len(L) / (n / fps / 60.0),
        "dur_q10": q[0], "dur_q50": q[1], "dur_q90": q[2],
        "frac_boundary": float(touch.mean()) if len(L) else np.nan,
        "longest_idle_frames": li,
        "longest_idle_sec": li / fps,
    }, L, touch, onset_cut


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    fs = V.frame_scores()
    n_stub = int((fs.dataset == "VIRAT").sum())
    fs = fs[fs.dataset.isin(BENCH)].copy()          # drop the 5,000 VIRAT stub frames
    ev = V.events_durations()
    n_ev_virat = int((ev.dataset == "VIRAT").sum())
    ev = ev[ev.dataset != "VIRAT"]

    # ---- corpus self-checks
    vids = fs.groupby("dataset").video.nunique().to_dict()
    assert vids == EXPECT_VIDEOS, vids
    evc = ev.groupby("dataset").size().to_dict()
    assert evc == EXPECT_EVENTS, evc
    n_frames = len(fs)
    frames_by_ds = fs.groupby("dataset").size().to_dict()
    checks = {"videos": vids, "events": evc, "frames_total": n_frames,
              "frames_by_dataset": frames_by_ds, "frames_expected": EXPECT_FRAMES,
              "frames_match": n_frames == EXPECT_FRAMES,
              "virat_stub_frames_dropped": n_stub, "virat_rows_in_events_durations": n_ev_virat}

    # event-file video key "cat__name" -> frame_scores name is the last component
    key_of = {}
    for ds, v in ev[["dataset", "video"]].drop_duplicates().itertuples(index=False):
        key_of[(ds, v.split("__")[-1])] = v
    assert len(key_of) == 121

    rows, dur_rows = [], []
    n_g_mismatch = 0
    for (ds, v), grp in fs.groupby(["dataset", "video"], sort=True):
        grp = grp.sort_values("frame")
        fr = grp.frame.values
        assert (np.diff(fr) == 1).all(), (ds, v)
        key = key_of[(ds, v)]
        det = V.bench_det(ds, key)
        g_a = grp.g_t.values.astype(np.int8)
        g_a_det = g_from_det(det, fr, None)
        n_g_mismatch += int((g_a != g_a_det).sum())
        g_b = g_from_det(det, fr, PERSON_CONF)
        # durations must reproduce the committed events table exactly
        _, L = runs(g_a)
        ref = ev[(ev.dataset == ds) & (ev.video == key)].duration.values
        assert sorted(L) == sorted(ref), (ds, v)

        if ds == "BMC":
            split, name = key.split("__")
            fps = video_fps(BMC_MP4 / split / f"{name}.mp4"); src = "mp4"
        elif key.split("__")[-1] in LOWFR:                 # CDnet lowFramerate: fps from the name
            fps, src = LOWFR[key.split("__")[-1]], "name"
        else:
            fps, src = None, None
        if fps is None:
            fps, src = NOMINAL_FPS, "assumed_nominal"
        base = {"dataset": ds, "video": key, "n_frames": len(fr), "fps": fps,
                "fps_source": src, "clip_sec": len(fr) / fps}
        row = dict(base)
        for tag, g in (("a", g_a), ("b", g_b), ("bp", merge_gaps(g_b, k_merge(fps)))):
            p, L, touch, oc = profile(g, fps)
            row.update({f"{k}_{tag}": val for k, val in p.items()})
            dur_rows += [{"dataset": ds, "video": key, "def": tag, "dur": int(x),
                          "censored": bool(c), "onset_cut": bool(o)} for x, c, o in zip(L, touch, oc)]
        rows.append(row)
    assert n_g_mismatch == 0, f"g_t != has-box in {n_g_mismatch} frames"
    checks["g_t_equals_hasbox_all_frames"] = True

    # ---- VIRAT-P (10 videos): (b) = committed g_t, (a) = has-box from virat_full JSON
    vp = pd.read_parquet(PROC / "frame_scores_viratp.parquet")
    vp_mis = 0
    for v, grp in vp.groupby("video", sort=True):
        grp = grp.sort_values("frame")
        fr = grp.frame.values
        cand = list((DET / "detections_cpu_virat_full" / "VIRAT").glob(f"*__{v}.json"))
        assert len(cand) == 1, v
        det = load_det(cand[0])
        g_b = grp.g_t.values.astype(np.int8)
        vp_mis += int((g_b != g_from_det(det, fr, PERSON_CONF)).sum())
        g_a = g_from_det(det, fr, None)
        fps = video_fps(VIRAT_MP4 / f"{v}.mp4") or 30.0
        row = {"dataset": "VIRAT-P", "video": v, "n_frames": len(fr), "fps": fps,
               "fps_source": "mp4", "clip_sec": len(fr) / fps}
        for tag, g in (("a", g_a), ("b", g_b), ("bp", merge_gaps(g_b, k_merge(fps)))):
            p, L, touch, oc = profile(g, fps)
            row.update({f"{k}_{tag}": val for k, val in p.items()})
            dur_rows += [{"dataset": "VIRAT-P", "video": v, "def": tag, "dur": int(x),
                          "censored": bool(c), "onset_cut": bool(o)} for x, c, o in zip(L, touch, oc)]
        rows.append(row)
    checks["viratp_frames"] = int(len(vp))
    checks["viratp_gt_vs_rebuilt_person07_mismatch_frames"] = vp_mis

    prof = pd.DataFrame(rows)
    prof.to_csv(OUT / "E1_profile_benchmark.csv", index=False, float_format="%.6g")
    dur = pd.DataFrame(dur_rows)
    checks["viratp_events_def_b"] = int(((dur.dataset == "VIRAT-P") & (dur["def"] == "b")).sum())

    # ---- Kaplan-Meier vs raw medians
    km = []
    for (ds, tag), d in dur.groupby(["dataset", "def"]):
        km.append({"dataset": ds, "def": tag, "n_events": len(d),
                   "frac_censored": d.censored.mean(),
                   "median_raw": float(np.median(d.dur)),
                   "median_uncensored_only": float(np.median(d.dur[~d.censored])) if (~d.censored).any() else np.nan,
                   "median_KM": km_median(d.dur.values, d.censored.values),
                   # Lemma 3: KM only corrects right-censoring -> drop onset-cut events, censor end-cut ones
                   "n_onset_in": int((~d.onset_cut).sum()),
                   "median_KM_onset_in": km_median(d.dur.values[~d.onset_cut.values],
                                                   d.censored.values[~d.onset_cut.values])})
    km = pd.DataFrame(km)
    km.to_csv(OUT / "E1_km_medians.csv", index=False, float_format="%.4g")

    # ---- headline numbers
    order = BENCH + ["VIRAT-P"]
    summ = {}
    for ds in order:
        p = prof[prof.dataset == ds]
        summ[ds] = {f"median_{k}_{t}": float(p[f"{k}_{t}"].median())
                    for t in ("a", "b", "bp") for k in ("duty_cycle", "longest_idle_sec",
                                                  "longest_idle_frames", "event_rate_per_min")}
        summ[ds]["n_videos"] = int(len(p))
        summ[ds]["median_clip_sec"] = float(p.clip_sec.median())
    allb = prof[prof.dataset.isin(BENCH)]
    summ["ALL121"] = {f"median_{k}_{t}": float(allb[f"{k}_{t}"].median())
                      for t in ("a", "b") for k in ("duty_cycle", "longest_idle_sec")}
    summ["ALL121"]["mean_frames_per_video"] = float(allb.n_frames.mean())
    json.dump({"checks": checks, "summary": summ}, open(OUT / "E1_summary.json", "w"),
              indent=2, default=float)
    make_fig(prof, order)
    print(json.dumps(checks, indent=1, default=float))
    print(km.to_string(index=False))
    print(pd.DataFrame(summ).T.round(3).to_string())


def make_fig(prof: pd.DataFrame, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"CDnet2014": "#2a78d6", "LASIESTA": "#eb6834", "BMC": "#1baf7a", "VIRAT-P": "#eda100"}
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    specs = [("duty_cycle", "Duty cycle (fraction of frames in-event)", False),
             ("longest_idle_sec", "Longest idle run (s, log scale; 0 s drawn at 0.04 s)", True)]
    for r, (tag, dname) in enumerate((("a", "(a) has-box, any class"),
                                      ("b", "(b) person, conf ≥ 0.7"))):
        for c, (k, lab, logx) in enumerate(specs):
            ax = axes[r, c]
            for i, ds in enumerate(order):
                x = prof.loc[prof.dataset == ds, f"{k}_{tag}"].values.astype(float)
                if logx:
                    x = np.maximum(x, 0.04)     # zero-length idle drawn at the floor
                y = i + rng.uniform(-0.18, 0.18, len(x))
                ax.scatter(x, y, s=22, color=col[ds], alpha=0.75, edgecolor="white", linewidth=0.6)
                med = np.median(x)
                ax.plot([med, med], [i - 0.32, i + 0.32], color="#222", lw=2)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([f"{d} (n={int((prof.dataset == d).sum())})" for d in order])
            ax.invert_yaxis()
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(lab)
            ax.set_title(f"{dname}", fontsize=10, loc="left")
            ax.grid(axis="x", color="#ddd", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle("E1 temporal profile of benchmark videos (dots = videos, bar = median)", fontsize=11)
    fig.savefig(OUT / "fig_E1_profile.png", dpi=160)


if __name__ == "__main__":
    main()
