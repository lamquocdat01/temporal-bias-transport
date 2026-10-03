"""Units u (300-frame windows) and strata h(u) = (S, D, R, C) per docs/STRATA_SPEC.md §2-4.

window_table(g, seg_id) -> DataFrame, one row per window of one segment.
build_strata_map(bench_windows) -> dict raw_key -> merged key (or "U"), from benchmark only (rule v1).
build_strata_cells(bench_windows, rule) -> (map raw_key -> cell id or "U", cells: cell id -> member raw keys)
  rule "v2" (STRATA_SPEC v2, main): pooled; rule "v1": partition merge (E6 sensitivity).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WIN = 300
D_EDGES = (5, 30)                    # none / <=5 / 6-30 / >30
R_EDGES = (300, 1800, 18000)         # <300 / 300-1800 / 1800-18000 / >18000
D_LAB = ["le5", "6-30", "gt30"]
R_LAB = ["lt300", "300-1800", "1800-18k", "gt18k"]
MIN_WIN, MIN_VID = 20, 5


def d_class(d: int) -> int:
    return 0 if d <= D_EDGES[0] else (1 if d <= D_EDGES[1] else 2)


def r_class(r: int) -> int:
    return 0 if r < R_EDGES[0] else (1 if r < R_EDGES[1] else (2 if r < R_EDGES[2] else 3))


def events(g: np.ndarray):
    g = np.asarray(g, np.int8)
    d = np.diff(np.r_[0, g, 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    L = e - s
    cens = (s == 0) | (e == len(g))
    return s, L, cens


def merge_gaps(g: np.ndarray, k: int) -> np.ndarray:
    """Definition (b'): fill idle gaps of length <= k frames that lie strictly between two active
    runs of the same segment (k = round(0.2 s x fps) of the source)."""
    g = np.asarray(g, np.int8).copy()
    if k <= 0:
        return g
    s, L, _ = events(1 - g)
    for a, l in zip(s, L):
        if a > 0 and a + l < len(g) and l <= k:
            g[a:a + l] = 1
    return g


def k_merge(fps: float) -> int:
    return int(round(0.2 * fps))


def window_table(g: np.ndarray, win: int = WIN) -> pd.DataFrame:
    g = np.asarray(g, np.int8)
    n = len(g)
    s, L, cens = events(g)
    e = s + L
    # last active frame strictly before each index: cumulative max of active indices
    idx = np.where(g == 1, np.arange(n), -1)
    last_active = np.maximum.accumulate(idx) if n else idx
    rows = []
    for a in range(0, n, win):
        b = min(n, a + win)
        frac = g[a:b].mean()
        S = "idle" if frac == 0 else ("full" if frac == 1 else "partial")
        ov = (s < b) & (e > a)
        c_flag = bool(cens[ov].any())
        if S == "idle":
            D = "none"
        else:
            Lmax = int(L[ov].max())
            k = d_class(Lmax)
            # a censored overlapping event may be longer than observed -> class only bounded below
            if k < 2 and bool(cens[ov].any()):
                D = f"cens>={D_LAB[k]}"
            else:
                D = D_LAB[k]
        la = last_active[a - 1] if a > 0 else -1
        if la >= 0:
            R = R_LAB[r_class(a - 1 - la)]
        else:
            k = r_class(a)
            if k == 3:
                R = R_LAB[3]
            else:
                R = f"cens>={R_LAB[k]}"
                c_flag = True
        rows.append(dict(w_start=a, n=b - a, S=S, D=D, R=R, C=int(c_flag),
                         key=f"{S}|{D}|{R}|C{int(c_flag)}"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sparse-stratum merging (§4)
def _coarsen(key: str, step: int) -> str:
    S, D, R, C = key.split("|")
    if step >= 1 and D in ("le5", "6-30"):
        D = "le30"
    if step >= 1 and D in ("cens>=le5", "cens>=6-30"):
        D = "cens>=le30"
    if step >= 2 and not R.startswith("cens") and not D.startswith("cens"):
        C = "C0"
    if step >= 3 and R in ("lt300", "300-1800"):
        R = "lt1800"
    return "|".join([S, D, R, C])


# v2 (pooled) coarsening order R -> D -> C: R is the dimension the illustrative policies' memory
# (<= 15 + 4 frames << 300) is least sensitive to; D drives detection probability directly
# (P1: min(1, d/K)); C (censoring) is merged last because it changes what duration is observable.
V2_ORDER = ("R", "D", "C")


def _coarsen_dim(key: str, dims) -> str:
    S, D, R, C = key.split("|")
    if "R" in dims and R in ("lt300", "300-1800"):
        R = "lt1800"
    if "D" in dims:
        D = {"le5": "le30", "6-30": "le30", "cens>=le5": "cens>=le30", "cens>=6-30": "cens>=le30"}.get(D, D)
    if "C" in dims and not R.startswith("cens") and not D.startswith("cens"):
        C = "C0"
    return "|".join([S, D, R, C])


def build_strata_cells(bw: pd.DataFrame, rule: str = "v2"):
    """v2 pooled rule: a supported raw stratum is its own cell. An unsupported raw stratum (>= 1
    benchmark window) is assigned, at the first coarsening step (R, then R+D, then R+D+C) whose
    coarsened cell is supported, to the cell made of ALL benchmark raw strata sharing that coarsened
    key (supported siblings included; they also keep their own cell). Raw strata with no benchmark
    window, or still unsupported after the last step -> U."""
    if rule == "v1":
        mp = build_strata_map(bw)
        cells = {}
        for k, c in mp.items():
            if c != "U":
                cells.setdefault(c, []).append(k)
        return mp, cells
    stats = bw.groupby("key").agg(n_win=("key", "size"), n_vid=("video_uid", "nunique"))
    vids = bw.groupby("key").video_uid.apply(set).to_dict()
    raw = sorted(stats.index)
    ok = lambda ks: (stats.loc[ks, "n_win"].sum() >= MIN_WIN  # noqa: E731
                     and len(set().union(*[vids[k] for k in ks])) >= MIN_VID)
    mp, cells = {}, {}
    for k in raw:
        if ok([k]):
            mp[k] = k
            cells[k] = [k]
            continue
        mp[k] = "U"
        for i in range(1, len(V2_ORDER) + 1):
            dims = V2_ORDER[:i]
            ck = _coarsen_dim(k, dims)
            members = [j for j in raw if _coarsen_dim(j, dims) == ck]
            if ok(members):
                cid = f"P{i}:{ck}"
                mp[k] = cid
                cells[cid] = members
                break
    return mp, cells


def build_strata_map(bw: pd.DataFrame) -> dict:
    """bw: benchmark windows with columns key, video_uid. Returns raw key -> merged key or 'U'."""
    stats = bw.groupby("key").agg(n_win=("key", "size"), n_vid=("video_uid", "nunique"))
    vids = bw.groupby("key").video_uid.apply(set).to_dict()
    ok = lambda nw, nv: nw >= MIN_WIN and nv >= MIN_VID  # noqa: E731
    mapping = {k: k for k, r in stats.iterrows() if ok(r.n_win, r.n_vid)}
    pending = [k for k in stats.index if k not in mapping]
    for step in (1, 2, 3):
        groups: dict[str, list] = {}
        for k in pending:
            groups.setdefault(_coarsen(k, step), []).append(k)
        pending = []
        for ck, members in groups.items():
            nw = int(stats.loc[members, "n_win"].sum())
            nv = len(set().union(*[vids[m] for m in members]))
            if ok(nw, nv):
                for m in members:
                    mapping[m] = "M:" + ck
            else:
                pending += members
    for k in pending:
        mapping[k] = "U"
    return mapping


def apply_map(keys: pd.Series, mapping: dict) -> pd.Series:
    return keys.map(lambda k: mapping.get(k, "U"))
