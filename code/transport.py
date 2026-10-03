"""Transport estimator, uncertainty and bounds (Theorem 1, docs/theory_notes.md §3; STRATA_SPEC v2 §7).

Shared by the pre-registration (code/prereg_build.py) and the verification (code/e2_e5.py, code/e6_sensitivity.py)
so that predictions and evaluation use the same code path.

Objects
  POLICIES     7 policy cells: P1 K in {2,5,10,30} (D3.4), P2, P3, P4 at the frozen operating points
  METRICS      processed_frac, frame_recall, event_recall, false_open_per_frame, precision (num/den columns)
  Bench        benchmark side: per video x cell x policy sums of the additive window columns + window moments
  units()      MEVA analysis units: camera (5), session (3), all, and the pooling-dependent block(s)
Estimators (for one unit, one policy, one metric)
  naive       pooled benchmark ratio over ALL main benchmark windows (incl. benchmark windows in U)
  transport   M = sum_h w_T(h) nu_B(h) / sum_h w_T(h) delta_B(h) over supported cells (h != U), w_T from the
              unit's transportable frames (pooling-dependent blocks removed, D3.2)
  PI          M +- t_{0.975, df} sqrt(v_boot + v_obs), df = n_sessions - 1 = 2 (D3.6), for the OBSERVED value M_obs
              (theory §3 item 3; the form validated in synthetic_check.py "COND", 1.96 replaced by t_2):
              v_boot: benchmark-video cluster bootstrap with w_T fixed at the observed profile, B = 1000, seed 42
              v_obs : sum_h nwin_T(h) Var_B(N - M Q | h) / Dhat^2, Dhat = transported denominator count
  CI_twoway   percentile CI for M_T(w_T): benchmark videos x target camera x session two-way ('all'), the unit's
              blocks for camera / session units (B = 1000, seed 42)
  CI_block    percentile CI with the unit's blocks resampled (secondary, D3.6)
  bounds      Theorem 1(b) closed forms on ALL frames of the unit (incl. U and pooling-dependent blocks):
              supported cells plug-in, U windows by per-window logical boxes; "oracle" variant replaces the
              plug-in by the observed supported part (must contain the observed value: E4 check)
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd
from scipy import stats

from policies import MCOLS, METRICS

SEED = 42
B = 1000
FPS_T = 30.0
POLICIES = [("P1", 2), ("P1", 5), ("P1", 10), ("P1", 30), ("P2", 0), ("P3", 0), ("P4", 0)]
PNAME = {pk: (f"P1_K{pk[1]}" if pk[0] == "P1" else pk[0]) for pk in POLICIES}
MET = list(METRICS)                                   # 5 metrics
CI = {c: i for i, c in enumerate(MCOLS)}               # n, P, TP, E, n_ev, ev_caught, n_fo
DF = 2
TCRIT = float(stats.t.ppf(0.975, DF))


def pidx(policy: str, K: int) -> int:
    return POLICIES.index((policy, int(K)))


# ------------------------------------------------------------------ benchmark side
class Bench:
    """units: benchmark window rows (one definition, main arm) with columns video_uid, policy, K, MCOLS and
    `cell` (cell id or 'U'). Only the 7 policy cells are used."""

    def __init__(self, units: pd.DataFrame):
        u = units[[(p, int(k)) in POLICIES for p, k in zip(units.policy, units.K)]].copy()
        u["pi"] = [pidx(p, k) for p, k in zip(u.policy, u.K)]
        self.videos = sorted(u.video_uid.unique())
        self.cells = sorted(c for c in u.cell.unique() if c != "U")
        hix = {c: i for i, c in enumerate(self.cells)}
        u["hi"] = u.cell.map(lambda c: hix.get(c, len(self.cells)))     # U -> last index
        u["vi"] = u.video_uid.map({v: i for i, v in enumerate(self.videos)})
        V, H, P, C = len(self.videos), len(self.cells) + 1, len(POLICIES), len(MCOLS)
        S = np.zeros((V, H, P, C))
        np.add.at(S, (u.vi.values, u.hi.values, u.pi.values), u[MCOLS].values)
        self.S = S
        self.win = u                                                   # window rows for moments
        # per cell: number of windows / videos (support info), from one policy's rows
        w0 = u[u.pi == 0]
        self.support = w0.groupby("cell").agg(n_win=("cell", "size"), n_vid=("video_uid", "nunique"),
                                              frames=("n", "sum"))
        tot = w0.groupby("cell").n.sum()
        self.w_B = (tot / tot.sum())                                    # incl. U
        sup = tot[tot.index != "U"]
        self.w_B_supp = (sup / sup.sum()).reindex(self.cells).values

    # nu, delta per supported cell for (policy index, metric) from a sums array S (V,H,P,C) or its total
    @staticmethod
    def nu_de(Stot: np.ndarray, p: int, met: str):
        num, den = METRICS[met]
        n = Stot[:-1, p, CI["n"]]
        with np.errstate(invalid="ignore", divide="ignore"):
            return Stot[:-1, p, CI[num]] / n, Stot[:-1, p, CI[den]] / n

    def naive(self, p: int, met: str, Stot=None) -> float:
        Stot = self.S.sum(0) if Stot is None else Stot
        num, den = METRICS[met]
        return float(Stot[:, p, CI[num]].sum() / Stot[:, p, CI[den]].sum())

    def resid_var(self, p: int, met: str, M: float) -> np.ndarray:
        """Var_B(N - M Q | h) per supported cell (ddof 1) from benchmark windows."""
        num, den = METRICS[met]
        x = self.win[(self.win.pi == p) & (self.win.cell != "U")]
        r = x[num].values - M * x[den].values
        s = pd.Series(r).groupby(x.cell.values).var(ddof=1)
        return s.reindex(self.cells).fillna(0.0).values


# ------------------------------------------------------------------ target side
def units(tw: pd.DataFrame, pooling_dep: set) -> list[dict]:
    """Analysis units from the target window table tw (columns block_id, cam_tag, session, ...).
    camera / session / all units exclude pooling-dependent blocks from the TRANSPORTABLE part (D3.2);
    every pooling-dependent block is also its own unit (reported separately)."""
    blk = tw.drop_duplicates("block_id")[["block_id", "cam_tag", "session"]].sort_values("block_id")
    out = []
    for c in sorted(blk.cam_tag.unique()):
        b = blk[blk.cam_tag == c].block_id.tolist()
        out.append(dict(level="camera", unit=c, blocks_all=b, blocks=[x for x in b if x not in pooling_dep]))
    for s in sorted(blk.session.unique()):
        b = blk[blk.session == s].block_id.tolist()
        out.append(dict(level="session", unit=s, blocks_all=b, blocks=[x for x in b if x not in pooling_dep]))
    b = blk.block_id.tolist()
    out.append(dict(level="all", unit="ALL", blocks_all=b, blocks=[x for x in b if x not in pooling_dep]))
    for x in sorted(pooling_dep):
        out.append(dict(level="pooling_dependent_block", unit=x, blocks_all=[x], blocks=[x]))
    meta = blk.set_index("block_id")
    for u in out:
        u["cams"] = {b: meta.cam_tag[b] for b in u["blocks"]}
        u["sess"] = {b: meta.session[b] for b in u["blocks"]}
    return out


class Target:
    """tw: target windows (one definition) with columns block_id, cam_tag, session, w_start, n, E, n_ev, cell."""

    def __init__(self, tw: pd.DataFrame, cells: list[str]):
        self.tw = tw
        self.cells = cells
        self.blocks = sorted(tw.block_id.unique())
        hix = {c: i for i, c in enumerate(cells)}
        hi = tw.cell.map(lambda c: hix.get(c, len(cells))).values
        bi = tw.block_id.map({b: i for i, b in enumerate(self.blocks)}).values
        Bn, H = len(self.blocks), len(cells) + 1
        self.F = np.zeros((Bn, H))                      # frames per block x cell (U last)
        self.NW = np.zeros((Bn, H))                     # windows per block x cell
        self.E = np.zeros((Bn, H))
        self.NEV = np.zeros((Bn, H))
        np.add.at(self.F, (bi, hi), tw.n.values)
        np.add.at(self.NW, (bi, hi), 1)
        np.add.at(self.E, (bi, hi), tw.E.values)
        np.add.at(self.NEV, (bi, hi), tw.n_ev.values)
        self.bix = {b: i for i, b in enumerate(self.blocks)}

    def wvec(self, blocks, weights=None):
        """Frame counts per cell (incl. U last) for a weighted multiset of blocks."""
        idx = [self.bix[b] for b in blocks]
        wts = np.ones(len(idx)) if weights is None else np.asarray(weights, float)
        return wts @ self.F[idx]


def transport_point(nu, de, F_supp):
    w = F_supp / F_supp.sum()
    return float((w @ nu) / (w @ de))


def _target_resamples(u: dict, rng: np.random.Generator, mode: str):
    """Yield (blocks, weights) for one bootstrap replicate of the unit's transportable blocks.
    mode 'twoway': camera x session pigeonhole for 'all'; blocks otherwise. mode 'block': blocks."""
    b = u["blocks"]
    if mode == "cond" or len(b) == 1:                   # conditional on the observed profile w_T
        return b, np.ones(len(b))
    if mode == "twoway" and u["level"] == "all":
        cams = sorted(set(u["cams"].values()))
        sess = sorted(set(u["sess"].values()))
        cc = pd.Series(rng.choice(cams, len(cams))).value_counts()
        ss = pd.Series(rng.choice(sess, len(sess))).value_counts()
        wts = np.array([cc.get(u["cams"][x], 0) * ss.get(u["sess"][x], 0) for x in b], float)
        if wts.sum() == 0:
            return None
        return b, wts
    pick = rng.choice(len(b), len(b))
    return b, np.bincount(pick, minlength=len(b)).astype(float)


def bootstrap_unit(bench: Bench, tgt: Target, u: dict, mode: str, seed: int = SEED, nboot: int = B):
    """Joint bootstrap: benchmark video counts x target block weights. Returns array (nboot, P, M) of
    transport estimates (NaN where undefined) for every policy x metric."""
    rng = np.random.default_rng(seed)
    V = len(bench.videos)
    out = np.full((nboot, len(POLICIES), len(MET)), np.nan)
    for r in range(nboot):
        cv = np.bincount(rng.integers(0, V, V), minlength=V).astype(float)
        Stot = np.tensordot(cv, bench.S, axes=(0, 0))
        res = _target_resamples(u, rng, mode)
        if res is None:
            continue
        F = tgt.wvec(*res)[:-1]
        if F.sum() == 0:
            continue
        for p in range(len(POLICIES)):
            for j, met in enumerate(MET):
                nu, de = Bench.nu_de(Stot, p, met)
                ok = F > 0
                if np.any(~np.isfinite(nu[ok])):              # a resample lost a cell the unit needs
                    continue
                d = F[ok] @ de[ok]
                if d > 0:
                    out[r, p, j] = (F[ok] @ nu[ok]) / d
    return out


def v_obs(bench: Bench, tgt: Target, u: dict, p: int, met: str, M: float, de_B: np.ndarray) -> float:
    """Within-stratum sampling variance of the observed target ratio (theory §3, 'M_obs'), exactly as validated in
    code/synthetic_check.py (COND + predictive term): sum_h nwin_T(h) Var_B(N - M Q | h) / Dhat^2 with the
    TRANSPORTED denominator count Dhat = sum_h F_T(h) delta_B(h) (profile w_T only, D3.7)."""
    idx = [tgt.bix[b] for b in u["blocks"]]
    nw = tgt.NW[idx].sum(0)[:-1]
    s2 = bench.resid_var(p, met, M)
    Q = float(tgt.F[idx].sum(0)[:-1] @ np.nan_to_num(de_B))
    return float(nw @ s2 / Q ** 2) if Q > 0 else float("nan")


# ------------------------------------------------------------------ Theorem 1(b) bounds on all frames of a unit
def boxes(policy: str, K: int, n: np.ndarray, w_start: np.ndarray):
    """Per-window logical boxes (per frame): processed [a, b], false-open upper bound."""
    n = n.astype(float)
    if policy == "P1":
        exact = (n / K + (w_start == 0) * (K - 1) / K) / n       # phase-averaged count, frame 0 always processed
        return exact, exact, exact                               # P1 runs: fo <= processed
    a = np.floor(n / 16) / n                                     # backstop Kr = 15: >= 1 frame in every 16
    return a, np.ones_like(n), np.ceil(n / 2) / n


def bounds_unit(nu, de, tw_unit: pd.DataFrame, cells: list[str], policy: str, K: int, met: str,
                observed_supp: tuple | None = None):
    """Sharp Theorem 1(b) bounds over ALL windows of a unit (rule v2). Supported windows enter through the
    plug-in (nu, de per cell) or, if observed_supp = (N_obs, D_obs) is given, through the observed supported
    sums ('oracle'). U windows enter through per-window boxes. Returns (L, U, omega, omega_E)."""
    N_all = tw_unit.n.sum()
    sup = tw_unit[tw_unit.cell != "U"]
    Uw = tw_unit[tw_unit.cell == "U"]
    F = sup.groupby("cell").n.sum().reindex(cells).fillna(0).values
    if observed_supp is None:
        ok = F > 0
        NS = float(F[ok] @ nu[ok]) / N_all
        DS = float(F[ok] @ de[ok]) / N_all
    else:
        NS, DS = observed_supp[0] / N_all, observed_supp[1] / N_all
    omega = Uw.n.sum() / N_all
    wU = Uw.n.values / N_all
    a, b, fo_b = boxes(policy, K, Uw.n.values, Uw.w_start.values) if len(Uw) else (np.zeros(0),) * 3
    num, den = METRICS[met]
    if met == "processed_frac":
        return NS + wU @ a, NS + wU @ b, omega, omega
    if met == "false_open_per_frame":
        return NS, NS + wU @ fo_b, omega, omega
    if met in ("frame_recall", "event_recall"):
        e = (Uw.E.values if met == "frame_recall" else Uw.n_ev.values) / Uw.n.values if len(Uw) else np.zeros(0)
        D = DS + wU @ e
        omE = float(wU @ e) / D if D > 0 else float("nan")
        return NS / D, (NS + wU @ e) / D, omega, omE
    # precision: x = TP in [0, min(y, e)], y = processed in [a, b]
    e = Uw.E.values / Uw.n.values if len(Uw) else np.zeros(0)
    ys = np.clip(e, a, b)
    Ub = (NS + wU @ np.minimum(e, ys)) / (DS + wU @ ys)
    Lb = NS / (DS + wU @ b)
    return Lb, Ub, omega, omega


def simple_bounds(M_supp: float, omega: float, met: str):
    """PREREG_DECISIONS P-2 form: [(1 - w) M + w*0, (1 - w) M + w*1] (metric range [0, 1])."""
    return (1 - omega) * M_supp, (1 - omega) * M_supp + omega


# ------------------------------------------------------------------ misc
def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def per_hour(x: float) -> float:
    return x * FPS_T * 3600 if x is not None and not (isinstance(x, float) and math.isnan(x)) else x
