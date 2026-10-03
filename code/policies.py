"""Illustrative policies P1-P4 and per-window additive metrics (docs/STRATA_SPEC.md §6-7).

All policies restart at the start of each segment and always process frame 0 of the segment.
Policies see only the causal cue (and, for P3, g_t at frames they processed).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit

KR = 15          # backstop: after 15 consecutive skipped frames the next one is processed
HOLD = 4         # P3: keep processing 4 frames after a processed frame with g = 1


@njit(cache=True)
def _thresh(score, tau, g, hold):
    n = len(score)
    p = np.zeros(n, np.int8)
    gap, h = 0, 0
    for t in range(n):
        take = t == 0 or gap >= KR or h > 0 or score[t] > tau
        if take:
            p[t] = 1
            gap = 0
            h = h - 1 if h > 0 else 0
            if hold > 0 and g[t] == 1:
                h = hold
        else:
            gap += 1
    return p


def p1(n: int, K: int, phase: int) -> np.ndarray:
    p = ((np.arange(n) - phase) % K == 0).astype(np.int8)
    p[0] = 1
    return p


def p2(dn, tau, g):
    return _thresh(np.nan_to_num(np.asarray(dn, float), nan=np.inf), float(tau), np.asarray(g, np.int8), 0)


def p3(dn, tau, g):
    return _thresh(np.nan_to_num(np.asarray(dn, float), nan=np.inf), float(tau), np.asarray(g, np.int8), HOLD)


def p4(pihat, tau, g):
    return _thresh(np.nan_to_num(np.asarray(pihat, float), nan=np.inf), float(tau), np.asarray(g, np.int8), 0)


# ------------------------------------------------------------------ P4 features (causal)
def p4_features(dn, fn) -> np.ndarray:
    d = pd.Series(np.asarray(dn, float)).fillna(1.0).clip(lower=1e-3)
    f = pd.Series(np.asarray(fn, float)).fillna(1.0).clip(lower=1e-3)
    return np.column_stack([np.log(d), np.log(f),
                            np.log(d.rolling(5, min_periods=1).max()),
                            np.log(d.rolling(30, min_periods=1).mean())])


# ------------------------------------------------------------------ per-window metrics
def window_metrics(p: np.ndarray, g: np.ndarray, win: int = 300) -> np.ndarray:
    """Return array (n_windows, 6): n, P, TP, E, n_onsets, n_onsets_caught, n_false_open.
    (7 columns). Events and processed runs are attributed to the window of their first frame."""
    p = np.asarray(p, np.int8)
    g = np.asarray(g, np.int8)
    n = len(p)
    nw = (n + win - 1) // win
    out = np.zeros((nw, 7))
    widx = np.arange(n) // win
    out[:, 0] = np.bincount(widx, minlength=nw)
    out[:, 1] = np.bincount(widx, weights=p, minlength=nw)
    out[:, 2] = np.bincount(widx, weights=p * g, minlength=nw)
    out[:, 3] = np.bincount(widx, weights=g, minlength=nw)
    # events
    d = np.diff(np.r_[0, g, 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    if len(s):
        cp = np.r_[0, np.cumsum(p)]
        caught = (cp[e] - cp[s]) > 0
        out[:, 4] = np.bincount(s // win, minlength=nw)
        out[:, 5] = np.bincount(s // win, weights=caught, minlength=nw)
    # processed runs without any event frame
    dp = np.diff(np.r_[0, p, 0])
    ps, pe = np.where(dp == 1)[0], np.where(dp == -1)[0]
    if len(ps):
        cg = np.r_[0, np.cumsum(g)]
        fo = (cg[pe] - cg[ps]) == 0
        out[:, 6] = np.bincount(ps // win, weights=fo, minlength=nw)
    return out


MCOLS = ["n", "P", "TP", "E", "n_ev", "ev_caught", "n_fo"]

# metric -> (numerator column, denominator column); false-open per frame (x fps*3600 for /hour)
METRICS = {"processed_frac": ("P", "n"), "frame_recall": ("TP", "E"),
           "event_recall": ("ev_caught", "n_ev"), "false_open_per_frame": ("n_fo", "n"),
           "precision": ("TP", "P")}
