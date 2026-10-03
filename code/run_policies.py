"""One policy engine for benchmark and MEVA (STRATA_SPEC v2 §5-7): frozen operating points, P4 frozen logistic.

policy_windows(mad, fpx, g, ops, Ks, win, norm_win) -> dict {(policy, K): window metrics array (n_win, 7)}
  cue normalisation per segment (cue.normalise), P1 averaged over phases, P2/P3/P4 at frozen tau,
  P4 pi = sigmoid(features @ coef + intercept) with the frozen coefficients of operating_points.json.
Self-test (no MEVA): python code/run_policies.py --selftest-bench
  recomputes the benchmark windows of the 7 pre-registered policy cells (definition (b'), 131 videos) with this
  engine and asserts equality with results/benchmark_units.parquet.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cue
import variant as V
from policies import p1, p2, p3, p4, p4_features, window_metrics, MCOLS

REPO = Path(__file__).resolve().parents[1]
OPS = json.loads((V.RES / "operating_points.json").read_text())


def normalise(mad, fpx, norm_win: int = 1800):
    if norm_win == cue.NORM_WIN:
        return cue.normalise(mad, fpx)
    old = cue.NORM_WIN
    try:
        cue.NORM_WIN = norm_win
        return cue.normalise(mad, fpx)
    finally:
        cue.NORM_WIN = old


def pihat(dn, fn, ops=OPS):
    X = p4_features(dn, fn)
    z = X @ np.asarray(ops["P4"]["coef"]) + ops["P4"]["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def policy_windows(mad, fpx, g, Ks=(2, 5, 10, 30), win=300, norm_win=1800, ops=OPS, dn_fn=None):
    dn, fn = normalise(mad, fpx, norm_win) if dn_fn is None else dn_fn
    g = np.asarray(g, np.int8)
    out = {("P2", 0): window_metrics(p2(dn, ops["P2"]["tau"], g), g, win),
           ("P3", 0): window_metrics(p3(dn, ops["P3"]["tau"], g), g, win),
           ("P4", 0): window_metrics(p4(pihat(dn, fn, ops), ops["P4"]["tau"], g), g, win)}
    n = len(g)
    for K in Ks:
        out[("P1", K)] = np.mean([window_metrics(p1(n, K, f), g, win) for f in range(K)], axis=0)
    return out


def selftest_bench():
    from bench_units import load_segments
    ref = pd.read_parquet(V.RES / "benchmark_units.parquet")
    ref = ref[ref.definition == "bprime"]
    c = pd.read_parquet(V.CACHE / "bench_cue.parquet")
    segs = load_segments()
    worst, n_cmp, pi_dev = 0.0, 0, 0.0
    from sklearn.linear_model import LogisticRegression  # noqa: F401  (pihat must equal predict_proba)
    for s in segs:
        x = c[(c.dataset == s["dataset"]) & (c.video == s["video"])].sort_values("frame")
        res = policy_windows(x.mad.values, x.fpx.values, s["g"]["bprime"])
        r = ref[ref.video_uid == s["uid"]]
        for (pol, K), m in res.items():
            rr = r[(r.policy == pol) & (r.K == K)].sort_values("win_idx")[MCOLS].values
            assert rr.shape == m.shape, (s["uid"], pol, K)
            worst = max(worst, float(np.abs(rr - m).max()))
            n_cmp += 1
    print(f"selftest: {len(segs)} videos x 7 policy cells = {n_cmp} comparisons, max |diff| = {worst:.3g}")
    assert worst < 1e-9, worst
    return worst


if __name__ == "__main__":
    if "--selftest-bench" in sys.argv:
        selftest_bench()
