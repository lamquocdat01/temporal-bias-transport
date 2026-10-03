"""MEVA OUTCOMES (policies on MEVA, m_T(h), E2-E6) -- LOCKED until pre-registration (Dot 3).

Refuses to run unless ALL hold:
  1. prereg/PREREG_MEVA.json and prereg/PREREG_MEVA.json.sha256 exist;
  2. the SHA-256 of the JSON equals the hash recorded in the .sha256 file;
  3. both files are tracked by git, committed (no uncommitted changes), and the commit that added
     the JSON is an ancestor of HEAD.
Only after that is any policy run on MEVA frames.

Dot 3 body (P2): the 7 pre-registered policy cells (P1 K in {2,5,10,30}, P2, P3, P4 at the frozen operating
points) plus the P1 appendix range K = 1..30 run on the MEVA S8 cue, definition (b'), restarting at every segment
(STRATA_SPEC v2 §6); window metrics per 300-frame window. Checkpoint per block (cache/meva_units/<block>.parquet),
resumable. Outputs:
  results/meva_units.parquet   one row per (block, segment, window, policy, K): MCOLS + strata key + v2 cell
  results/meva_mT_strata.csv   m_T(h) per v2 cell x policy x metric (scope transportable / all S8)
  results/meva_observed.csv    observed outcome per level (camera x session block, camera, session, all,
                               pooling-dependent block) x scope (transportable / full) x policy x metric
Checks (P2.2): frames per block = sum of n_decoded of its clips = detection frames; processed <= n; every
observed cell has n_frames > 0.
"""
from __future__ import annotations
import variant as V
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRE = V.PRE_JSON
SHA = V.PRE_SHA


def git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)


def check_lock() -> str:
    if not PRE.exists() or not SHA.exists():
        sys.exit("LOCKED: prereg/PREREG_MEVA.json (+ .sha256) not found -- outcomes stay sealed.")
    h = hashlib.sha256(PRE.read_bytes()).hexdigest()
    rec = SHA.read_text().split()[0].strip()
    if h != rec:
        sys.exit(f"LOCKED: hash mismatch (file {h[:12]}..., recorded {rec[:12]}...).")
    for p in (PRE, SHA):
        rel = str(p.relative_to(REPO)).replace("\\", "/")
        if git("ls-files", "--error-unmatch", rel).returncode != 0:
            sys.exit(f"LOCKED: {rel} is not tracked by git.")
        if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
            sys.exit(f"LOCKED: {rel} has uncommitted changes.")
    first = git("log", "--diff-filter=A", "--format=%H", "--", V.PRE_REL).stdout.split()
    if not first or git("merge-base", "--is-ancestor", first[-1], "HEAD").returncode != 0:
        sys.exit("LOCKED: commit adding the pre-registration is not an ancestor of HEAD.")
    return h


def block_units(block_segs, cellmap) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd
    from policies import MCOLS
    from run_policies import policy_windows
    from strata import window_table
    rows = []
    for s in block_segs:
        wt = window_table(s["g_bp"])
        assert wt.n.sum() == s["n"]
        res = policy_windows(s["mad"], s["fpx"], s["g_bp"], Ks=range(1, 31))
        base = wt.assign(block_id=s["block_id"], cam_tag=s["cam_tag"], session=s["session"], seg=s["seg"],
                         win_idx=np.arange(len(wt)), cell=wt.key.map(lambda k: cellmap.get(k, "U")))
        for (pol, K), m in res.items():
            assert (m[:, 1] <= m[:, 0] + 1e-9).all() and np.allclose(m[:, 0], wt.n.values)
            rows.append(base.assign(policy=pol, K=K, **{c: m[:, i] for i, c in enumerate(MCOLS)}))
    return pd.concat(rows, ignore_index=True)


def observed_table(u, pdep):
    import pandas as pd
    from policies import METRICS
    pre = [("P1", K) for K in (2, 5, 10, 30)] + [("P2", 0), ("P3", 0), ("P4", 0)]
    u = u[[(p, k) in pre for p, k in zip(u.policy, u.K)]]
    blk = u.drop_duplicates("block_id").set_index("block_id")[["cam_tag", "session"]]
    levels = [("cell", b, [b]) for b in blk.index]
    levels += [("camera", c, blk.index[blk.cam_tag == c].tolist()) for c in sorted(blk.cam_tag.unique())]
    levels += [("session", s, blk.index[blk.session == s].tolist()) for s in sorted(blk.session.unique())]
    levels += [("all", "ALL", blk.index.tolist())]
    levels += [("pooling_dependent_block", b, [b]) for b in sorted(pdep)]
    rows = []
    for lvl, unit, blocks in levels:
        for scope in ("transportable", "full"):
            bl = blocks if (scope == "full" or lvl == "pooling_dependent_block") else [b for b in blocks if b not in pdep]
            x = u[u.block_id.isin(bl)]
            if scope == "transportable":
                x = x[x.cell != "U"]
            g = x.groupby(["policy", "K"])
            for (pol, K), y in g:
                nfr = float(y.n.sum())
                for met, (num, den) in METRICS.items():
                    N, D = float(y[num].sum()), float(y[den].sum())
                    rows.append(dict(level=lvl, unit=unit, scope=scope, blocks="|".join(bl),
                                     cam_tag=blk.cam_tag.get(unit, unit if lvl == "camera" else ""),
                                     session=blk.session.get(unit, unit if lvl == "session" else ""),
                                     policy=f"P1_K{K}" if pol == "P1" else pol, metric=met, num=N, den=D,
                                     value=N / D if D > 0 else float("nan"), n_frames=nfr,
                                     per_hour=(N / D * 30 * 3600) if (met == "false_open_per_frame" and D > 0) else float("nan")))
    return pd.DataFrame(rows)


def main():
    h = check_lock()
    print(f"pre-registration verified: sha256 {h}")
    import json as _json
    import numpy as np
    import pandas as pd
    from meva_frames import load_segments
    from policies import METRICS
    res = V.RES
    ck = V.CACHE / "meva_units"
    ck.mkdir(parents=True, exist_ok=True)
    smap = pd.read_csv(res / "strata_map_benchmark.csv")
    smap = smap[(smap.definition == "bprime") & (smap.rule == "v2") & (smap.arm == "main")]
    cellmap = dict(zip(smap.key, smap.h))
    pre = _json.loads(PRE.read_text(encoding="utf-8"))
    pdep = set(pre["design"]["pooling_dependent_blocks"])
    segs = load_segments()
    for b in sorted({s["block_id"] for s in segs}):
        f = ck / f"{b}.parquet"
        if f.exists():
            print("resume: skip", b)
            continue
        bu = block_units([s for s in segs if s["block_id"] == b], cellmap)
        tmp = f.with_suffix(".tmp")
        bu.to_parquet(tmp, index=False)
        tmp.replace(f)
        print("done", b, len(bu))
    u = pd.concat([pd.read_parquet(f) for f in sorted(ck.glob("*.parquet"))], ignore_index=True)
    # ---- P2.2 checks
    ndec = {}
    for s in segs:
        for st in s["stems"]:
            ndec[s["block_id"]] = ndec.get(s["block_id"], 0) + int(np.load(
                __import__("paths").MEVA_DIR / "cue" / f"{st}.npz")["n_decoded"])
    one = u[(u.policy == "P2")]
    fb = one.groupby("block_id").n.sum()
    assert all(int(fb[b]) == ndec[b] for b in ndec), "frames processed != n_decoded"
    assert int(fb.sum()) == sum(s["n"] for s in segs) == 1062229
    assert (u.P <= u.n + 1e-9).all() and (u.TP <= u.E + 1e-9).all()
    u.to_parquet(res / "meva_units.parquet", index=False)
    # ---- m_T(h)
    pre_cells = [("P1", K) for K in (2, 5, 10, 30)] + [("P2", 0), ("P3", 0), ("P4", 0)]
    up = u[[(p, k) in pre_cells for p, k in zip(u.policy, u.K)]]
    rows = []
    for scope, x in (("transportable", up[~up.block_id.isin(pdep)]), ("all", up)):
        agg = x.groupby(["cell", "policy", "K"])[["n", "P", "TP", "E", "n_ev", "ev_caught", "n_fo"]].sum()
        agg["n_win"] = x.groupby(["cell", "policy", "K"]).size()
        for met, (num, den) in METRICS.items():
            agg[f"m_{met}"] = agg[num] / agg[den].where(agg[den] > 0)
        rows.append(agg.reset_index().assign(scope=scope))
    pd.concat(rows).to_csv(res / "meva_mT_strata.csv", index=False, float_format="%.6g")
    obs = observed_table(u, pdep)
    assert (obs.n_frames > 0).all()
    obs.to_csv(res / "meva_observed.csv", index=False, float_format="%.6g")
    print(f"units {len(u)} rows; frames {int(fb.sum())} == n_decoded; observed rows {len(obs)}")


if __name__ == "__main__":
    main()
