"""Flip radius r* (Proposition 2, docs/theory_notes.md §4) and identified-set bounds helpers.

Ball(r) = { w in simplex : w_h = 0 for h not in `allowed`, TV(w_B, w) = 0.5*||w - w_B||_1 <= r }.
A ranking "policy 1 better than policy 2" (Delta_B = M_1(w_B) - M_2(w_B) > 0) is FLIPPED at w if
M_1(w) - M_2(w) <= 0 (tie counts as flip). r* = inf{ r : some w in Ball(r) flips }.
Ranking preserved for every w in Ball(r)  <=>  r < r*.

Functions
  r_star_linear(delta, w_B, allowed=None)          exact (greedy / water-filling), linear metrics
  r_star_closed_form(delta, w_B, allowed=None)     |Delta_B| / (max delta - min delta): lower bound,
                                                   exact iff w_B(argmax-side) >= that value
  r_star_ratio_shared_den(nu1, nu2, den, w_B, ...) ratio metrics with a policy-independent
                                                   denominator (frame/event recall): exact, reduces
                                                   to r_star_linear(nu1 - nu2) (sign of a single ratio)
  min_ratio_in_ball(nu, den, w_B, r, ...)          exact min/max of a single ratio over Ball(r)
                                                   (Charnes-Cooper LP)
  r_star_ratio_general(nu1, den1, nu2, den2, w_B)  policy-dependent denominators (precision):
                                                   r* = min_t rho(t), rho(t) an exact LP; outer 1-D
                                                   minimisation numerical (grid + refinement). Returned
                                                   value is attained by a verified flipping w.
  r_star_ratio_lower(...)                          certified lower bound for the general case (exact LPs)
  ratio_bounds_lp(...)                             sharp bounds of (N_S + sum_U w x)/(D_S + sum_U w y)
                                                   with per-stratum polytopes for (x, y) (Theorem 1b).
Self-test: python code/flip_radius.py   (seed 42; brute force on random small instances).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

SEED = 42


def _allowed(n, allowed):
    a = np.ones(n, bool) if allowed is None else np.asarray(allowed, bool)
    return a


# ------------------------------------------------------------------ linear metrics
def min_linear_in_ball(delta, w_B, r, allowed=None):
    """Exact min of sum_h w_h delta_h over Ball(r): move up to r mass, taking it from the strata
    with the largest delta first (only where w_B > 0) and putting it on the allowed stratum with
    the smallest delta. Returns (value, w)."""
    delta, w_B = np.asarray(delta, float), np.asarray(w_B, float)
    a = _allowed(len(delta), allowed)
    w = w_B.copy()
    sink = np.flatnonzero(a)[np.argmin(delta[a])]
    budget = r
    for h in np.argsort(-delta):
        if budget <= 0 or delta[h] <= delta[sink]:
            break
        take = min(budget, w[h])
        w[h] -= take
        w[sink] += take
        budget -= take
    return float(w @ delta), w


def r_star_linear(delta, w_B, allowed=None):
    """Exact r* for a linear metric difference delta(h) = m_1(h) - m_2(h).
    min over Ball(r) of Delta(w) = Delta_B - phi(r), phi piecewise linear concave; r* = phi^{-1}(|Delta_B|).
    Returns (r_star, w_at_r_star). r_star = inf if no w in the simplex flips the ranking."""
    delta, w_B = np.asarray(delta, float), np.asarray(w_B, float)
    a = _allowed(len(delta), allowed)
    DB = float(w_B @ delta)
    if DB < 0:                       # orient so that policy 1 is ahead
        delta, DB = -delta, -DB
    if DB == 0:
        return 0.0, w_B.copy()
    dmin = delta[a].min()
    sink = np.flatnonzero(a)[np.argmin(delta[a])]
    need, r, w = DB, 0.0, w_B.copy()
    for h in np.argsort(-delta):
        gain = delta[h] - dmin               # reduction of Delta per unit of mass moved
        if gain <= 0:
            break
        m = w_B[h]
        if m * gain >= need:
            x = need / gain
            w[h] -= x
            w[sink] += x
            return r + x, w
        w[h] -= m
        w[sink] += m
        need -= m * gain
        r += m
    return float("inf"), w


def r_star_closed_form(delta, w_B, allowed=None):
    delta, w_B = np.asarray(delta, float), np.asarray(w_B, float)
    a = _allowed(len(delta), allowed)
    if float(w_B @ delta) < 0:
        delta = -delta
    DB = float(w_B @ delta)
    rng = delta[w_B > 0].max() - delta[a].min()      # remove from support, add to allowed
    return DB / rng if rng > 0 else float("inf")


# ------------------------------------------------------------------ ratio metrics
def r_star_ratio_shared_den(nu1, nu2, den, w_B, allowed=None):
    """R_i(w) = sum w nu_i / sum w den with the SAME den (recall metrics). Since den(w) > 0,
    sign(R_1 - R_2) = sign(sum w (nu1 - nu2)): r* is exactly r_star_linear(nu1 - nu2) provided the
    worst-case w keeps den(w) > 0 (checked)."""
    r, w = r_star_linear(np.asarray(nu1) - np.asarray(nu2), w_B, allowed)
    if np.isfinite(r) and float(w @ np.asarray(den)) <= 0:
        raise ValueError("worst case puts all mass on strata with zero denominator; ratio undefined")
    return r, w


def _ball_lp_parts(w_B, r, allowed):
    """Variables z = [w (H), p (H), q (H)], w = w_B + p - q, sum p = sum q <= r, 0<=q<=w_B, p on allowed."""
    H = len(w_B)
    a = _allowed(H, allowed)
    Aeq = np.zeros((H + 1, 3 * H))
    beq = np.zeros(H + 1)
    Aeq[:H, :H] = np.eye(H)
    Aeq[:H, H:2 * H] = -np.eye(H)
    Aeq[:H, 2 * H:] = np.eye(H)
    beq[:H] = w_B
    Aeq[H, H:2 * H] = 1
    Aeq[H, 2 * H:] = -1
    Aub = np.zeros((1, 3 * H))
    Aub[0, H:2 * H] = 1
    bub = np.array([r])
    bounds = [(0, None) if a[h] else (0, 0) for h in range(H)] + \
             [(0, None) if a[h] else (0, 0) for h in range(H)] + [(0, w_B[h]) for h in range(H)]
    return Aeq, beq, Aub, bub, bounds


def min_ratio_in_ball(nu, den, w_B, r, allowed=None, maximize=False):
    """Exact min (or max) of sum w nu / sum w den over Ball(r), den >= 0 and den(w) > 0 on the ball.
    Charnes-Cooper: y = s*w, s = 1/den(w) >= 0 -> LP in (y, p~, q~, s)."""
    nu, den, w_B = (np.asarray(x, float) for x in (nu, den, w_B))
    H = len(nu)
    a = _allowed(H, allowed)
    # variables: y (H), p (H), q (H), s (1); y = s w_B + p - q ; sum p = sum q <= r s ; q <= s w_B
    n = 3 * H + 1
    c = np.zeros(n)
    c[:H] = -nu if maximize else nu
    Aeq, beq = [], []
    for h in range(H):
        row = np.zeros(n); row[h] = 1; row[H + h] = -1; row[2 * H + h] = 1; row[-1] = -w_B[h]
        Aeq.append(row); beq.append(0)
    row = np.zeros(n); row[H:2 * H] = 1; row[2 * H:3 * H] = -1; Aeq.append(row); beq.append(0)
    row = np.zeros(n); row[:H] = den; Aeq.append(row); beq.append(1)
    Aub, bub = [], []
    row = np.zeros(n); row[H:2 * H] = 1; row[-1] = -r; Aub.append(row); bub.append(0)
    for h in range(H):
        row = np.zeros(n); row[2 * H + h] = 1; row[-1] = -w_B[h]; Aub.append(row); bub.append(0)
    bounds = [(0, None) if a[h] else (0, 0) for h in range(H)] * 2 + [(0, None)] * H + [(0, None)]
    res = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=bounds,
                  method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    y, s = res.x[:H], res.x[-1]
    w = y / s
    return float(nu @ w / (den @ w)), w


def _rho(t, nu1, den1, nu2, den2, w_B, allowed):
    """rho(t) = min TV(w, w_B) s.t. w in simplex (allowed support), N1 - t D1 <= 0, N2 - t D2 >= 0.
    Exact LP in (w, p, q). Returns (rho, w) or (inf, None) if infeasible."""
    H = len(w_B)
    Aeq, beq, _, _, bounds = _ball_lp_parts(w_B, 1.0, allowed)
    Aub = np.vstack([np.r_[nu1 - t * den1, np.zeros(2 * H)],
                     np.r_[-(nu2 - t * den2), np.zeros(2 * H)]])
    c = np.r_[np.zeros(H), np.ones(H), np.zeros(H)]
    res = linprog(c, A_ub=Aub, b_ub=[0, 0], A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        return float("inf"), None
    return float(res.fun), res.x[:H]


def r_star_ratio_general(nu1, den1, nu2, den2, w_B, allowed=None, n_t=201):
    """Policy-dependent denominators (e.g. precision). Returns (r_star, w_flip, info).
    Orientation: policy 1 ahead at w_B. A flip at w means R1(w) <= R2(w)  <=>  exists t with
    N1(w) - t D1(w) <= 0 <= N2(w) - t D2(w) (take t = R1(w)). Hence
        r* = min_t rho(t),   rho(t) = min{ TV(w, w_B) : N1 - t D1 <= 0, N2 - t D2 >= 0 }  (an LP).
    Each rho(t) is exact; the outer 1-D minimisation over t (grid of n_t points over the attainable
    ratio range + bounded scalar refinement around the 3 best grid points) is numerical. Every
    returned value is ACHIEVED by the returned w (verified), so r_star is an exact upper bound on r*;
    it can exceed r* only if rho has a narrow dip missed by the grid."""
    from scipy.optimize import minimize_scalar
    nu1, den1, nu2, den2, w_B = (np.asarray(x, float) for x in (nu1, den1, nu2, den2, w_B))
    if (nu1 @ w_B) / (den1 @ w_B) < (nu2 @ w_B) / (den2 @ w_B):
        nu1, den1, nu2, den2 = nu2, den2, nu1, den1
    lo_t = min(min_ratio_in_ball(nu1, den1, w_B, 1.0, allowed)[0], min_ratio_in_ball(nu2, den2, w_B, 1.0, allowed)[0])
    hi_t = max(min_ratio_in_ball(nu1, den1, w_B, 1.0, allowed, True)[0], min_ratio_in_ball(nu2, den2, w_B, 1.0, allowed, True)[0])
    ts = np.linspace(lo_t, hi_t, n_t)
    vals = [_rho(t, nu1, den1, nu2, den2, w_B, allowed) for t in ts]
    rho = np.array([v[0] for v in vals])
    if not np.isfinite(rho).any():
        return float("inf"), None, dict(eps_t=float(ts[1] - ts[0]))
    best_r, best_w, best_t = np.inf, None, None
    for k in np.argsort(rho)[:3]:
        if not np.isfinite(rho[k]):
            continue
        a, b = ts[max(k - 1, 0)], ts[min(k + 1, n_t - 1)]
        f = lambda t: min(_rho(t, nu1, den1, nu2, den2, w_B, allowed)[0], 10.0)  # noqa: E731
        res = minimize_scalar(f, bounds=(a, b), method="bounded", options=dict(xatol=1e-9))
        for t in (ts[k], res.x):
            r, w = _rho(t, nu1, den1, nu2, den2, w_B, allowed)
            if r < best_r:
                best_r, best_w, best_t = r, w, t
    R1 = (nu1 @ best_w) / (den1 @ best_w); R2 = (nu2 @ best_w) / (den2 @ best_w)
    tv = 0.5 * np.abs(best_w - w_B).sum()
    assert R1 <= R2 + 1e-7 and tv <= best_r + 1e-7, (R1, R2, tv, best_r)
    return float(best_r), best_w, dict(eps_t=float(ts[1] - ts[0]), t=float(best_t))


def r_star_ratio_lower(nu1, den1, nu2, den2, w_B, allowed=None, tol=1e-6):
    """Certified LOWER bound on r* for distinct denominators: a flip at radius r needs
    min_{Ball(r)} R1 <= max_{Ball(r)} R2 (necessary, not sufficient). Both sides are exact
    Charnes-Cooper LPs and monotone in r, so bisection gives r_lower <= r* exactly (up to tol)."""
    nu1, den1, nu2, den2, w_B = (np.asarray(x, float) for x in (nu1, den1, nu2, den2, w_B))
    if (nu1 @ w_B) / (den1 @ w_B) < (nu2 @ w_B) / (den2 @ w_B):
        nu1, den1, nu2, den2 = nu2, den2, nu1, den1
    gap = lambda r: min_ratio_in_ball(nu1, den1, w_B, r, allowed)[0] - min_ratio_in_ball(nu2, den2, w_B, r, allowed, True)[0]  # noqa: E731
    if gap(1.0) > 0:
        return float("inf")
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return lo


# ------------------------------------------------------------------ Theorem 1(b) bounds
def ratio_bounds_lp(N_S, D_S, w_U, polys):
    """Sharp bounds of M = (N_S + sum_h w_U[h] x_h) / (D_S + sum_h w_U[h] y_h) where for each
    unsupported (or set-valued) stratum h the per-frame pair (x_h, y_h) ranges over the polytope
    {A_h [x, y] <= b_h} (polys[h] = (A_h, b_h)). N_S, D_S are the known supported parts (already
    multiplied by the weights). Linear-fractional in (x, y) over a product polytope -> exact via
    Charnes-Cooper; the identified set is the whole interval [lo, hi] (connected image).
    Returns (lo, hi)."""
    H = len(w_U)
    out = []
    for sign in (1, -1):
        # variables: X_h = s*x_h, Y_h = s*y_h, s ; objective sign*(N_S s + sum w X)
        n = 2 * H + 1
        c = np.zeros(n); c[:H] = sign * np.asarray(w_U); c[-1] = sign * N_S
        Aeq = np.zeros((1, n)); Aeq[0, H:2 * H] = w_U; Aeq[0, -1] = D_S
        Aub, bub = [], []
        for h, (A, b) in enumerate(polys):
            A = np.atleast_2d(A)
            for k in range(A.shape[0]):
                row = np.zeros(n); row[h] = A[k, 0]; row[H + h] = A[k, 1]; row[-1] = -b[k]
                Aub.append(row); bub.append(0)
        res = linprog(c, A_ub=np.array(Aub) if Aub else None, b_ub=bub if bub else None,
                      A_eq=Aeq, b_eq=[1], bounds=[(None, None)] * (2 * H) + [(0, None)], method="highs")
        if res.status != 0:
            raise RuntimeError(res.message)
        out.append(sign * res.fun)
    return out[0], out[1]


# ------------------------------------------------------------------ self-test
def _brute_linear(delta, w_B, r, allowed, rng, n=20000):
    """min of Delta over random points of Ball(r) (inner approximation) + all 2-stratum moves."""
    H = len(delta)
    a = _allowed(H, allowed)
    best = w_B @ delta
    for i in range(H):
        for j in range(H):
            if i != j and a[j]:
                m = min(r, w_B[i])
                w = w_B.copy(); w[i] -= m; w[j] += m
                best = min(best, w @ delta)
    for _ in range(n):
        w = rng.dirichlet(np.ones(H)) * a
        w /= w.sum()
        tv = 0.5 * np.abs(w - w_B).sum()
        lam = min(1.0, r / tv) if tv > 0 else 1.0
        w = w_B + lam * (w - w_B)
        best = min(best, w @ delta)
    return best


def _selftest():
    rng = np.random.default_rng(SEED)
    rep = {}
    # equality example (theory notes §4): w_B=(.5,.5), delta=(3,-1): Delta_B=1, r*=1/4, w*=(.25,.75)
    r, w = r_star_linear([3, -1], [.5, .5])
    assert abs(r - 0.25) < 1e-12 and np.allclose(w, [.25, .75]) and abs(np.dot(w, [3, -1])) < 1e-12
    assert abs(r_star_closed_form([3, -1], [.5, .5]) - 0.25) < 1e-12
    rep["equality_example"] = dict(r_star=r, w=w.tolist())
    # closed form is only a lower bound when w_B(argmax) is small
    r2, _ = r_star_linear([10, 1, -1], [.05, .95, 0])
    cf2 = r_star_closed_form([10, 1, -1], [.05, .95, 0])
    assert abs(r2 - 0.5) < 1e-12 and abs(cf2 - 1.45 / 11) < 1e-12
    rep["closed_form_not_tight_example"] = dict(r_star=r2, closed_form=cf2)
    # random instances: (1) at r*-eps no flip found by brute force, at r*+eps the greedy w flips;
    # (2) closed form <= exact r*
    n_ok = 0
    for _ in range(200):
        H = rng.integers(2, 7)
        w_B = rng.dirichlet(np.ones(H))
        delta = rng.normal(size=H)
        allowed = rng.random(H) < 0.8
        allowed[rng.integers(H)] = True
        r, _ = r_star_linear(delta, w_B, allowed)
        if not np.isfinite(r) or r > 1:
            continue
        sgn = np.sign(w_B @ delta)
        vmin = _brute_linear(sgn * delta, w_B, max(r - 1e-3, 0), allowed, rng, 2000)
        assert vmin > -1e-9, (vmin, r)
        v, _ = min_linear_in_ball(sgn * delta, w_B, r + 1e-3, allowed)
        assert v <= 1e-9
        assert r_star_closed_form(delta, w_B, allowed) <= r + 1e-12
        n_ok += 1
    rep["linear_random_instances_ok"] = n_ok
    # Charnes-Cooper min ratio vs brute force
    worst = 0.0
    for _ in range(50):
        H = rng.integers(2, 6)
        w_B = rng.dirichlet(np.ones(H)); nu = rng.random(H); den = rng.random(H) + 0.1
        r = rng.uniform(0.05, 0.6)
        v, w = min_ratio_in_ball(nu, den, w_B, r)
        assert 0.5 * np.abs(w - w_B).sum() <= r + 1e-7
        for _k in range(3000):
            x = rng.dirichlet(np.ones(H)); tv = 0.5 * np.abs(x - w_B).sum()
            x = w_B + min(1, r / tv) * (x - w_B)
            worst = max(worst, v - (nu @ x) / (den @ x))
    assert worst < 1e-7, worst
    rep["charnes_cooper_bruteforce_max_violation"] = worst
    # shared denominator ratio == linear on numerator difference
    w_B = np.array([.3, .3, .4]); nu1 = np.array([.9, .2, .5]); nu2 = np.array([.5, .6, .45]); den = np.array([1, .8, .6])
    rs, w = r_star_ratio_shared_den(nu1, nu2, den, w_B)
    assert abs((nu1 @ w) / (den @ w) - (nu2 @ w) / (den @ w)) < 1e-9
    rep["shared_den_example"] = dict(r_star=rs)
    # general ratio: certificate check + compare with dense brute force on 3-stratum instances
    diffs = []
    for _ in range(5):
        H = 3
        w_B = rng.dirichlet(np.ones(H)); nu1 = rng.random(H); den1 = nu1 + rng.random(H)
        nu2 = rng.random(H); den2 = nu2 + rng.random(H)
        rg, w, info = r_star_ratio_general(nu1, den1, nu2, den2, w_B, n_t=101)
        # brute force r*: fine grid over the simplex
        g = np.linspace(0, 1, 301)
        W = np.array([(a, b, 1 - a - b) for a in g for b in g if a + b <= 1 + 1e-12])
        R1 = (W @ nu1) / (W @ den1); R2 = (W @ nu2) / (W @ den2)
        s = np.sign((w_B @ nu1) / (w_B @ den1) - (w_B @ nu2) / (w_B @ den2))
        flip = s * (R1 - R2) <= 0
        tv = 0.5 * np.abs(W - w_B).sum(1)
        rb = tv[flip].min() if flip.any() else np.inf
        rl = r_star_ratio_lower(nu1, den1, nu2, den2, w_B)
        assert rl <= rg + 1e-6
        diffs.append((rg, rb, rl))
        assert rg <= rb + 1e-9 or not np.isfinite(rb)   # LP optimum over the continuous simplex <= grid optimum
        if np.isfinite(rg):
            assert s * ((nu1 @ w) / (den1 @ w) - (nu2 @ w) / (den2 @ w)) <= 1e-7
    rep["general_ratio_vs_grid_bruteforce"] = [dict(upper_lp=a, grid=b, certified_lower=c) for a, b, c in diffs]
    # bounds LP: processed fraction with one unknown stratum x in [a,b], y = 1
    lo, hi = ratio_bounds_lp(0.3, 0.6, [0.4], [(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]), np.array([1, -1 / 16, 1, -1]))])
    assert abs(lo - (0.3 + 0.4 / 16)) < 1e-9 and abs(hi - 0.7) < 1e-9
    rep["bounds_example"] = dict(lo=lo, hi=hi)
    return rep


if __name__ == "__main__":
    import json
    print(json.dumps(_selftest(), indent=1, default=float))
