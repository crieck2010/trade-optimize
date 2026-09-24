"""Tests for trade-optimize."""

from __future__ import annotations

import json
import math

import pytest

from trade_optimize import (
    efficient_frontier,
    equal_weight,
    max_sharpe,
    min_variance,
    optimize,
    risk_parity,
    trades_to_target,
)
from trade_optimize import analytics, estimates, linalg
from trade_optimize.adapters import (
    dicts_to_matrix,
    returns_from_dict_bars,
    to_agent_allocation,
    to_backtest_weights,
)


# ------------------------------------------------------------------ linalg

def test_cholesky_known():
    A = [[4.0, 2.0], [2.0, 3.0]]
    L = linalg.cholesky(A)
    assert L[0][0] == pytest.approx(2.0)
    assert L[1][0] == pytest.approx(1.0)
    assert L[1][1] == pytest.approx(math.sqrt(2.0))
    # L·Lᵀ reconstructs A
    Lt = linalg.transpose(L)
    R = linalg.mat_mat(L, Lt)
    for i in range(2):
        for j in range(2):
            assert R[i][j] == pytest.approx(A[i][j])


def test_cholesky_rejects_non_pd():
    with pytest.raises(ValueError):
        linalg.cholesky([[1.0, 2.0], [2.0, 1.0]])


def test_solve():
    A = [[2.0, 1.0], [1.0, 3.0]]
    x = linalg.solve(A, [5.0, 6.0])
    assert x == pytest.approx([1.8, 1.4])


def test_invert():
    A = [[2.0, 1.0], [1.0, 3.0]]
    inv = linalg.invert(A)
    I = linalg.mat_mat(A, inv)
    for i in range(2):
        for j in range(2):
            assert I[i][j] == pytest.approx(1.0 if i == j else 0.0)


def test_max_eigenvalue():
    assert linalg.max_eigenvalue([[3.0, 0.0], [0.0, 1.0]]) == pytest.approx(3.0, rel=1e-6)


# --------------------------------------------------------------- estimates

RETS = [
    [0.01, 0.02, -0.01],
    [0.02, -0.01, 0.03],
    [-0.01, 0.01, 0.02],
    [0.03, 0.02, -0.02],
    [0.00, -0.02, 0.01],
]


def test_mean_returns():
    mu = estimates.mean_returns(RETS)
    assert mu[0] == pytest.approx(sum(r[0] for r in RETS) / 5)


def test_sample_covariance_symmetry_and_known():
    s = estimates.sample_covariance(RETS)
    n = len(s)
    for i in range(n):
        for j in range(n):
            assert s[i][j] == pytest.approx(s[j][i])
    # variance of asset 0 by hand
    m0 = sum(r[0] for r in RETS) / 5
    v0 = sum((r[0] - m0) ** 2 for r in RETS) / 4
    assert s[0][0] == pytest.approx(v0)
    assert all(s[i][i] > 0 for i in range(n))


def test_shrink_covariance_kills_off_diagonal():
    s = estimates.shrink_covariance(RETS, delta=1.0)
    n = len(s)
    for i in range(n):
        for j in range(n):
            assert s[i][j] == pytest.approx(0.0 if i != j else s[i][i])
    raw = estimates.sample_covariance(RETS)
    assert s[0][0] == pytest.approx(raw[0][0])  # variances untouched


def test_shrink_mean_toward_grand():
    mu = estimates.shrink_mean(RETS, delta=1.0)
    grand = sum(estimates.mean_returns(RETS)) / 3
    assert all(m == pytest.approx(grand) for m in mu)


def test_ewma_weights_recent_more():
    # a shock in the last row should move EWMA var more than sample var
    calm = [[0.001, -0.001]] * 20
    shock = calm + [[0.10, -0.10]]
    e = estimates.ewma_covariance(shock, lam=0.94)
    s = estimates.sample_covariance(shock)
    assert e[0][0] > s[0][0]


# --------------------------------------------------------------- optimize

def test_min_variance_two_asset_closed_form():
    # σ1=0.2, σ2=0.1, uncorrelated → w1 = σ2²/(σ1²+σ2²) = 0.2
    sigma = [[0.04, 0.0], [0.0, 0.01]]
    w = min_variance(sigma)
    assert w[0] == pytest.approx(0.2, abs=1e-4)
    assert w[1] == pytest.approx(0.8, abs=1e-4)


def test_min_variance_correlated_closed_form():
    # ρ=0.5: w1 = (σ2² − ρσ1σ2)/(σ1² + σ2² − 2ρσ1σ2)
    s1, s2, rho = 0.2, 0.1, 0.5
    cov = rho * s1 * s2
    sigma = [[s1**2, cov], [cov, s2**2]]
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    w = min_variance(sigma)
    assert w[0] == pytest.approx(w1, abs=1e-4)


def test_weights_sum_to_one_and_nonneg():
    sigma = estimates.shrink_covariance(RETS)
    mu = estimates.mean_returns(RETS)
    for w in (min_variance(sigma), max_sharpe(sigma, mu),
              risk_parity(sigma), equal_weight(3)):
        assert sum(w) == pytest.approx(1.0, abs=1e-6)
        assert all(x >= -1e-9 for x in w)


def test_max_weight_cap_respected():
    sigma = estimates.shrink_covariance(RETS)
    mu = estimates.mean_returns(RETS)
    for w in (min_variance(sigma, max_weight=0.4),
              max_sharpe(sigma, mu, max_weight=0.4),
              risk_parity(sigma, max_weight=0.4)):
        assert max(w) <= 0.4 + 1e-9


def test_max_sharpe_beats_baselines():
    sigma = estimates.shrink_covariance(RETS)
    mu = estimates.mean_returns(RETS)
    rf = 0.0

    def sharpe(w):
        st = analytics.portfolio_stats(w, mu, sigma, rf)
        return st["sharpe"]

    assert sharpe(max_sharpe(sigma, mu)) >= sharpe(min_variance(sigma)) - 1e-9
    assert sharpe(max_sharpe(sigma, mu)) >= sharpe(equal_weight(3)) - 1e-9


def test_frontier_monotonic():
    sigma = estimates.shrink_covariance(RETS)
    mu = estimates.mean_returns(RETS)
    pts = efficient_frontier(sigma, mu, n_points=10)
    ers = [p["expected_return"] for p in pts]
    vols = [p["volatility"] for p in pts]
    assert all(b >= a - 1e-9 for a, b in zip(ers, ers[1:]))
    assert all(b >= a - 1e-9 for a, b in zip(vols, vols[1:]))
    assert pts[0]["volatility"] <= pts[-1]["volatility"]
    json.dumps(pts)  # JSON-safe


def test_risk_parity_equalizes_risk():
    sigma = [[0.04 if i == j == 0 else 0.01 if i == j == 1 else
              0.0025 if i == j == 2 else 0.0 for j in range(3)] for i in range(3)]
    w = risk_parity(sigma)
    rc = analytics.risk_contributions(w, sigma)
    shares = [c["risk_share"] for c in rc]
    assert max(shares) - min(shares) < 0.01
    # lower-vol asset gets the bigger weight
    assert w[2] > w[1] > w[0]


def test_equal_weight_baseline():
    w = equal_weight(4)
    assert w == pytest.approx([0.25] * 4)


def test_optimize_dispatcher():
    sigma = estimates.shrink_covariance(RETS)
    mu = estimates.mean_returns(RETS)
    w = optimize(sigma, mu, method="max_sharpe")
    assert sum(w) == pytest.approx(1.0, abs=1e-6)
    assert optimize(sigma, method="min_variance") == pytest.approx(min_variance(sigma))
    with pytest.raises(ValueError):
        optimize(sigma, mu, method="nope")
    with pytest.raises(ValueError):
        optimize(sigma, method="max_sharpe")  # mu required


def test_infeasible_cap_rejected():
    with pytest.raises(ValueError):
        min_variance([[0.04, 0.0], [0.0, 0.01]], max_weight=0.4)


# --------------------------------------------------------------- analytics

def test_risk_contributions_sum_to_variance():
    sigma = estimates.shrink_covariance(RETS)
    w = max_sharpe(sigma, estimates.mean_returns(RETS))
    rc = analytics.risk_contributions(w, sigma)
    var = sum(c["risk_contribution"] for c in rc)
    port_var = sum(wi * s for wi, s in
                   zip(w, linalg.mat_vec(sigma, w)))
    assert var == pytest.approx(port_var)
    assert sum(c["risk_share"] for c in rc) == pytest.approx(1.0)


def test_herfindahl_bounds():
    assert analytics.herfindahl([0.25] * 4) == pytest.approx(0.25)
    assert analytics.herfindahl([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_portfolio_stats_annualization():
    mu = [0.001, 0.002]
    sigma = [[0.0004, 0.0], [0.0, 0.0009]]
    st = analytics.portfolio_stats([0.5, 0.5], mu, sigma, periods_per_year=252)
    assert st["expected_return_ann"] == pytest.approx(0.0015 * 252)
    assert st["volatility_ann"] == pytest.approx(st["volatility"] * math.sqrt(252))
    assert st["n_held"] == 2


def test_diversification_ratio():
    sigma = [[0.04, 0.0], [0.0, 0.01]]
    dr = analytics.diversification_ratio([0.5, 0.5], sigma)
    assert dr > 1.0  # uncorrelated assets diversify


# --------------------------------------------------------------- rebalance

def test_trades_to_target():
    orders = trades_to_target(
        {"AAA": 10.0}, {"AAA": 0.5, "BBB": 0.5},
        {"AAA": 100.0, "BBB": 50.0}, capital=10_000.0)
    by_sym = {o["symbol"]: o for o in orders}
    # AAA: target 50 sh, hold 10 → buy 40 ; BBB: target 100 sh → buy 100
    assert by_sym["AAA"]["side"] == "buy"
    assert by_sym["AAA"]["quantity"] == pytest.approx(40.0)
    assert by_sym["BBB"]["quantity"] == pytest.approx(100.0)


def test_trades_to_target_liquidates_dropped_names():
    orders = trades_to_target({"AAA": 10.0, "ZZZ": 5.0}, {"AAA": 1.0},
                              {"AAA": 100.0, "ZZZ": 20.0}, capital=10_000.0)
    by_sym = {o["symbol"]: o for o in orders}
    assert by_sym["ZZZ"]["side"] == "sell"
    assert by_sym["ZZZ"]["target_weight"] == 0.0


def test_trades_to_target_validates():
    with pytest.raises(ValueError):
        trades_to_target({}, {"AAA": 0.6, "BBB": 0.6},
                         {"AAA": 1.0, "BBB": 1.0}, capital=100.0)


# ---------------------------------------------------------------- adapters

def test_dicts_to_matrix():
    syms, m = dicts_to_matrix([{"B": 0.02, "A": 0.01}, {"A": 0.03, "B": -0.01}])
    assert syms == ["A", "B"]
    assert m == [[0.01, 0.02], [0.03, -0.01]]


def test_returns_from_dict_bars():
    bars = [
        {"symbol": "AAA", "close": 100.0}, {"symbol": "AAA", "close": 110.0},
        {"symbol": "AAA", "close": 121.0},
        {"symbol": "BBB", "close": 50.0}, {"symbol": "BBB", "close": 50.0},
        {"symbol": "BBB", "close": 55.0},
    ]
    syms, m = returns_from_dict_bars(bars, ["AAA", "BBB"])
    assert syms == ["AAA", "BBB"]
    assert m[0][0] == pytest.approx(0.10)
    assert m[1][0] == pytest.approx(0.10)
    assert m[0][1] == pytest.approx(0.0)
    assert m[1][1] == pytest.approx(0.10)


def test_to_agent_allocation():
    syms = ["A", "B", "C"]
    alloc = to_agent_allocation(syms, RETS, method="max_sharpe")
    assert alloc["source"] == "trade-optimize"
    assert set(alloc["weights"]) == set(syms)
    assert sum(alloc["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    assert len(alloc["risk_contributions"]) == 3
    json.dumps(alloc)


def test_to_backtest_weights():
    out = to_backtest_weights(["A", "B", "C"], RETS, method="min_variance")
    assert set(out["weights"]) == {"A", "B", "C"}
    assert out["source"] == "trade-optimize"
