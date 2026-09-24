# trade-optimize

Markowitz portfolio optimization for the **trade-suite**: pure-Python
mean-variance, maximum-Sharpe, efficient-frontier, and risk-parity
optimizers with zero third-party dependencies.

```
max_sharpe portfolio (6 assets):
    ASSET1   63.58%
    ASSET4   11.98%
    ASSET6   24.44%
expected return 32.59%  vol 15.28%  sharpe 2.13
```

Part of the [trade-suite](https://github.com/crieck2010/trade-suite) algorithmic
and agentic trading system. Research/backtesting/paper-trading only — never live
trading, never personalized investment advice.

## Install

```bash
pip install git+https://github.com/crieck2010/trade-optimize.git
```

Requires Python 3.10+. Stdlib-only.

## Quick start

```bash
# optimal portfolio from the synthetic demo universe
trade-optimize optimize --method max_sharpe

# trace the efficient frontier
trade-optimize frontier --points 12

# from your own returns CSV (wide format: date,AAA,BBB,...)
trade-optimize optimize --method min_variance --csv returns.csv --max-weight 0.35

# agent-ready allocation JSON for the trade-agents PM
trade-optimize allocate --method risk_parity
```

```python
from trade_optimize import estimates, optimize, analytics

rets = [[0.01, 0.02], [0.02, -0.01], [-0.01, 0.015], ...]  # T x N
mu = estimates.mean_returns(rets)
sigma = estimates.shrink_covariance(rets, delta=0.2)

w = optimize(sigma, mu, method="max_sharpe", max_weight=0.4)
print(analytics.portfolio_stats(w, mu, sigma))
print(analytics.risk_contributions(w, sigma))
```

## Methods

| Method | What it does |
|---|---|
| `min_variance` | Global minimum-variance portfolio (γ = 0) |
| `max_sharpe` | Long-only tangency portfolio (max Sharpe on the frontier) |
| `risk_parity` | Equal risk contribution per asset |
| `equal_weight` | 1/N baseline |

All are long-only with an optional per-asset `--max-weight` cap.

## What's inside

| Module | Purpose |
|---|---|
| `linalg.py` | Pure-Python Cholesky, solve, inverse, power iteration |
| `estimates.py` | Mean / sample / shrinkage / EWMA covariance estimators |
| `optimize.py` | Projected-gradient QP + the four optimizers + frontier |
| `analytics.py` | Portfolio stats, risk contributions, diversification ratio |
| `rebalance.py` | Target weights → concrete buy/sell orders |
| `adapters.py` | Lazy bridges to `trade-agents`, `trade-backtest`, `trade-paper`, `trade-risk` |

## Suite interop

| Sibling | Bridge |
|---|---|
| `trade-agents` | `to_agent_allocation()` — a smarter PM allocator than inverse-vol |
| `trade-backtest` | `to_backtest_weights()` — allocation block for the weight schedule |
| `trade-paper` | `to_paper_orders()` — target weights → approval-queue order intents |
| `trade-risk` | `portfolio_vol_for_risk()` + `risk_contributions()` for limit checks |
| `trade-data-*` | `returns_from_dict_bars()` — duck-typed bar ingestion, no imports |

See `docs/METHODOLOGY.md` for the math and the honest limitations
(estimation error dominates everything — that's why shrinkage is on by default).
