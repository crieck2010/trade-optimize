# Architecture — trade-optimize

## Layout

```
src/trade_optimize/
  __init__.py     public API surface (versioned exports)
  __main__.py     python -m trade_optimize
  linalg.py       Cholesky / solve / invert / power iteration (lists of lists)
  estimates.py    mean, sample/shrinkage/EWMA covariance estimators
  optimize.py     projected-gradient QP + 4 optimizers + frontier sweep
  analytics.py    portfolio stats, risk contributions, diversification
  rebalance.py    target weights → concrete orders
  adapters.py     lazy bridges to trade-agents / trade-backtest / trade-paper / trade-risk
  cli.py          optimize / frontier / allocate / license / update-check
  licensing.py    license-key + update-check hooks (suite convention)
```

## Design decisions

- **Stdlib-only numerics.** Matrices are lists of lists; `linalg.py` is
  small and auditable on purpose — Cholesky doubles as the PD check, and
  power iteration supplies the QP step size. No NumPy means zero install
  friction and identical behaviour everywhere.
- **Estimators are pure functions** over T×N return matrices. Swapping in a
  sibling's estimator (or a future Ledoit-Wolf) changes one call site.
- **One QP solver, four optimizers.** min-variance, max-Sharpe, and the
  frontier are all the same projected-gradient core with different γ; risk
  parity is a separate fixed-point loop. One well-tested hot path.
- **Plain data at every boundary.** Weights are lists, allocations are
  dicts, stats are JSON-safe dicts. `dicts_to_matrix` / `returns_from_dict_bars`
  ingest duck-typed bars without importing sibling packages.
- **Lazy interop.** `adapters.py` imports siblings inside functions only:
  - `to_agent_allocation` → the trade-agents PM's allocator (weights +
    stats + risk contributions);
  - `to_backtest_weights` → allocation block for trade-backtest's schedule;
  - `to_paper_orders` → approval-queue intents for trade-paper;
  - `portfolio_vol_for_risk` + `risk_contributions` → trade-risk limit inputs.

## Scaling notes

- Pure-Python QP is O(n²) per iteration; comfortable to a few hundred
  assets. Beyond that, the `optimize()` signatures are backend-agnostic —
  a NumPy/CVXPY core can drop in without changing callers.
- Frontier sweeps are embarrassingly parallel: each γ point is independent.
  `efficient_frontier` is deterministic, so sharding γ across processes and
  merging the point lists is trivially correct.
- Annualization is a reporting concern only (`periods_per_year`); the
  optimizer works in whatever per-period units the returns arrive in.
